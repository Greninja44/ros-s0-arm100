#!/usr/bin/env python3
"""Octo VLA policy node for the SO-100 arm.

The node subscribes to the (simulated) camera and ``/joint_states``, runs a
pretrained Octo model to predict an action chunk from a natural-language
instruction, and sends it to the ``arm_controller`` via the
``FollowJointTrajectory`` action.

Pipeline
--------
image + joint_states --[observation]--> Octo --[action chunk]--> arm_controller

Notes
-----
* Octo is loaded lazily on the first observation so the node starts fast and
  the ROS pipeline can be validated without a checkpoint.
* ``mock:=true`` skips Octo entirely and streams a small scripted motion so the
  control loop can be tested without a GPU / torch.
* The model is trained for Cartesian end-effector actions (e.g. Bridge), so the
  sampled chunk is a Cartesian delta. For true joint-space control replace the
  action -> joint mapping below with inverse kinematics or a joint-space policy.
"""

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter


def _create_action_client(node, action_type, action_name):
    """Compat helper: some rclpy builds lack Node.create_action_client()."""
    if hasattr(node, "create_action_client"):
        return node.create_action_client(action_type, action_name)
    return ActionClient(node, action_type, action_name)

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectoryPoint

ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper_joint",
]


def image_to_numpy(msg):
    """Decode a raw sensor_msgs/Image (bgr8/rgb8/mono8/32FC1) to an HWC array."""
    if msg.encoding in ("bgr8", "rgb8"):
        dtype, channels = np.uint8, 3
    elif msg.encoding == "mono8":
        dtype, channels = np.uint8, 1
    elif msg.encoding == "32FC1":
        dtype, channels = np.float32, 1
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    arr = np.frombuffer(msg.data, dtype=dtype).reshape(
        msg.height, msg.width, channels
    )
    if msg.encoding == "bgr8":
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


class OctoPolicyNode(Node):
    def __init__(self):
        super().__init__("octo_policy")

        # ---- Parameters -----------------------------------------------------
        self.set_parameters([Parameter("use_sim_time", value=True)])
        self.instruction = self.declare_parameter(
            "instruction", "pick up the object and place it down"
        ).value
        self.model_id = self.declare_parameter(
            "model_id", "hf://rail-berkeley/octo-small"
        ).value
        self.dataset = self.declare_parameter("dataset", "octo").value
        self.dataset_name = self.declare_parameter(
            "dataset_name", "bridge_orig"
        ).value
        self.action_horizon = self.declare_parameter("action_horizon", 50).value
        self.control_frequency = self.declare_parameter(
            "control_frequency", 5.0
        ).value
        self.execution_duration = self.declare_parameter(
            "execution_duration", 1.0
        ).value
        self.state_dim = self.declare_parameter("state_dim", 9).value
        self.image_topic = self.declare_parameter(
            "image_topic", "/image"
        ).value
        self.mock = self.declare_parameter("mock", False).value

        # ---- Subscribers ----------------------------------------------------
        self.create_subscription(Image, self.image_topic, self._image_cb, 1)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 1)

        # ---- Action client for the arm controller ---------------------------
        self._action_client = _create_action_client(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )

        # ---- State ----------------------------------------------------------
        self._latest_image = None
        self._latest_joints = None
        self._obs_lock = threading.Lock()

        self._model = None
        self._rng = None
        self._task = None
        self._unnorm_stats = None
        self._logged_waiting = False

        self.create_timer(1.0 / self.control_frequency, self._control_loop)

    # ------------------------------------------------------------------ #
    # Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _image_cb(self, msg):
        try:
            image = image_to_numpy(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return
        with self._obs_lock:
            self._latest_image = image

    def _joint_cb(self, msg):
        values = [0.0] * len(ARM_JOINTS)
        for name, value in zip(msg.name, msg.position):
            if name in ARM_JOINTS:
                values[ARM_JOINTS.index(name)] = value
        with self._obs_lock:
            self._latest_joints = np.array(values, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Octo                                                                #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        try:
            from octo.model.octo_model import OctoModel
        except ImportError:
            self.get_logger().fatal(
                "Octo is not installed. Install it with 'pip install octo' "
                "or run with mock:=true to test the ROS pipeline."
            )
            return False

        self.get_logger().info(f"Loading Octo model {self.model_id} ...")
        self._model = OctoModel.load_pretrained(self.model_id)
        self._rng = self._model.create_training_rng()
        self._task = self._model.create_tasks(texts=[self.instruction])
        self._unnorm_stats = self._model.dataset_statistics[self.dataset][
            self.dataset_name
        ]
        self.get_logger().info("Octo model ready.")
        return True

    def _build_observation(self, image, joints):
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[: len(joints)] = joints
        return {"image_primary": image, "state": state}

    def _sample_actions(self, observation):
        """Returns an (action_horizon, action_dim) numpy array."""
        if self.mock:
            return self._mock_actions()
        actions = self._model.sample_actions(
            observation,
            self._task,
            rng=self._rng,
            unnormalization_statistics=self._unnorm_stats,
        )
        return np.asarray(actions).astype(np.float32)

    def _mock_actions(self):
        """Small scripted motion to exercise the control loop without Octo."""
        base = np.array(self._latest_joints, dtype=np.float32)
        t = np.arange(self.action_horizon) / self.control_frequency
        actions = np.zeros((self.action_horizon, len(ARM_JOINTS)), dtype=np.float32)
        for j in range(len(ARM_JOINTS)):
            actions[:, j] = base[j] + 0.05 * np.sin(t + j)
        actions[:, 5] = base[5] + 0.2 * np.sin(t / 2.0)
        return actions

    # ------------------------------------------------------------------ #
    # Control loop                                                        #
    # ------------------------------------------------------------------ #

    def _control_loop(self):
        with self._obs_lock:
            image = self._latest_image
            joints = self._latest_joints
        if image is None or joints is None:
            if not self._logged_waiting:
                self.get_logger().info("Waiting for image and joint_states ...")
                self._logged_waiting = True
            return

        if self._model is None and not self.mock:
            if not self._load_model():
                return

        observation = self._build_observation(image, joints)
        try:
            actions = self._sample_actions(observation)
        except Exception as exc:  # noqa: BLE001 - model errors should not kill the node
            self.get_logger().error(f"Policy inference failed: {exc}")
            return

        self._send_actions(actions)

    def _send_actions(self, actions):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("arm_controller not available, dropping chunk")
            return

        horizon = actions.shape[0]
        action_dim = actions.shape[1]
        step = self.execution_duration / horizon

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_JOINTS)

        for t in range(horizon):
            point = JointTrajectoryPoint()
            # Map the first action_dim columns onto the arm joints.
            point.positions = [float(v) for v in actions[t, :action_dim]]
            # Hold the remaining joints (e.g. gripper if policy has no gripper).
            for j in range(action_dim, len(ARM_JOINTS)):
                point.positions.append(float(self._latest_joints[j]))
            point.time_from_start = Duration(
                sec=int((t + 1) * step),
                nanosec=int(((t + 1) * step) % 1 * 1e9),
            )
            goal.trajectory.points.append(point)

        self.get_logger().info(
            f"Sending {horizon}-step action chunk (dim {action_dim}) to arm_controller"
        )
        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_feedback)

    def _goal_feedback(self, future):
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Action chunk rejected by arm_controller")
            return
        result_future = goal_handle.get_result_async()

        def _done(result_future):
            status = result_future.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info("Action chunk executed")
            else:
                self.get_logger().warn(f"Action chunk finished with status {status}")

        result_future.add_done_callback(_done)


def main():
    rclpy.init()
    node = OctoPolicyNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001 - context may already be shutting down
            pass


if __name__ == "__main__":
    main()
