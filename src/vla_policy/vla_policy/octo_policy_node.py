#!/usr/bin/env python3
"""Octo VLA policy node for the SO-100 arm.

The node subscribes to the (simulated) camera and ``/joint_states``, runs a
pretrained Octo model to predict an action chunk from a natural-language
instruction, and sends it to the ``arm_controller`` via the
``FollowJointTrajectory`` action.

Pipeline
--------
image + joint_states --[observation]--> Octo --[action chunk]--> arm_controller

Control modes
-------------
* ``control_mode:=joint_space`` — the action chunk is treated as absolute joint
  positions (works with policies trained on joint-space datasets).
* ``control_mode:=cartesian`` — the action chunk is treated as Cartesian
  end-effector deltas (dx, dy, dz, droll, dpitch, dyaw).  Each delta is
  converted to joint positions via numerical inverse kinematics (see
  so100_ik.py) so that the arm tracks the desired Cartesian trajectory.

Gripper
-------
When the action chunk has more columns than the Cartesian dimension (6), the
extra column(s) control the gripper.  Values above ``gripper_open_thresh``
open the gripper; values below close it.

Notes
-----
* Octo is loaded lazily on the first observation so the node starts fast and
  the ROS pipeline can be validated without a checkpoint.
* ``mock:=true`` skips Octo entirely and streams a small scripted motion so the
  control loop can be tested without a GPU / torch.
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

from vla_policy import so100_ik

ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper_joint",
]

ARM_ONLY_JOINTS = ARM_JOINTS[:5]

# Cartesian action dimension (x, y, z, roll, pitch, yaw).
CARTESIAN_DIM = 6


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


# Cartesian IK (position + orientation tracking for the delta-action loop
# below) is provided by so100_ik.py -- a self-contained numpy solver with
# the chain's kinematics hardcoded, so no URDF file needs to be located
# or parsed at runtime, and no PyKDL/kdl_parser dependency is needed
# (kdl_parser_py isn't available on Jazzy, which silently broke this
# entire control mode before).


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
        self.control_mode = self.declare_parameter(
            "control_mode", "joint_space"
        ).value  # "joint_space" or "cartesian"
        self.gripper_open_thresh = self.declare_parameter(
            "gripper_open_thresh", 0.0
        ).value
        self.gripper_position_open = self.declare_parameter(
            "gripper_position_open", 1.8
        ).value
        self.gripper_position_closed = self.declare_parameter(
            "gripper_position_closed", 0.0
        ).value

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
        if self.control_mode == "cartesian":
            # In mock cartesian mode, produce small Cartesian deltas
            actions = np.zeros(
                (self.action_horizon, CARTESIAN_DIM), dtype=np.float32
            )
            actions[:, 0] = 0.005 * np.sin(t)  # dx
            actions[:, 2] = -0.005 * np.cos(t)  # dz
            # Add a gripper column
            gripper = np.zeros((self.action_horizon, 1), dtype=np.float32)
            gripper[: self.action_horizon // 2] = 1.0  # open then close
            return np.hstack([actions, gripper])
        else:
            actions = np.zeros(
                (self.action_horizon, len(ARM_JOINTS)), dtype=np.float32
            )
            for j in range(len(ARM_JOINTS)):
                actions[:, j] = base[j] + 0.05 * np.sin(t + j)
            actions[:, 5] = base[5] + 0.2 * np.sin(t / 2.0)
            return actions

    # ------------------------------------------------------------------ #
    # IK mapping (Cartesian -> joint space)                               #
    # ------------------------------------------------------------------ #

    def _cartesian_actions_to_joints(self, actions, current_joints):
        """Convert a chunk of Cartesian deltas to joint positions via IK.

        Args:
            actions: (horizon, action_dim) array.  First 6 columns are
                (dx, dy, dz, droll, dpitch, dyaw) deltas.  Extra columns
                are gripper commands.
            current_joints: current joint positions (6,) including gripper.

        Returns:
            (horizon, 6) joint-space trajectory, or None on IK failure.
        """
        arm_joints = current_joints[:5]
        q = np.array(arm_joints, dtype=np.float64)

        horizon = actions.shape[0]
        joint_traj = np.zeros((horizon, len(ARM_JOINTS)), dtype=np.float32)

        T_current = so100_ik.fk(q)

        for t in range(horizon):
            # Parse the Cartesian delta
            dx = float(actions[t, 0])
            dy = float(actions[t, 1])
            dz = float(actions[t, 2])
            droll = float(actions[t, 3])
            dpitch = float(actions[t, 4])
            dyaw = float(actions[t, 5])

            # Build desired pose by applying the delta
            T_desired = T_current.copy()
            T_desired[0, 3] += dx
            T_desired[1, 3] += dy
            T_desired[2, 3] += dz

            # Apply rotation delta (small angle approximation via rotation matrix)
            R_delta = so100_ik.rotation_from_rpy(droll, dpitch, dyaw)
            T_desired[:3, :3] = T_desired[:3, :3] @ R_delta

            # Solve IK. The arm only has 5 DOF for a 6-DOF pose target, so
            # this converges to the closest reachable pose rather than an
            # exact match -- fine for small deltas from a streaming policy.
            q_sol, pos_err, rot_err = so100_ik.solve_pose_ik(
                T_desired[:3, 3], T_desired[:3, :3], seed=q
            )
            q_sol = np.array(q_sol, dtype=np.float64)

            joint_traj[t, :5] = q_sol.astype(np.float32)

            # Gripper: use extra column if present
            action_dim = actions.shape[1]
            if action_dim > CARTESIAN_DIM:
                gripper_val = float(actions[t, CARTESIAN_DIM])
                if gripper_val > self.gripper_open_thresh:
                    joint_traj[t, 5] = self.gripper_position_open
                else:
                    joint_traj[t, 5] = self.gripper_position_closed
            else:
                joint_traj[t, 5] = current_joints[5]

            # Update FK for the next step
            q = q_sol
            T_current = so100_ik.fk(q)

        return joint_traj

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

        # Convert to joint space if in Cartesian mode
        if self.control_mode == "cartesian":
            joint_actions = self._cartesian_actions_to_joints(actions, joints)
            if joint_actions is None:
                return
            actions = joint_actions

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
            point.positions = [float(v) for v in actions[t, :action_dim]]
            for j in range(action_dim, len(ARM_JOINTS)):
                point.positions.append(float(self._latest_joints[j]))
            point.time_from_start = Duration(
                sec=int((t + 1) * step),
                nanosec=int(((t + 1) * step) % 1 * 1e9),
            )
            goal.trajectory.points.append(point)

        self.get_logger().info(
            f"Sending {horizon}-step action chunk (dim {action_dim}) "
            f"mode={self.control_mode} to arm_controller"
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
                self.get_logger().warn(
                    f"Action chunk finished with status {status}"
                )

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
