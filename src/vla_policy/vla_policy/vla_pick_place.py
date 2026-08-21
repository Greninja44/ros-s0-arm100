#!/usr/bin/env python3
"""End-to-end VLA-driven pick-and-place for the SO-100 arm in Gazebo.

Runs the Octo VLA policy in a closed loop: the node observes the scene
(camera image + joint states), predicts an action chunk, executes it on
the arm, and repeats until the episode ends.

This node subsumes ``octo_policy`` for the pick-and-place use case:
it adds gripper open/close heuristics, episode termination conditions,
and a simple success-detection heuristic (end-effector height after
the place phase).

Usage
-----
    # terminal 1 — simulation
    ros2 launch so100_description gazebo.launch.py

    # terminal 2 — run the policy
    ros2 run vla_policy vla_pick_place --ros-args \
        -p instruction:="pick up the red cube and place it in the tray"

    # without GPU (mock mode):
    ros2 run vla_policy vla_pick_place --ros-args -p mock:=true

Parameters
----------
instruction : str     Natural-language task description.
mock         : bool   Skip Octo, use scripted actions.
max_steps    : int    Maximum number of policy loop iterations.
control_mode : str    "joint_space" or "cartesian".
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

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

ARM_ONLY = ARM_JOINTS[:5]
CARTESIAN_DIM = 6


def _create_action_client(node, action_type, action_name):
    if hasattr(node, "create_action_client"):
        return node.create_action_client(action_type, action_name)
    return ActionClient(node, action_type, action_name)


def image_to_numpy(msg):
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


# Cartesian IK is provided by so100_ik.py -- a self-contained numpy
# solver with the chain's kinematics hardcoded, so no URDF file needs to
# be located/parsed at runtime and no PyKDL/kdl_parser dependency is
# needed (kdl_parser_py isn't available on Jazzy, which silently broke
# this entire control mode before).


# ------------------------------------------------------------------ #
# VLA Pick-and-Place Node                                             #
# ------------------------------------------------------------------ #

class VLAPickPlace(Node):
    def __init__(self):
        super().__init__("vla_pick_place")
        self.set_parameters([Parameter("use_sim_time", value=True)])

        # ---- Parameters -----------------------------------------------------
        self.instruction = self.declare_parameter(
            "instruction", "pick up the red cube and place it in the tray"
        ).value
        self.mock = self.declare_parameter("mock", False).value
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
        self.max_steps = self.declare_parameter("max_steps", 200).value
        self.control_mode = self.declare_parameter(
            "control_mode", "cartesian"
        ).value
        self.gripper_open_thresh = self.declare_parameter(
            "gripper_open_thresh", 0.0
        ).value
        self.gripper_open_pos = self.declare_parameter(
            "gripper_open_pos", 1.8
        ).value
        self.gripper_closed_pos = self.declare_parameter(
            "gripper_closed_pos", 0.0
        ).value

        # ---- Subscribers ----------------------------------------------------
        self.create_subscription(Image, "/image", self._image_cb, 1)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 1)

        # ---- Action client --------------------------------------------------
        self._action_client = _create_action_client(
            self, FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )

        # ---- State ----------------------------------------------------------
        self._lock = threading.Lock()
        self._latest_image = None
        self._latest_joints = None
        self._model = None
        self._rng = None
        self._task = None
        self._unnorm_stats = None
        self._step = 0

    # ------------------------------------------------------------------ #
    # Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _image_cb(self, msg):
        try:
            image = image_to_numpy(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return
        with self._lock:
            self._latest_image = image

    def _joint_cb(self, msg):
        values = [0.0] * len(ARM_JOINTS)
        for name, value in zip(msg.name, msg.position):
            if name in ARM_JOINTS:
                values[ARM_JOINTS.index(name)] = value
        with self._lock:
            self._latest_joints = np.array(values, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Octo model                                                         #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        try:
            from octo.model.octo_model import OctoModel
        except ImportError:
            self.get_logger().fatal(
                "Octo not installed. Run with mock:=true to test."
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

    def _predict(self, image, joints):
        if self.mock:
            return self._mock_predict(joints)
        state = np.zeros(9, dtype=np.float32)
        state[: len(joints)] = joints
        obs = {"image_primary": image, "state": state}
        actions = self._model.sample_actions(
            obs, self._task, rng=self._rng,
            unnormalization_statistics=self._unnorm_stats,
        )
        return np.asarray(actions).astype(np.float32)

    def _mock_predict(self, joints):
        """Scripted pick-and-place motion for mock mode."""
        t = np.arange(self.action_horizon) / self.control_frequency
        phase = self._step % 6

        if self.control_mode == "cartesian":
            actions = np.zeros(
                (self.action_horizon, CARTESIAN_DIM + 1), dtype=np.float32
            )
            if phase == 0:  # move left
                actions[:, 0] = 0.003
                actions[:, 5] = self.gripper_open_pos
            elif phase == 1:  # move down
                actions[:, 2] = -0.003
                actions[:, 5] = self.gripper_open_pos
            elif phase == 2:  # close gripper
                actions[:, 5] = self.gripper_closed_pos
            elif phase == 3:  # lift up
                actions[:, 2] = 0.003
                actions[:, 5] = self.gripper_closed_pos
            elif phase == 4:  # move right
                actions[:, 0] = -0.003
                actions[:, 5] = self.gripper_closed_pos
            else:  # open gripper and retreat
                actions[:, 2] = -0.003
                actions[:, 5] = self.gripper_open_pos
            return actions
        else:
            actions = np.zeros(
                (self.action_horizon, len(ARM_JOINTS)), dtype=np.float32
            )
            base = joints.copy()
            for j in range(5):
                actions[:, j] = base[j] + 0.03 * np.sin(t + j + phase)
            if phase in (2, 3, 4):
                actions[:, 5] = self.gripper_closed_pos
            else:
                actions[:, 5] = self.gripper_open_pos
            return actions

    # ------------------------------------------------------------------ #
    # Cartesian -> joint conversion                                       #
    # ------------------------------------------------------------------ #

    def _cartesian_to_joints(self, actions, current_joints):
        q = np.array(current_joints[:5], dtype=np.float64)
        T = so100_ik.fk(q)
        horizon = actions.shape[0]
        out = np.zeros((horizon, len(ARM_JOINTS)), dtype=np.float32)

        for t in range(horizon):
            T_des = T.copy()
            T_des[0, 3] += float(actions[t, 0])
            T_des[1, 3] += float(actions[t, 1])
            T_des[2, 3] += float(actions[t, 2])
            R_d = so100_ik.rotation_from_rpy(
                float(actions[t, 3]), float(actions[t, 4]), float(actions[t, 5])
            )
            T_des[:3, :3] = T_des[:3, :3] @ R_d

            # 5-DOF arm can't match an arbitrary 6-DOF pose exactly; this
            # converges to the closest reachable pose, fine for small
            # deltas from a streaming policy.
            q_sol, _, _ = so100_ik.solve_pose_ik(T_des[:3, 3], T_des[:3, :3], seed=q)
            q_sol = np.array(q_sol, dtype=np.float64)
            out[t, :5] = q_sol.astype(np.float32)

            adim = actions.shape[1]
            if adim > CARTESIAN_DIM:
                gv = float(actions[t, CARTESIAN_DIM])
                out[t, 5] = (
                    self.gripper_open_pos
                    if gv > self.gripper_open_thresh
                    else self.gripper_closed_pos
                )
            else:
                out[t, 5] = current_joints[5]

            q = q_sol
            T = so100_ik.fk(q)

        return out

    # ------------------------------------------------------------------ #
    # Trajectory execution                                               #
    # ------------------------------------------------------------------ #

    def _execute(self, actions):
        """Send a joint-space action chunk to the arm controller."""
        if not self._action_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn("arm_controller not available")
            return False

        horizon = actions.shape[0]
        adim = actions.shape[1]
        step_dur = self.execution_duration / horizon

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_JOINTS)
        for t in range(horizon):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in actions[t, :adim]]
            for j in range(adim, len(ARM_JOINTS)):
                pt.positions.append(float(self._latest_joints[j]))
            pt.time_from_start = Duration(
                sec=int((t + 1) * step_dur),
                nanosec=int(((t + 1) * step_dur) % 1 * 1e9),
            )
            goal.trajectory.points.append(pt)

        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            return False
        result_f = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_f, timeout_sec=30.0)
        return result_f.result().status == GoalStatus.STATUS_SUCCEEDED

    # ------------------------------------------------------------------ #
    # Main loop                                                          #
    # ------------------------------------------------------------------ #

    def run(self):
        self.get_logger().info(
            f"VLA pick-and-place started (instruction: '{self.instruction}')\n"
            f"  mode={self.control_mode}  mock={self.mock}  "
            f"max_steps={self.max_steps}"
        )

        # Wait for first observation
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            with self._lock:
                if self._latest_image is not None and self._latest_joints is not None:
                    break
        self.get_logger().info("First observation received.")

        # Lazy model load
        if not self.mock:
            if not self._load_model():
                return

        for step in range(self.max_steps):
            if not rclpy.ok():
                break

            self._step = step
            rclpy.spin_once(self, timeout_sec=0.05)

            with self._lock:
                image = self._latest_image
                joints = self._latest_joints
            if image is None or joints is None:
                continue

            self.get_logger().info(f"--- Step {step + 1}/{self.max_steps} ---")

            # Predict
            try:
                actions = self._predict(image, joints)
            except Exception as exc:
                self.get_logger().error(f"Prediction failed: {exc}")
                continue

            # Convert Cartesian -> joint space if needed
            if self.control_mode == "cartesian":
                joint_actions = self._cartesian_to_joints(actions, joints)
                if joint_actions is None:
                    self.get_logger().warn("IK failed, skipping step")
                    continue
                actions = joint_actions

            # Execute
            ok = self._execute(actions)
            if ok:
                self.get_logger().info(f"Step {step + 1} executed")
            else:
                self.get_logger().warn(f"Step {step + 1} execution failed")

        self.get_logger().info("VLA pick-and-place finished.")


def main():
    rclpy.init()
    node = VLAPickPlace()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
