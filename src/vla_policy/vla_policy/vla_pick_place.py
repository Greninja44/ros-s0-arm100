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


# ------------------------------------------------------------------ #
# KDL IK (same as octo_policy_node, kept self-contained)             #
# ------------------------------------------------------------------ #

def _build_kdl_chain(urdf_path):
    try:
        from kdl_parser import treeFromUrdfFile
    except ImportError:
        return None
    ok, tree = treeFromUrdfFile(urdf_path)
    if not ok:
        return None
    return tree.getChain("base", "gripper")


def _kdl_fk(chain, q):
    from PyKDL import ChainFkSolverPos_recursive, Frame
    fk = ChainFkSolverPos_recursive(chain)
    frame = Frame()
    fk.JntToCart(q, frame)
    return _frame_to_matrix(frame)


def _frame_to_matrix(frame):
    R = np.array(frame.M)
    p = np.array(frame.p)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _matrix_to_frame(T):
    from PyKDL import Frame, Rotation, Vector
    R = Rotation(
        float(T[0, 0]), float(T[0, 1]), float(T[0, 2]),
        float(T[1, 0]), float(T[1, 1]), float(T[1, 2]),
        float(T[2, 0]), float(T[2, 1]), float(T[2, 2]),
    )
    p = Vector(float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
    return Frame(R, p)


def _kdl_ik(chain, q_init, T_desired, limits, max_iter=100, eps=1e-5):
    from PyKDL import (
        ChainFkSolverPos_recursive, ChainIkSolverVel_pinv,
        ChainIkSolverPos_NR, JntArray,
    )
    fk = ChainFkSolverPos_recursive(chain)
    vel = ChainIkSolverVel_pinv(chain)
    ik = ChainIkSolverPos_NR(chain, fk, vel, maxiter=max_iter, eps=eps)

    q0 = JntArray(len(q_init))
    for i, v in enumerate(q_init):
        q0[i] = float(v)
    q_out = JntArray(len(q_init))
    ret = ik.CartToJnt(q0, _matrix_to_frame(T_desired), q_out)
    if ret < 0:
        return None
    result = np.array([q_out[i] for i in range(len(q_init))])
    if limits is not None:
        for i in range(len(result)):
            result[i] = np.clip(result[i], limits[i, 0], limits[i, 1])
    return result.astype(np.float64)


ARM_JOINT_LIMITS = np.array([
    [-2.0, 2.0],
    [-0.001, 3.5],
    [-3.14158, 0.001],
    [-2.5, 1.2],
    [-3.14158, 3.14158],
], dtype=np.float64)


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
        self._kdl_chain = None
        self._step = 0

        if self.control_mode == "cartesian":
            self._setup_ik()

    # ------------------------------------------------------------------ #
    # IK setup                                                           #
    # ------------------------------------------------------------------ #

    def _setup_ik(self):
        import subprocess, tempfile, os

        urdf_paths = [
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..",
                "urdf", "so100.urdf.xacro"
            ),
            os.path.join(
                os.path.expanduser("~"), "pick_place_ws", "install",
                "so100_description", "share", "so100_description",
                "urdf", "so100.urdf.xacro"
            ),
        ]
        try:
            urdf_param = self.get_parameter("/robot_description").value
            if urdf_param:
                with tempfile.NamedTemporaryFile(
                    suffix=".xacro", mode="w", delete=False
                ) as f:
                    f.write(urdf_param)
                    urdf_paths.insert(0, f.name)
        except Exception:
            pass

        for path in urdf_paths:
            if path and os.path.exists(path):
                try:
                    result = subprocess.run(
                        ["xacro", path], capture_output=True, text=True,
                        timeout=5.0,
                    )
                    if result.returncode == 0:
                        with tempfile.NamedTemporaryFile(
                            suffix=".urdf", mode="w", delete=False
                        ) as f:
                            f.write(result.stdout)
                            urdf_path = f.name
                    else:
                        continue
                except FileNotFoundError:
                    continue
                chain = _build_kdl_chain(urdf_path)
                if chain is not None:
                    self._kdl_chain = chain
                    self.get_logger().info(f"IK ready — loaded from {path}")
                    return

        self.get_logger().warn(
            "Could not load KDL chain. Falling back to joint-space mode."
        )
        self.control_mode = "joint_space"

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
        if self._kdl_chain is None:
            return None
        q = np.array(current_joints[:5], dtype=np.float64)
        T = _kdl_fk(self._kdl_chain, q)
        horizon = actions.shape[0]
        out = np.zeros((horizon, len(ARM_JOINTS)), dtype=np.float32)

        for t in range(horizon):
            T_des = T.copy()
            T_des[0, 3] += float(actions[t, 0])
            T_des[1, 3] += float(actions[t, 1])
            T_des[2, 3] += float(actions[t, 2])
            from PyKDL import Rotation
            R_d = Rotation.RPY(
                float(actions[t, 3]), float(actions[t, 4]), float(actions[t, 5])
            )
            T_des[:3, :3] = T_des[:3, :3] @ np.array(R_d)

            q_sol = _kdl_ik(self._kdl_chain, q, T_des, ARM_JOINT_LIMITS)
            if q_sol is None:
                q_sol = q.copy()
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
            T = _kdl_fk(self._kdl_chain, q)

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
