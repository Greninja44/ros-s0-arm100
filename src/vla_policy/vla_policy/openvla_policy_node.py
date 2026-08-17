#!/usr/bin/env python3
"""OpenVLA policy node for the SO-100 arm.

Drop-in replacement for ``octo_policy_node`` that uses the OpenVLA
foundation model instead of Octo.  OpenVLA (7B) is a vision-language-action
model fine-tuned on the Open X-Embodiment dataset and supports both
joint-space and Cartesian control modes.

Usage
-----
    ros2 run vla_policy openvla_policy --ros-args \
        -p instruction:="pick up the red cube"

    # mock mode (no GPU required):
    ros2 run vla_policy openvla_policy --ros-args -p mock:=true

    # Cartesian mode with IK:
    ros2 run vla_policy openvla_policy --ros-args \
        -p control_mode:=cartesian \
        -p instruction:="pick up the red cube"

Requirements
------------
    pip install openvla transformers torch
"""

import os
import threading

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

ARM_ONLY_JOINTS = ARM_JOINTS[:5]
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
# KDL helpers (shared with octo_policy_node)                          #
# ------------------------------------------------------------------ #

def _build_kdl_chain_from_urdf(urdf_path):
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
# OpenVLA Policy Node                                                  #
# ------------------------------------------------------------------ #

class OpenVLAPolicyNode(Node):
    def __init__(self):
        super().__init__("openvla_policy")

        # ---- Parameters -----------------------------------------------------
        self.set_parameters([Parameter("use_sim_time", value=True)])
        self.instruction = self.declare_parameter(
            "instruction", "pick up the object and place it down"
        ).value
        self.model_id = self.declare_parameter(
            "model_id", "openvla/openvla-7b"
        ).value
        self.action_horizon = self.declare_parameter("action_horizon", 50).value
        self.control_frequency = self.declare_parameter(
            "control_frequency", 5.0
        ).value
        self.execution_duration = self.declare_parameter(
            "execution_duration", 1.0
        ).value
        self.image_topic = self.declare_parameter(
            "image_topic", "/image"
        ).value
        self.mock = self.declare_parameter("mock", False).value
        self.control_mode = self.declare_parameter(
            "control_mode", "joint_space"
        ).value
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

        # ---- Action client --------------------------------------------------
        self._action_client = _create_action_client(
            self, FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )

        # ---- State ----------------------------------------------------------
        self._latest_image = None
        self._latest_joints = None
        self._obs_lock = threading.Lock()
        self._model = None
        self._processor = None
        self._kdl_chain = None

        if self.control_mode == "cartesian":
            self._setup_ik()

        self.create_timer(1.0 / self.control_frequency, self._control_loop)

    # ------------------------------------------------------------------ #
    # IK setup (identical to octo_policy_node)                            #
    # ------------------------------------------------------------------ #

    def _setup_ik(self):
        import subprocess, tempfile

        urdf_paths = [
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..",
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
                chain = _build_kdl_chain_from_urdf(urdf_path)
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
    # OpenVLA model                                                       #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            self.get_logger().fatal(
                "transformers is not installed. "
                "Install with: pip install transformers\n"
                "Or run with mock:=true to test the ROS pipeline."
            )
            return False

        self.get_logger().info(f"Loading OpenVLA model {self.model_id} ...")
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.get_logger().info("OpenVLA model ready.")
        return True

    def _predict(self, image, joints):
        """Run OpenVLA inference and return action chunk."""
        if self._model is None:
            return self._mock_actions()

        import torch
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)

        prompt = (
            f"In: What action should the robot take to {self.instruction}?\n"
            f"Out:"
        )

        inputs = self._processor(
            prompt, pil_img, return_tensors="pt"
        ).to(self._model.device, dtype=torch.float16)

        with torch.no_grad():
            action_tokens = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.5,
            )

        action = self._processor.decode_actions(
            action_tokens,
            unnormalize_outputs=True,
        )

        action_np = np.array(action, dtype=np.float32)
        if action_np.ndim == 1:
            action_np = action_np.reshape(1, -1)
        return action_np

    def _mock_actions(self):
        base = np.array(self._latest_joints, dtype=np.float32)
        t = np.arange(self.action_horizon) / self.control_frequency
        if self.control_mode == "cartesian":
            actions = np.zeros(
                (self.action_horizon, CARTESIAN_DIM), dtype=np.float32
            )
            actions[:, 0] = 0.005 * np.sin(t)
            actions[:, 2] = -0.005 * np.cos(t)
            gripper = np.zeros((self.action_horizon, 1), dtype=np.float32)
            gripper[: self.action_horizon // 2] = 1.0
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
    # Cartesian -> joint conversion (shared IK)                           #
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
                    self.gripper_position_open
                    if gv > self.gripper_open_thresh
                    else self.gripper_position_closed
                )
            else:
                out[t, 5] = current_joints[5]
            q = q_sol
            T = _kdl_fk(self._kdl_chain, q)
        return out

    # ------------------------------------------------------------------ #
    # Control loop                                                        #
    # ------------------------------------------------------------------ #

    def _control_loop(self):
        with self._obs_lock:
            image = self._latest_image
            joints = self._latest_joints
        if image is None or joints is None:
            return

        if self._model is None and not self.mock:
            if not self._load_model():
                return

        observation = {"image_primary": image, "state": joints}
        try:
            actions = self._predict(image, joints)
        except Exception as exc:
            self.get_logger().error(f"OpenVLA inference failed: {exc}")
            return

        if self.control_mode == "cartesian":
            joint_actions = self._cartesian_to_joints(actions, joints)
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
            f"Sending {horizon}-step OpenVLA action chunk (dim {action_dim}) "
            f"mode={self.control_mode}"
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
    node = OpenVLAPolicyNode()
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
        except Exception:
            pass


if __name__ == "__main__":
    main()
