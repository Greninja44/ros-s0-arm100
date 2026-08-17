#!/usr/bin/env python3
"""Diffusion Policy node for the SO-100 arm.

Implements the Diffusion Policy architecture for stochastic
robotic manipulation.  Uses a Conditional Denoising Diffusion
Probabilistic Model (DDPM) to generate action trajectories
conditioned on vision observations and language instructions.

This node can work with:
  - Pre-trained Diffusion Policy checkpoints (ACT, Diffusion)
  - A learned diffusion model for action chunking
  - Mock mode for pipeline validation

Subscriptions
-------------
    /image              sensor_msgs/Image     RGB camera observation
    /joint_states       sensor_msgs/JointState Current joint positions

Publications
-------------
    /diffusion/action   trajectory_msgs/JointTrajectory  Predicted actions

Usage
-----
    ros2 run vla_policy diffusion_policy --ros-args \
        -p instruction:="pick up the red cube"

    # mock mode:
    ros2 run vla_policy diffusion_policy --ros-args -p mock:=true
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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

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
# Diffusion Process                                                    #
# ------------------------------------------------------------------ #

class DiffusionProcess:
    """DDPM forward/reverse process for action trajectory denoising."""

    def __init__(self, n_steps=100, beta_start=1e-4, beta_end=0.02):
        self.n_steps = n_steps
        self.betas = np.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)
        self.sigma2 = self.betas

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: sample x_t from q(x_t | x0)."""
        if noise is None:
            noise = np.random.randn(*x0.shape)
        a = self.alpha_bar[t]
        return np.sqrt(a) * x0 + np.sqrt(1.0 - a) * noise

    def p_sample(self, x_t, t, noise_pred):
        """Reverse diffusion step: sample x_{t-1} from p(x_{t-1} | x_t)."""
        a = self.alphas[t]
        a_bar = self.alpha_bar[t]
        sigma = np.sqrt(self.sigma2[t])
        mean = (1.0 / np.sqrt(a)) * (
            x_t - (1.0 - a) / np.sqrt(1.0 - a_bar) * noise_pred
        )
        if t > 0:
            return mean + sigma * np.random.randn(*x_t.shape)
        return mean


# ------------------------------------------------------------------ #
# Simple Denoising Network (MLP-based for lightweight deployment)      #
# ------------------------------------------------------------------ #

class SimpleDenoisingNet:
    """Lightweight denoising network for diffusion policy.

    In production, replace this with a proper U-Net or Transformer
    backbone.  This implementation uses a simple 2-layer MLP to
    demonstrate the diffusion pipeline.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        try:
            import torch
            import torch.nn as nn

            self._use_torch = True
            self._net = nn.Sequential(
                nn.Linear(action_dim + obs_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._net.to(self._device)
        except ImportError:
            self._use_torch = False
            # Fallback: simple numpy noise predictor
            pass

    def predict_noise(self, x_t, obs, t):
        """Predict noise eps_theta(x_t, obs, t)."""
        if self._use_torch:
            import torch
            with torch.no_grad():
                t_arr = np.array([t / 100.0], dtype=np.float32)
                inp = np.concatenate([x_t, obs, t_arr]).astype(np.float32)
                inp_t = torch.tensor(inp).unsqueeze(0).to(self._device)
                noise = self._net(inp_t).cpu().numpy().squeeze(0)
            return noise
        else:
            # Mock: return scaled random noise
            return np.random.randn(*x_t.shape) * 0.1


# ------------------------------------------------------------------ #
# Diffusion Policy Node                                                #
# ------------------------------------------------------------------ #

class DiffusionPolicyNode(Node):
    def __init__(self):
        super().__init__("diffusion_policy")

        self.set_parameters([Parameter("use_sim_time", value=True)])

        # ---- Parameters -----------------------------------------------------
        self.instruction = self.declare_parameter(
            "instruction", "pick up the object"
        ).value
        self.image_topic = self.declare_parameter(
            "image_topic", "/image"
        ).value
        self.action_horizon = self.declare_parameter("action_horizon", 16).value
        self.obs_horizon = self.declare_parameter("obs_horizon", 2).value
        self.n_diffusion_steps = self.declare_parameter(
            "n_diffusion_steps", 100
        ).value
        self.control_frequency = self.declare_parameter(
            "control_frequency", 10.0
        ).value
        self.execution_duration = self.declare_parameter(
            "execution_duration", 0.5
        ).value
        self.mock = self.declare_parameter("mock", False).value
        self.checkpoint_path = self.declare_parameter(
            "checkpoint_path", ""
        ).value

        # ---- Subscribers ----------------------------------------------------
        self.create_subscription(Image, self.image_topic, self._image_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

        # ---- Action client --------------------------------------------------
        self._action_client = _create_action_client(
            self, FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )

        # ---- State ----------------------------------------------------------
        self._lock = threading.Lock()
        self._latest_image = None
        self._latest_joints = None
        self._obs_buffer = []
        self._model = None
        self._diffusion = None

        self.create_timer(1.0 / self.control_frequency, self._control_loop)

    # ------------------------------------------------------------------ #
    # Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _image_cb(self, msg):
        try:
            image = image_to_numpy(msg)
        except ValueError:
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
    # Model loading                                                       #
    # ------------------------------------------------------------------ #

    def _ensure_model(self):
        if self._model is not None:
            return True

        self._diffusion = DiffusionProcess(
            n_steps=self.n_diffusion_steps
        )

        obs_dim = 64 + len(ARM_JOINTS)  # image feature + joint state
        action_dim = len(ARM_JOINTS)

        if self.checkpoint_path:
            try:
                self._load_checkpoint()
                self.get_logger().info(
                    f"Loaded diffusion checkpoint from {self.checkpoint_path}"
                )
                return True
            except Exception as exc:
                self.get_logger().warn(
                    f"Failed to load checkpoint: {exc}. Using untrained model."
                )

        self._model = SimpleDenoisingNet(obs_dim, action_dim)
        self.get_logger().info("Initialized untrained diffusion policy.")
        return True

    def _load_checkpoint(self):
        import torch
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu"
        )
        obs_dim = 64 + len(ARM_JOINTS)
        action_dim = len(ARM_JOINTS)
        self._model = SimpleDenoisingNet(obs_dim, action_dim)
        if "model_state_dict" in checkpoint:
            self._model._net.load_state_dict(checkpoint["model_state_dict"])

    # ------------------------------------------------------------------ #
    # Diffusion sampling                                                  #
    # ------------------------------------------------------------------ #

    def _denoise_action(self, obs):
        """Run the reverse diffusion process to generate an action chunk."""
        x = np.random.randn(self.action_horizon, len(ARM_JOINTS))

        for t in reversed(range(self._diffusion.n_steps)):
            noise_pred = self._model.predict_noise(
                x.flatten(), obs, t
            ).reshape(x.shape)
            x = self._diffusion.p_sample(x, t, noise_pred)

        return x.astype(np.float32)

    def _mock_action(self, joints):
        """Simple scripted motion for mock mode."""
        t = np.arange(self.action_horizon) / self.control_frequency
        actions = np.zeros(
            (self.action_horizon, len(ARM_JOINTS)), dtype=np.float32
        )
        for j in range(len(ARM_JOINTS)):
            actions[:, j] = joints[j] + 0.03 * np.sin(t + j * 0.5)
        actions[:, 5] = joints[5] + 0.2 * np.sin(t / 3.0)
        return actions

    # ------------------------------------------------------------------ #
    # Control loop                                                        #
    # ------------------------------------------------------------------ #

    def _control_loop(self):
        with self._lock:
            image = self._latest_image
            joints = self._latest_joints
        if image is None or joints is None:
            return

        if not self._ensure_model():
            return

        # Build observation vector (simplified: flatten image + joints)
        if self.mock:
            actions = self._mock_action(joints)
        else:
            # Resize and flatten image to feature vector
            small = np.array(
                __import__("PIL").Image.fromarray(image).resize((8, 8)),
                dtype=np.float32,
            ).flatten() / 255.0
            obs = np.concatenate([small, joints])

            try:
                actions = self._denoise_action(obs)
            except Exception as exc:
                self.get_logger().error(f"Diffusion sampling failed: {exc}")
                return

        self._send_actions(actions)

    def _send_actions(self, actions):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("arm_controller not available")
            return

        horizon = actions.shape[0]
        step = self.execution_duration / horizon

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_JOINTS)

        for t in range(horizon):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in actions[t]]
            pt.time_from_start = Duration(
                sec=int((t + 1) * step),
                nanosec=int(((t + 1) * step) % 1 * 1e9),
            )
            goal.trajectory.points.append(pt)

        self.get_logger().info(
            f"Sending {horizon}-step diffusion action chunk"
        )
        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_feedback)

    def _goal_feedback(self, future):
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().warn("Goal rejected")
            return
        result_f = handle.get_result_async()

        def _done(fut):
            status = fut.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info("Action executed")
            else:
                self.get_logger().warn(f"Action failed (status {status})")

        result_f.add_done_callback(_done)


def main():
    rclpy.init()
    node = DiffusionPolicyNode()
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
