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
  converted to joint positions via KDL inverse kinematics so that the arm
  tracks the desired Cartesian trajectory.

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

import os
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


# ------------------------------------------------------------------ #
# KDL helpers                                                         #
# ------------------------------------------------------------------ #

def _build_kdl_chain_from_urdf(urdf_path):
    """Parse a URDF file and return a KDL Chain from base_link -> gripper."""
    try:
        from kdl_parser import treeFromUrdfFile
        from PyKDL import Chain
    except ImportError:
        return None

    ok, tree = treeFromUrdfFile(urdf_path)
    if not ok:
        return None

    # The chain runs base -> shoulder -> upper_arm -> lower_arm -> wrist -> gripper
    chain = tree.getChain("base", "gripper")
    return chain


def _kdl_fk(chain, q):
    """Forward kinematics: return the 4x4 pose matrix for joint config q."""
    from PyKDL import ChainFkSolverPos_recursive, Frame, Rotation, Vector

    fk_solver = ChainFk_solverPos_recursive(chain)
    frame = Frame()
    fk_solver.JntToCart(q, frame)
    return _frame_to_matrix(frame)


def _frame_to_matrix(frame):
    """Convert a PyKDL Frame to a 4x4 numpy array."""
    R = np.array(frame.M)
    p = np.array(frame.p)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _matrix_to_frame(T):
    """Convert a 4x4 numpy array to a PyKDL Frame."""
    from PyKDL import Frame, Rotation, Vector

    R = Rotation(
        float(T[0, 0]), float(T[0, 1]), float(T[0, 2]),
        float(T[1, 0]), float(T[1, 1]), float(T[1, 2]),
        float(T[2, 0]), float(T[2, 1]), float(T[2, 2]),
    )
    p = Vector(float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
    return Frame(R, p)


def _kdl_ik(chain, q_init, T_desired, joint_limits, max_iter=100, eps=1e-5):
    """Numerical IK using the KDL pseudo-inverse (Jacobian transpose) method.

    Returns the joint solution or None if convergence fails.
    """
    from PyKDL import (
        ChainFkSolverPos_recursive,
        ChainIkSolverVel_pinv,
        ChainIkSolverPos_NR,
        JntArray,
    )

    fk_solver = ChainFkSolverPos_recursive(chain)
    ik_vel_solver = ChainIkSolverVel_pinv(chain)
    ik_solver = ChainIkSolverPos_NR(
        chain, fk_solver, ik_vel_solver, maxiter=max_iter, eps=eps
    )

    q_init_kdl = JntArray(len(q_init))
    for i, v in enumerate(q_init):
        q_init_kdl[i] = float(v)

    q_out = JntArray(len(q_init))
    target_frame = _matrix_to_frame(T_desired)

    ret = ik_solver.CartToJnt(q_init_kdl, target_frame, q_out)
    if ret < 0:
        return None

    result = np.array([q_out[i] for i in range(len(q_init))])

    # Clamp to joint limits
    for i in range(len(result)):
        if joint_limits is not None:
            result[i] = np.clip(result[i], joint_limits[i, 0], joint_limits[i, 1])

    return result.astype(np.float64)


# ------------------------------------------------------------------ #
# Joint limits (from URDF)                                            #
# ------------------------------------------------------------------ #

ARM_JOINT_LIMITS = np.array([
    [-2.0, 2.0],       # shoulder_pan
    [-0.001, 3.5],     # shoulder_lift
    [-3.14158, 0.001], # elbow_flex
    [-2.5, 1.2],       # wrist_flex
    [-3.14158, 3.14158], # wrist_roll
], dtype=np.float64)


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

        # ---- KDL IK setup --------------------------------------------------
        self._kdl_chain = None
        if self.control_mode == "cartesian":
            self._setup_ik()

        self.create_timer(1.0 / self.control_frequency, self._control_loop)

    def _setup_ik(self):
        """Load the URDF and build the KDL chain + IK solver."""
        # Try to find the URDF from the package share directory
        urdf_paths = [
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..",
                "urdf", "so100.urdf.xacro"
            ),
            # Installed location
            os.path.join(
                os.path.expanduser("~"), "pick_place_ws", "install",
                "so100_description", "share", "so100_description",
                "urdf", "so100.urdf.xacro"
            ),
        ]

        # Also try through the parameter server
        try:
            urdf_param = self.get_parameter("/robot_description").value
            if urdf_param:
                # Write to temp file for kdl_parser
                import tempfile
                with tempfile.NamedTemporaryFile(
                    suffix=".xacro", mode="w", delete=False
                ) as f:
                    f.write(urdf_param)
                    urdf_paths.insert(0, f.name)
        except Exception:
            pass

        for path in urdf_paths:
            if path and os.path.exists(path):
                # Need to process xacro first
                import subprocess
                try:
                    result = subprocess.run(
                        ["xacro", path],
                        capture_output=True, text=True, timeout=5.0
                    )
                    if result.returncode == 0:
                        import tempfile
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
                    self.get_logger().info(
                        f"IK ready — loaded chain from {path}"
                    )
                    return

        self.get_logger().warn(
            "Could not load KDL chain for IK. "
            "Cartesian control mode will fall back to joint-space."
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
        if self._kdl_chain is None:
            self.get_logger().warn("No KDL chain available, cannot do IK")
            return None

        arm_joints = current_joints[:5]
        q = np.array(arm_joints, dtype=np.float64)

        horizon = actions.shape[0]
        joint_traj = np.zeros((horizon, len(ARM_JOINTS)), dtype=np.float32)

        # Get initial FK
        T_current = _kdl_fk(self._kdl_chain, q)

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
            from PyKDL import Rotation
            R_delta = Rotation.RPY(
                float(droll), float(dpitch), float(dyaw)
            )
            R_delta_np = np.array(R_delta)
            T_desired[:3, :3] = T_desired[:3, :3] @ R_delta_np

            # Solve IK
            q_sol = _kdl_ik(
                self._kdl_chain, q, T_desired, ARM_JOINT_LIMITS
            )
            if q_sol is None:
                self.get_logger().warn(
                    f"IK failed at step {t}, using last valid config"
                )
                q_sol = q.copy()

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
            T_current = _kdl_fk(self._kdl_chain, q)

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
