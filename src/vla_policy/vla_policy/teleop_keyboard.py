#!/usr/bin/env python3
"""Keyboard teleoperation for the SO-100 arm in Gazebo Harmonic.

Controls the 5 arm joints and the gripper via keyboard input.  Each key
press sends an incremental joint command through the ``arm_controller``
(FollowJointTrajectory action).

Layout
------
    Shoulder pan  : a / d   (left / right)
    Shoulder lift : w / s   (up / down)
    Elbow flex    : q / e   (close / open)
    Wrist flex    : r / f   (up / down)
    Wrist roll    : z / x   (ccw / cw)
    Gripper       : o / p   (open / close)
    Quit          : Ctrl-C

Usage
-----
    ros2 run vla_policy teleop_keyboard

The node publishes to ``/arm_controller/follow_joint_trajectory`` and
reads from ``/joint_states`` to get the current positions before adding
the increment.
"""

import sys
import tty
import termios

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper_joint",
]

# Joint limits from urdf/so100.urdf.xacro (rad) — used for clamping.
JOINT_MIN = np.array([-2.0, -0.001, -3.14158, -2.5, -3.14158, -0.2], dtype=np.float64)
JOINT_MAX = np.array([2.0, 3.5, 0.001, 1.2, 3.14158, 2.0], dtype=np.float64)

DEFAULT_STEP = 0.05  # rad per key press


def _create_action_client(node, action_type, action_name):
    """Compat helper: some rclpy builds lack Node.create_action_client()."""
    if hasattr(node, "create_action_client"):
        return node.create_action_client(action_type, action_name)
    return ActionClient(node, action_type, action_name)


class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__("teleop_keyboard")
        self.set_parameters([Parameter("use_sim_time", value=True)])

        self.step = self.declare_parameter("step", DEFAULT_STEP).value

        self._action_client = _create_action_client(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

        self._joints = np.zeros(len(ARM_JOINTS), dtype=np.float64)
        self._got_joints = False

        self._key_map = {
            "a": (0, -self.step),
            "d": (0, self.step),
            "w": (1, self.step),
            "s": (1, -self.step),
            "q": (2, -self.step),
            "e": (2, self.step),
            "r": (3, self.step),
            "f": (3, -self.step),
            "z": (4, -self.step),
            "x": (4, self.step),
            "o": (5, -self.step),
            "p": (5, self.step),
        }

    def _joint_cb(self, msg):
        for i, name in enumerate(ARM_JOINTS):
            if name in msg.name:
                idx = msg.name.index(name)
                self._joints[i] = msg.position[idx]
        self._got_joints = True

    def _send_command(self, target):
        target = np.clip(target, JOINT_MIN, JOINT_MAX)
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("arm_controller not available")
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in target]
        point.time_from_start = Duration(sec=0, nanosec=200_000_000)
        goal.trajectory.points = [point]

        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_done)

    def _goal_done(self, future):
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().warn("Goal rejected")
            return
        result_f = handle.get_result_async()

        def _cb(fut):
            status = fut.result().status
            if status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().warn(f"Goal finished with status {status}")

        result_f.add_done_callback(_cb)

    def spin(self):
        self.get_logger().info(
            "Keyboard teleop ready.\n"
            "  a/d : shoulder_pan  left/right\n"
            "  w/s : shoulder_lift up/down\n"
            "  q/e : elbow_flex   close/open\n"
            "  r/f : wrist_flex   up/down\n"
            "  z/x : wrist_roll   ccw/cw\n"
            "  o/p : gripper       open/close\n"
            "  Ctrl-C to quit"
        )
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if not self._got_joints:
                continue
            ch = self._read_key()
            if ch is None:
                continue
            if ch == "\x03":  # Ctrl-C
                break
            if ch in self._key_map:
                idx, delta = self._key_map[ch]
                new_joints = self._joints.copy()
                new_joints[idx] += delta
                self._send_command(new_joints)
                self.get_logger().info(
                    f"{ARM_JOINTS[idx]}: {self._joints[idx]:.3f} -> "
                    f"{new_joints[idx]:.3f} rad"
                )
            else:
                self.get_logger().info(f"Unknown key: {repr(ch)}")

    @staticmethod
    def _read_key():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch


def main():
    rclpy.init()
    node = TeleopKeyboard()
    try:
        node.spin()
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
