#!/usr/bin/env python3
"""Gripper control for the SO-100 arm in Gazebo.

Commands the ``gripper_joint`` through the ros2_control ``arm_controller``
(``/arm_controller/follow_joint_trajectory``) the same way the arm joints are
commanded, so it works with the existing gazebo.launch.py setup.
"""

import os
import subprocess
import time

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
from trajectory_msgs.msg import JointTrajectoryPoint

GRIPPER_JOINT = "gripper_joint"

# URDF limit for gripper_joint is [-0.2, 2.0]. Convention used here:
# 0.0 rad = fully closed, 2.0 rad = fully open.
DEFAULT_OPEN_POSITION = 1.8
DEFAULT_CLOSED_POSITION = 0.0

# Topics for the DetachableJoint plugin (see so100.urdf.xacro): real
# box-vs-box contact between the jaws and a grasped object doesn't
# reliably register in ODE for these thin, actively-controlled fingers,
# so friction alone can't hold anything. Closing the gripper instead
# creates a rigid joint between the gripper and the object; opening it
# removes it.
DETACHABLE_ATTACH_TOPIC = "/model/so100/detachable_joint/attach"
DETACHABLE_DETACH_TOPIC = "/model/so100/detachable_joint/detach"


def _gz_publish_empty(topic, duration=1, logger=None, retries=3):
    """Publish a one-shot gz.msgs.Empty on a Gazebo Transport topic.

    Sent up to ``retries`` times: gz-transport discovery for a
    freshly-started publisher process is occasionally too slow to land
    within a single ~1s publish window, which silently drops the
    attach/detach request with no error -- retrying is cheap insurance
    against that race.
    """
    env = dict(os.environ)
    env.setdefault("GZ_IP", "127.0.0.1")
    env.setdefault("GZ_PARTITION", "localhost")
    for attempt in range(retries):
        try:
            result = subprocess.run(
                [
                    "gz", "topic",
                    "-t", topic,
                    "-m", "gz.msgs.Empty",
                    "-p", "unused: true",
                    "-d", str(duration),
                ],
                env=env,
                timeout=duration + 5,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 and logger:
                logger.error(
                    f"gz topic publish to {topic} failed "
                    f"(attempt {attempt + 1}/{retries}, rc={result.returncode}): "
                    f"{result.stderr.strip()}"
                )
        except (subprocess.SubprocessError, OSError) as exc:
            if logger:
                logger.error(
                    f"gz topic publish to {topic} raised "
                    f"(attempt {attempt + 1}/{retries}): {exc}"
                )


class Gripper(Node):
    """Synchronous helper to open/close the SO-100 gripper."""

    def __init__(self, node_name="gripper"):
        super().__init__(node_name)
        self.set_parameters([Parameter("use_sim_time", value=True)])
        self._action_client = _create_action_client(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )
        self.open_position = self.declare_parameter(
            "open_position", DEFAULT_OPEN_POSITION
        ).value
        self.close_position = self.declare_parameter(
            "close_position", DEFAULT_CLOSED_POSITION
        ).value

    def _send_position(self, position, duration):
        rclpy.spin_once(self, timeout_sec=0.2)
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("arm_controller action server not available")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [GRIPPER_JOINT]
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        point.time_from_start = Duration(
            sec=int(duration), nanosec=int((duration % 1) * 1e9)
        )
        goal.trajectory.points = [point]

        self.get_logger().info(
            f"Moving {GRIPPER_JOINT} to {position:.3f} rad over {duration}s"
        )

        goal_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=10.0)
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Gripper trajectory goal was rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
        result = result_future.result()
        if result is None:
            self.get_logger().error("Timed out waiting for gripper trajectory result")
            return False
        return result.status == GoalStatus.STATUS_SUCCEEDED

    def open(self, duration=2.5):
        """Fully open the gripper, releasing anything held. Returns True on success."""
        _gz_publish_empty(DETACHABLE_DETACH_TOPIC, logger=self.get_logger())
        return self._send_position(self.open_position, duration)

    def close(self, duration=2.5):
        """Fully close the gripper and grasp whatever's between the jaws.

        Returns True on success.
        """
        result = self._send_position(self.close_position, duration)
        if result:
            _gz_publish_empty(DETACHABLE_ATTACH_TOPIC, logger=self.get_logger())
        return result

    def set_position(self, position, duration=1.0):
        """Set the gripper to an arbitrary position in the joint range."""
        return self._send_position(position, duration)


def main():
    """gripper_demo: repeatedly open/close the gripper to test it."""
    rclpy.init()
    gripper = Gripper()
    try:
        while rclpy.ok():
            if not gripper.open():
                break
            time.sleep(2.0)
            if not gripper.close():
                break
            time.sleep(2.0)
    except KeyboardInterrupt:
        pass
    finally:
        gripper.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001 - context may already be shutting down
            pass


if __name__ == "__main__":
    main()
