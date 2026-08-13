#!/usr/bin/env python3
"""Full pick-and-place cycle for the SO-100 arm in Gazebo.

Plans the arm motion through the MoveIt ``move_group`` node (MoveGroup action)
and commands the gripper through the ``arm_controller`` FollowJointTrajectory
action.

Requires the simulation AND the MoveIt move_group node to be running, e.g.:

    # terminal 1
    ros2 launch so100_description gazebo.launch.py

    # terminal 2
    ros2 launch so100_moveit_config demo.launch.py

    # terminal 3
    ros2 run vla_policy pick_place --ros-args \
        -p object_x:=0.25 -p object_y:=0.0 -p object_z:=0.05
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
    WorkspaceParameters,
)
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header

from vla_policy.gripper import Gripper

ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
HOME_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0]

# Tolerances used for pose goals (metre / radian).
POSE_TOLERANCE = 0.02
ORIENTATION_TOLERANCE = 0.2


class PickPlace(Node):
    def __init__(self):
        super().__init__("pick_place")
        self.set_parameters([Parameter("use_sim_time", value=True)])

        self.planning_time = self.declare_parameter(
            "planning_time", 5.0
        ).value
        self.approach_offset = self.declare_parameter(
            "approach_offset", 0.10
        ).value
        self.object_position = [
            self.declare_parameter(f"object_{ax}", default).value
            for ax, default in zip(("x", "y", "z"), (0.25, 0.0, 0.05))
        ]
        self.place_position = [
            self.declare_parameter(f"place_{ax}", default).value
            for ax, default in zip(("x", "y", "z"), (-0.05, 0.25, 0.05))
        ]

        self.gripper = Gripper("gripper")
        self._move_group = ActionClient(self, MoveGroup, "/move_action")

    # ------------------------------------------------------------------ #
    # MoveGroup helpers                                                   #
    # ------------------------------------------------------------------ #

    def _new_request(self):
        request = MotionPlanRequest()
        request.workspace_parameters = WorkspaceParameters()
        request.workspace_parameters.header = Header(frame_id="world")
        request.workspace_parameters.min_corner = Point(x=-1.0, y=-1.0, z=-1.0)
        request.workspace_parameters.max_corner = Point(x=1.0, y=1.0, z=1.0)
        request.start_state.is_diff = True
        request.group_name = "arm"
        request.pipeline_id = "ompl"
        request.num_planning_attempts = 10
        request.allowed_planning_time = self.planning_time
        request.goal_constraints = [Constraints()]
        return request

    def _add_joint_goal(self, constraints, target):
        for name, position in zip(ARM_JOINTS, target):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(position)
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

    def _add_pose_goal(self, constraints, position):
        pc = PositionConstraint()
        pc.header = Header(frame_id="world")
        pc.link_name = "gripper"
        pc.weight = 1.0

        volume = BoundingVolume()
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [
            POSE_TOLERANCE,
            POSE_TOLERANCE,
            POSE_TOLERANCE,
        ]
        volume.primitives.append(box)
        volume.primitive_poses.append(
            Pose(
                position=Point(x=position[0], y=position[1], z=position[2]),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )
        pc.constraint_region = volume
        constraints.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header = Header(frame_id="world")
        oc.link_name = "gripper"
        oc.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        oc.absolute_x_axis_tolerance = ORIENTATION_TOLERANCE
        oc.absolute_y_axis_tolerance = ORIENTATION_TOLERANCE
        oc.absolute_z_axis_tolerance = ORIENTATION_TOLERANCE
        oc.weight = 1.0
        constraints.orientation_constraints.append(oc)

    def _plan_and_execute(self, label):
        rclpy.spin_once(self, timeout_sec=0.2)
        if not self._move_group.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                "move_group action server not available - "
                "is demo.launch.py running?"
            )
            return False

        goal = MoveGroup.Goal()
        goal.request = self._current_request
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        self.get_logger().info(f"Planning + executing '{label}' ...")
        future = self._move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"Goal rejected for '{label}'")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        result = result_future.result()
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                f"Failed '{label}' (status {result.status}) - "
                "adjust the object/place poses or planning tolerances"
            )
            return False
        self.get_logger().info(f"Executed '{label}'")
        return True

    def _move_home(self):
        self._current_request = self._new_request()
        self._add_joint_goal(self._current_request.goal_constraints[0], HOME_POSITIONS)
        return self._plan_and_execute("home")

    def _move_to_pose(self, position, label):
        self._current_request = self._new_request()
        self._add_pose_goal(self._current_request.goal_constraints[0], position)
        return self._plan_and_execute(label)

    # ------------------------------------------------------------------ #
    # Pick & place cycle                                                  #
    # ------------------------------------------------------------------ #

    def run(self):
        obj = self.object_position
        place = self.place_position
        above_obj = [obj[0], obj[1], obj[2] + self.approach_offset]
        above_place = [place[0], place[1], place[2] + self.approach_offset]

        self.get_logger().info(
            f"=== Pick & Place cycle: object at {obj}, place at {place} ==="
        )

        if not self._move_home():
            return
        if not self.gripper.open():
            return

        if not self._move_to_pose(above_obj, "pre-grasp"):
            return
        if not self._move_to_pose(obj, "approach"):
            return

        if not self.gripper.close():
            return
        if not self._move_to_pose(above_obj, "lift"):
            return

        if not self._move_to_pose(above_place, "pre-place"):
            return
        if not self._move_to_pose(place, "lower"):
            return

        if not self.gripper.open():
            return
        if not self._move_to_pose(above_place, "retreat"):
            return
        if not self._move_home():
            return

        self.get_logger().info("=== Pick & Place cycle finished ===")


def main():
    rclpy.init()
    demo = PickPlace()
    try:
        demo.run()
    except KeyboardInterrupt:
        pass
    finally:
        demo.gripper.destroy_node()
        demo.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001 - context may already be shutting down
            pass


if __name__ == "__main__":
    main()
