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
    ros2 run vla_policy pick_place

Grasp/place poses are solved with a custom top-down IK (see so100_ik.py)
rather than MoveIt's pose-goal planning: the SO-100 only has 5 joints, so
MoveIt's IK plugin can only solve position (not a full 6-DOF pose), which
leaves the gripper's approach orientation arbitrary -- fine for reaching a
point, useless for actually grasping something.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Vector3
from std_msgs.msg import Header

from vla_policy.gripper import Gripper
from vla_policy.so100_ik import ARM_JOINTS, solve_ik

try:
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import (
        Constraints,
        JointConstraint,
        MotionPlanRequest,
        WorkspaceParameters,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    MoveGroup = None
    Constraints = JointConstraint = MotionPlanRequest = WorkspaceParameters = None
    _MOVEIT_IMPORT_ERROR = exc
else:
    _MOVEIT_IMPORT_ERROR = None

HOME_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0]

# Position error (metres) above which an IK solution is rejected outright
# rather than sent to the arm.
IK_POS_TOLERANCE = 0.01


class PickPlace(Node):
    def __init__(self):
        super().__init__("pick_place")
        if _MOVEIT_IMPORT_ERROR is not None:
            raise RuntimeError(
                "pick_place requires moveit_msgs. Install MoveIt ROS packages "
                "and source the correct environment before running this node."
            ) from _MOVEIT_IMPORT_ERROR
        self.set_parameters([Parameter("use_sim_time", value=True)])

        self.planning_time = self.declare_parameter(
            "planning_time", 5.0
        ).value
        self.approach_offset = self.declare_parameter(
            "approach_offset", 0.10
        ).value
        # Defaults match worlds/pick_place.world: a table with its top
        # surface at z=0.08, a 3cm pick cube resting on it (center at
        # z=0.095) and a place pad on top of it (surface at z=0.09, so a
        # cube placed there comes to rest at z=0.105).
        self.object_position = [
            self.declare_parameter(f"object_{ax}", default).value
            for ax, default in zip(("x", "y", "z"), (0.20, 0.0, 0.095))
        ]
        self.place_position = [
            self.declare_parameter(f"place_{ax}", default).value
            for ax, default in zip(("x", "y", "z"), (-0.20, 0.0, 0.105))
        ]

        # When set, the object position above is a fallback: run() waits
        # for one detection from the vision pipeline (see
        # sam2_segmentation.py) and grasps wherever it actually found the
        # object instead of the fixed object_x/y/z parameters.
        self.use_vision = self.declare_parameter("use_vision", False).value
        self.vision_topic = self.declare_parameter(
            "vision_topic", "/sam2/centroid"
        ).value
        self.vision_timeout = self.declare_parameter(
            "vision_timeout", 10.0
        ).value
        self._vision_pose = None

        self.gripper = Gripper("gripper")
        self._move_group = ActionClient(self, MoveGroup, "/move_action")

    # ------------------------------------------------------------------ #
    # MoveGroup helpers                                                   #
    # ------------------------------------------------------------------ #

    def _new_request(self):
        request = MotionPlanRequest()
        request.workspace_parameters = WorkspaceParameters()
        request.workspace_parameters.header = Header(frame_id="world")
        request.workspace_parameters.min_corner = Vector3(x=-1.0, y=-1.0, z=-1.0)
        request.workspace_parameters.max_corner = Vector3(x=1.0, y=1.0, z=1.0)
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

    def _solve_pose(self, position, label, constrain_orientation):
        """Solve position (+ top-down approach) IK, returning joint values.

        MoveIt's own KDL plugin only does position-only IK for this 5-DOF
        chain, which leaves the gripper's approach orientation arbitrary --
        fine for reaching a point, useless for actually grasping something,
        since the jaws end up facing an unpredictable direction. Solving
        this ourselves for a specific (top-down) approach and sending the
        result as a joint-space goal gives real control over how the
        gripper arrives, and is also faster/more reliable than pose-goal
        OMPL sampling (joint goals plan almost instantly).

        A strict top-down orientation is only asked for where the gripper
        actually has to touch the object (grasp/place); the hover
        waypoints in between only need position, and constraining their
        orientation too would needlessly shrink the reachable set.
        """
        q, pos_err, orient_err, ok = solve_ik(
            position,
            constrain_orientation=constrain_orientation,
            pos_tolerance=IK_POS_TOLERANCE,
        )
        if not ok:
            self.get_logger().error(
                f"No good IK solution for '{label}' at {position} "
                f"(pos_err={pos_err:.4f}m, orient_err={orient_err:.4f}) - "
                "target may be out of reach for a top-down approach"
            )
            return None
        return q

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
        if result is None:
            self.get_logger().error(f"Timed out waiting for '{label}' result")
            return False
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

    def _move_to_pose(self, position, label, constrain_orientation=False):
        q = self._solve_pose(position, label, constrain_orientation)
        if q is None:
            return False
        self._current_request = self._new_request()
        self._add_joint_goal(self._current_request.goal_constraints[0], q)
        return self._plan_and_execute(label)

    # ------------------------------------------------------------------ #
    # Vision                                                              #
    # ------------------------------------------------------------------ #

    def _wait_for_vision_pose(self):
        """Block (spinning) for one detection on ``vision_topic``.

        Returns True and updates ``self.object_position`` on success,
        False on timeout (leaving object_position at its parameter
        default so callers can choose to fail or fall back).
        """
        sub = self.create_subscription(
            PoseStamped, self.vision_topic, self._vision_pose_cb, 10
        )
        self.get_logger().info(
            f"use_vision: waiting up to {self.vision_timeout}s for a "
            f"detection on {self.vision_topic} ..."
        )
        deadline = self.get_clock().now().nanoseconds / 1e9 + self.vision_timeout
        while self._vision_pose is None:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.get_clock().now().nanoseconds / 1e9 > deadline:
                self.get_logger().error(
                    f"No detection received on {self.vision_topic} within "
                    f"{self.vision_timeout}s; is sam2_segmentation running?"
                )
                self.destroy_subscription(sub)
                return False

        self.object_position = list(self._vision_pose)
        self.get_logger().info(f"Vision detection: object at {self.object_position}")
        self.destroy_subscription(sub)
        return True

    def _vision_pose_cb(self, msg):
        if self._vision_pose is None:
            p = msg.pose.position
            self._vision_pose = (p.x, p.y, p.z)

    # ------------------------------------------------------------------ #
    # Pick & place cycle                                                  #
    # ------------------------------------------------------------------ #

    def run(self):
        if self.use_vision and not self._wait_for_vision_pose():
            return

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
        if not self._move_to_pose(obj, "approach", constrain_orientation=True):
            return

        if not self.gripper.close():
            return
        if not self._move_to_pose(above_obj, "lift"):
            return

        if not self._move_to_pose(above_place, "pre-place"):
            return
        if not self._move_to_pose(place, "lower", constrain_orientation=True):
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
