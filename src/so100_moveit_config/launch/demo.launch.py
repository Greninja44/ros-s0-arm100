import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    # ---------------------------------------------------------
    # Package paths
    # ---------------------------------------------------------

    description_pkg = get_package_share_directory(
        "so100_description"
    )

    moveit_pkg = get_package_share_directory(
        "so100_moveit_config"
    )

    urdf_file = os.path.join(
        description_pkg,
        "urdf",
        "so100.urdf.xacro"
    )

    srdf_file = os.path.join(
        moveit_pkg,
        "config",
        "so100.srdf"
    )

    kinematics_file = os.path.join(
        moveit_pkg,
        "config",
        "kinematics.yaml"
    )

    controllers_file = os.path.join(
        moveit_pkg,
        "config",
        "moveit_controllers.yaml"
    )

    # ---------------------------------------------------------
    # Robot description
    # ---------------------------------------------------------

    robot_description = {
        "robot_description": ParameterValue(
            Command([
                "xacro ",
                urdf_file
            ]),
            value_type=str
        )
    }

    # ---------------------------------------------------------
    # Semantic robot description
    #
    # IMPORTANT:
    # Read the SRDF directly instead of relying on the
    # MoveIt demo launch to pass it through.
    # ---------------------------------------------------------

    with open(srdf_file, "r") as f:
        robot_description_semantic = {
            "robot_description_semantic": f.read()
        }

    # ---------------------------------------------------------
    # Kinematics
    # ---------------------------------------------------------

    kinematics = MoveItConfigsBuilder(
        "so100",
        package_name="so100_moveit_config"
    ).robot_description_kinematics(
        file_path=kinematics_file
    ).to_moveit_configs()

    # ---------------------------------------------------------
    # MoveIt configuration
    # ---------------------------------------------------------

    moveit_config = (
        MoveItConfigsBuilder(
            "so100",
            package_name="so100_moveit_config"
        )
        .robot_description(
            file_path=urdf_file
        )
        .robot_description_semantic(
            file_path=srdf_file
        )
        .robot_description_kinematics(
            file_path=kinematics_file
        )
        .trajectory_execution(
            file_path=controllers_file
        )
        .planning_pipelines(
            pipelines=["ompl"]
        )
        .to_moveit_configs()
    )

    # ---------------------------------------------------------
    # Move Group
    # ---------------------------------------------------------

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.trajectory_execution,
            moveit_config.joint_limits,
            {
                "use_sim_time": True,

                # Planning scene monitor
                "publish_planning_scene": True,
                "publish_geometry_updates": True,
                "publish_state_updates": True,
                "publish_transforms_updates": True,

                "publish_robot_description": True,
                "publish_robot_description_semantic": True,

                # Planning scene service
                "publish_planning_scene_hz": 10.0,

                # Give trajectory execution generous slack before it
                # decides the controller is "taking too long" and cancels
                # the goal -- Gazebo physics can run behind wall-clock
                # under load, and a cancelled-but-still-succeeding
                # trajectory looks identical to a real failure otherwise.
                "trajectory_execution.allowed_execution_duration_scaling": 5.0,
                "trajectory_execution.allowed_goal_duration_margin": 5.0,
            },
        ],
    )

    # ---------------------------------------------------------
    # RViz
    # ---------------------------------------------------------

    rviz_config = os.path.join(
        moveit_pkg,
        "config",
        "moveit.rviz"
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        output="screen",
        arguments=[
            "-d",
            rviz_config
        ],
        parameters=[
            robot_description,
            robot_description_semantic,
            moveit_config.robot_description_kinematics,
            {
                "use_sim_time": True
            }
        ],
    )

    # ---------------------------------------------------------
    # Launch
    #
    # world -> base is published by robot_state_publisher itself:
    # the URDF now has a "world" link fixed-jointed to "base" (also
    # what anchors the robot to the static simulation frame in
    # Gazebo), so no separate static_transform_publisher is needed
    # for the SRDF's virtual_joint frame.
    # ---------------------------------------------------------

    return LaunchDescription([
        move_group,
        rviz,
    ])