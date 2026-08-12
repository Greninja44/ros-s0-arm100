import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    package_name = "so100_description"

    pkg_share = get_package_share_directory(package_name)

    urdf_file = os.path.join(
        pkg_share,
        "urdf",
        "so100.urdf.xacro"
    )

    world_file = os.path.join(
        pkg_share,
        "worlds",
        "empty.world"
    )

    # =========================================================
    # Gazebo Harmonic
    # =========================================================

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py"
            )
        ),
        launch_arguments={
            "gz_args": f"-r {world_file}"
        }.items()
    )

    # =========================================================
    # Gazebo -> ROS 2 clock bridge
    # =========================================================

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"
        ],
        output="screen"
    )

    # =========================================================
    # Robot description
    # =========================================================

    robot_description = {
        "robot_description": ParameterValue(
            Command([
                "xacro ",
                urdf_file
            ]),
            value_type=str
        )
    }

    # =========================================================
    # Robot State Publisher
    # =========================================================

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            robot_description,
            {
                "use_sim_time": True
            }
        ],
        output="screen"
    )

    # =========================================================
    # Spawn SO-ARM100
    # =========================================================

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic",
            "robot_description",
            "-name",
            "so100",
            "-x",
            "0",
            "-y",
            "0",
            "-z",
            "0.05"
        ],
        output="screen"
    )

    # =========================================================
    # Joint State Broadcaster
    # =========================================================

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager"
        ],
        output="screen"
    )

    # =========================================================
    # Arm Controller
    # =========================================================

    arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_controller",
            "--controller-manager",
            "/controller_manager"
        ],
        output="screen"
    )

    # =========================================================
    # Launch
    # =========================================================

    return LaunchDescription([
        gazebo,
        clock_bridge,
        robot_state_publisher,
        spawn_robot,
        joint_state_broadcaster,
        arm_controller,
    ])