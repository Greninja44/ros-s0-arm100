import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression, TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    package_name = "so100_description"

    pkg_share = get_package_share_directory(package_name)

    world_arg = DeclareLaunchArgument(
        "world",
        default_value="empty.world",
        description="World file name (e.g. empty.world, pick_place.world)"
    )

    spawn_z_arg = DeclareLaunchArgument(
        "spawn_z",
        default_value="0.13",
        description="Robot base spawn height. pick_place.world has a table "
                     "with its top surface at z=0.08 (default here is "
                     "0.08 + the same 0.05 ground clearance used on the "
                     "bare ground plane); pass spawn_z:=0.05 with "
                     "world:=empty.world instead."
    )

    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo server-only, without the GUI client"
    )

    urdf_file = os.path.join(
        pkg_share,
        "urdf",
        "so100.urdf.xacro"
    )

    world_file = PathJoinSubstitution([
        pkg_share,
        "worlds",
        LaunchConfiguration("world")
    ])

    headless_flag = PythonExpression([
        "'-s ' if '", LaunchConfiguration("headless"), "' == 'true' else ''"
    ])

    gz_args = [
        headless_flag,
        TextSubstitution(text="-r "),
        world_file
    ]

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
            "gz_args": gz_args
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
    # Gazebo -> ROS 2 camera bridge
    #
    # The overhead camera sensor in pick_place.world publishes to
    # "image" inside its model namespace, which resolves to the
    # topic /overhead_camera/link/image.  The bridge must use
    # this full topic path.
    # =========================================================

    camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/overhead_camera/link/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/overhead_camera/link/image/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        remappings=[
            ("/overhead_camera/link/image", "/image"),
            ("/overhead_camera/link/image/camera_info", "/image/camera_info"),
        ],
        output="screen"
    )

    # =========================================================
    # Gazebo -> ROS 2 depth camera bridge
    # (for pick_place_depth.world and domain_randomized.world)
    # =========================================================

    depth_camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/overhead_camera/link/depth/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/overhead_camera/link/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        remappings=[
            ("/overhead_camera/link/depth/image", "/depth/image"),
            ("/overhead_camera/link/depth/camera_info", "/depth/camera_info"),
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
            LaunchConfiguration("spawn_z")
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
            "/controller_manager",
            "--controller-manager-timeout",
            "30"
        ],
        parameters=[{"use_sim_time": True}],
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
            "/controller_manager",
            "--controller-manager-timeout",
            "30"
        ],
        parameters=[{"use_sim_time": True}],
        output="screen"
    )

    # =========================================================
    # Sequencing
    # =========================================================
    #
    # The joint_state_broadcaster/arm_controller spawners race each other
    # and the controller_manager (loaded by the gz_ros2_control plugin only
    # once the robot is spawned) if launched all at once, occasionally
    # timing out and dying before the controller_manager service is even
    # up. Chain them off process-exit events instead so each spawner only
    # starts once the previous step has actually finished.

    spawn_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_robot,
            on_exit=[joint_state_broadcaster]
        )
    )

    spawn_arm_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[arm_controller]
        )
    )

    # =========================================================
    # Launch
    # =========================================================

    return LaunchDescription([
        world_arg,
        spawn_z_arg,
        headless_arg,
        gazebo,
        clock_bridge,
        camera_bridge,
        depth_camera_bridge,
        robot_state_publisher,
        spawn_robot,
        spawn_joint_state_broadcaster,
        spawn_arm_controller,
    ])