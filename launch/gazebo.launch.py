import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    package_name = "so100_description"

    pkg_share = get_package_share_directory(package_name)

    urdf_file = os.path.join(
        pkg_share,
        "urdf",
        "so100.urdf.xacro"
    )

    controllers_file = os.path.join(
        pkg_share,
        "config",
        "controllers.yaml"
    )

    world_arg = DeclareLaunchArgument(
        "world",
        default_value="pick_place.world",
        description="World file to load (pick_place.world or empty.world)"
    )

    world_file = PathJoinSubstitution([
        FindPackageShare(package_name),
        "worlds",
        LaunchConfiguration("world"),
    ])

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
            "gz_args": ["-r ", world_file]
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
    # Controller Manager — load controller definitions from YAML
    #
    # The gz_ros2_control Gazebo plugin creates the
    # controller_manager, but we still need to load the parameter
    # file that defines the controllers (joint_state_broadcaster,
    # arm_controller, etc.).
    # =========================================================

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            robot_description,
            controllers_file,
        ],
        output="screen",
    )

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
        world_arg,
        gazebo,
        clock_bridge,
        camera_bridge,
        depth_camera_bridge,
        robot_state_publisher,
        controller_manager,
        spawn_robot,
        joint_state_broadcaster,
        arm_controller,
    ])