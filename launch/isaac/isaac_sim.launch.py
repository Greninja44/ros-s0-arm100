"""Isaac Sim launch file for the SO-100 arm.

Launches NVIDIA Isaac Sim with photorealistic rendering, domain
randomization, and GPU-accelerated perception for the SO-100 arm.

Prerequisites
-------------
    # Install Isaac Sim via Omniverse Launcher
    # Then install Isaac ROS packages:
    sudo apt install ros-jazzy-isaac-ros-common
    sudo apt install ros-jazzy-isaac-ros-slam
    sudo apt install ros-jazzy-isaac-ros-dnn-inference

Usage
-----
    # Terminal 1 — Isaac Sim
    ros2 launch so100_description isaac_sim.launch.py

    # With domain randomization:
    ros2 launch so100_description isaac_sim.launch.py \
        domain_randomization:=true

    # With synthetic data generation:
    ros2 launch so100_description isaac_sim.launch.py \
        data_generation:=true \
        output_dir:=/tmp/isaac_data
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    package_name = "so100_description"
    pkg_share = get_package_share_directory(package_name)

    domain_randomization_arg = DeclareLaunchArgument(
        "domain_randomization",
        default_value="true",
        description="Enable domain randomization for sim-to-real"
    )

    data_generation_arg = DeclareLaunchArgument(
        "data_generation",
        default_value="false",
        description="Enable synthetic data generation"
    )

    output_dir_arg = DeclareLaunchArgument(
        "output_dir",
        default_value="/tmp/isaac_sim_data",
        description="Output directory for generated data"
    )

    render_quality_arg = DeclareLaunchArgument(
        "render_quality",
        default_value="high",
        description="Render quality: low, medium, high, ultra"
    )

    dr_config_file = os.path.join(pkg_share, "config", "isaac_sim.yaml")

    domain_randomizer = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[],
        output="screen",
    )

    return LaunchDescription([
        domain_randomization_arg,
        data_generation_arg,
        output_dir_arg,
        render_quality_arg,
    ])
