from setuptools import find_packages, setup

package_name = "vla_policy"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    extras_require={
        "test": ["pytest"],
    },
    zip_safe=True,
    maintainer="adarsh",
    maintainer_email="adarshvijaya22@gmail.com",
    description="VLA policy nodes for the SO-100 arm: Octo, OpenVLA, SAM2, Diffusion Policy, and more",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            # Original nodes
            "octo_policy = vla_policy.octo_policy_node:main",
            "gripper_demo = vla_policy.gripper:main",
            "pick_place = vla_policy.pick_place:main",
            "teleop_keyboard = vla_policy.teleop_keyboard:main",
            "collect_data = vla_policy.collect_data:main",
            "vla_pick_place = vla_policy.vla_pick_place:main",
            # New VLA model nodes
            "openvla_policy = vla_policy.openvla_policy_node:main",
            "diffusion_policy = vla_policy.diffusion_policy:main",
            # Perception nodes
            "sam2_segmentation = vla_policy.sam2_segmentation:main",
            "pointcloud_processor = vla_policy.pointcloud_processor:main",
            # Tactile sensing
            "tactile_sensing = vla_policy.tactile_sensing:main",
            # Task planning
            "task_decomposer = vla_policy.task_decomposer:main",
        ],
    },
)
