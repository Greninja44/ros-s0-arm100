"""Smoke test: every vla_policy module must import cleanly.

ament_python's colcon build doesn't actually import each module, so a
plain syntax error or typo'd import can sit in a node file undetected
until someone tries to run it. This just needs ROS 2 + the message
packages already required to build the workspace -- none of these
modules import heavy ML dependencies (Octo/torch) at module scope, only
inside the functions that need them, lazily.
"""

import importlib

import pytest

MODULES = [
    "vla_policy.so100_ik",
    "vla_policy.gripper",
    "vla_policy.pick_place",
    "vla_policy.teleop_keyboard",
    "vla_policy.collect_data",
    "vla_policy.vla_pick_place",
    "vla_policy.octo_policy_node",
    "vla_policy.openvla_policy_node",
    "vla_policy.sam2_segmentation",
    "vla_policy.diffusion_policy",
    "vla_policy.tactile_sensing",
    "vla_policy.task_decomposer",
    "vla_policy.pointcloud_processor",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports_cleanly(module_name):
    importlib.import_module(module_name)
