"""Unit tests for the classical-CV fallback in vla_policy.sam2_segmentation.

The live Gazebo pipeline (RGB -> color mask -> depth back-projection ->
world-frame grasp pose) hasn't been runnable in this environment (the
depth-camera world's rendering never came up), so these tests cover the
two pieces that don't need a running simulation: the color detector
actually distinguishing colors, and the camera-to-world projection math
being self-consistent. They do NOT substitute for an end-to-end run in
Gazebo -- see the module docstring in sam2_segmentation.py.
"""

import numpy as np
import pytest

from vla_policy.sam2_segmentation import (
    _COLOR_DETECTORS,
    camera_optical_to_world,
    color_from_prompt,
)


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("red cube", "red"),
        ("pick up the BLUE block", "blue"),
        ("the yellow cylinder", "yellow"),
        ("a green ball on the table", "green"),
        ("the object", None),
    ],
)
def test_color_from_prompt(prompt, expected):
    assert color_from_prompt(prompt) == expected


@pytest.mark.parametrize("color", sorted(_COLOR_DETECTORS))
def test_color_detector_finds_its_own_solid_color_image(color):
    solid_colors = {
        "red": (220, 20, 20),
        "blue": (20, 20, 220),
        "green": (20, 220, 20),
        "yellow": (220, 220, 20),
    }
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :] = solid_colors[color]
    mask = _COLOR_DETECTORS[color](img)
    assert mask.all(), f"{color} detector missed a solid {color} image"


@pytest.mark.parametrize("color", sorted(_COLOR_DETECTORS))
def test_color_detector_rejects_gray(color):
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    mask = _COLOR_DETECTORS[color](img)
    assert not mask.any(), f"{color} detector false-positived on neutral gray"


def test_color_detectors_are_mutually_exclusive_on_solid_images():
    # A solid red image shouldn't also register as blue/green/yellow --
    # otherwise multiple prompts would all lock onto the same object.
    solid_colors = {
        "red": (220, 20, 20),
        "blue": (20, 20, 220),
        "green": (20, 220, 20),
        "yellow": (220, 220, 20),
    }
    for true_color, rgb in solid_colors.items():
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[:, :] = rgb
        for name, detector in _COLOR_DETECTORS.items():
            mask = detector(img)
            if name == true_color:
                assert mask.all()
            else:
                assert not mask.any(), (
                    f"{name} detector fired on a solid {true_color} image"
                )


def test_camera_optical_to_world_straight_down_point():
    # For the overhead cameras used here (pitch = pi/2, i.e. pointing
    # straight down), a point on the optical axis (x_opt = y_opt = 0) is
    # directly below the camera by the depth value -- world x/y stay at
    # the camera's own x/y, world z drops by exactly the depth.
    camera_position = [0.1, 0.1, 1.2]
    depth = 0.9
    world = camera_optical_to_world(0.0, 0.0, depth, camera_position, np.pi / 2)
    np.testing.assert_allclose(
        world, [camera_position[0], camera_position[1], camera_position[2] - depth],
        atol=1e-9,
    )


def test_camera_optical_to_world_is_linear_in_depth():
    # Doubling the depth along the optical axis should double the world
    # offset from the camera -- a basic sanity check on the projection
    # (catches e.g. an accidental depth-independent term).
    camera_position = [0.0, 0.0, 1.0]
    p1 = camera_optical_to_world(0.05, 0.02, 0.5, camera_position, np.pi / 2)
    p2 = camera_optical_to_world(0.10, 0.04, 1.0, camera_position, np.pi / 2)
    offset1 = np.array(p1) - camera_position
    offset2 = np.array(p2) - camera_position
    np.testing.assert_allclose(offset2, 2 * offset1, atol=1e-9)


def test_camera_optical_to_world_zero_pitch_looks_along_world_x():
    # At pitch=0 the camera looks along its own unrotated forward axis,
    # which by construction is world +X -- a different, independently
    # checkable case from the pitch=pi/2 (straight down) one every world
    # here actually uses.
    camera_position = [0.0, 0.0, 0.0]
    world = camera_optical_to_world(0.0, 0.0, 1.0, camera_position, 0.0)
    np.testing.assert_allclose(world, [1.0, 0.0, 0.0], atol=1e-9)
