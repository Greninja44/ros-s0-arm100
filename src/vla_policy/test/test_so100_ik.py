"""Unit tests for the hand-derived FK/IK in vla_policy.so100_ik.

so100_ik.py hand-codes forward kinematics from the URDF and solves IK
numerically with no external verification (MoveIt's own KDL plugin can't
check it -- see that module's docstring for why). These tests exist so a
typo in the chain table, a sign error in the TCP offset, or a regression
in the safety margin around joint limits gets caught before it turns into
a silent mis-grasp in simulation, the way the TCP offset / limit-margin
bugs fixed on this branch were only found by hand.
"""

import numpy as np
import pytest

from vla_policy.so100_ik import (
    ARM_JOINTS,
    JOINT_LIMITS,
    _LOCAL_APPROACH,
    _SOLVE_LIMITS,
    _TCP_OFFSET,
    fk,
    rotation_from_rpy,
    solve_ik,
    solve_pose_ik,
)


def test_arm_joints_matches_chain_length():
    assert len(ARM_JOINTS) == 5
    assert len(JOINT_LIMITS) == 5
    assert len(_SOLVE_LIMITS) == 5


@pytest.mark.parametrize(
    "q",
    [
        [0, 0, 0, 0, 0],
        [0.5, 1.0, -1.5, 0.3, -0.2],
        [-1.8, 3.0, -2.9, -2.4, 3.1],
    ],
)
def test_fk_returns_valid_rigid_transform(q):
    T = fk(q)
    assert T.shape == (4, 4)
    R = T[:3, :3]
    # Rotation block must be a proper rotation: orthonormal, determinant 1.
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])


def test_tcp_offset_points_along_local_approach_axis():
    # The offset from "gripper" to "gripper_tcp" must point along the
    # gripper's own local approach axis (see urdf/so100.urdf.xacro): that
    # is what makes the offset extend the pinch point further out along
    # the direction the jaws face, rather than sideways. Getting this
    # wrong was the root cause of a real mis-grasp bug on this branch.
    offset_dir = _TCP_OFFSET / np.linalg.norm(_TCP_OFFSET)
    assert np.isclose(np.dot(offset_dir, _LOCAL_APPROACH), 1.0, atol=1e-9)


@pytest.mark.parametrize("rpy", [
    (0, 0, 0),
    (0.3, -1.2, 2.5),
    (np.pi, np.pi / 2, -np.pi / 3),
])
def test_rotation_from_rpy_is_a_valid_rotation(rpy):
    R = rotation_from_rpy(*rpy)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_solve_ik_reaches_a_known_reachable_point():
    # (0.20, 0, 0.095) is the pick_place.world cube position -- verified
    # by hand to solve with ~zero error, and this is the exact point the
    # simulated arm has to reach top-down to grasp it.
    target = [0.20, 0.0, 0.095]
    q, pos_err, orient_err, ok = solve_ik(
        target,
        target_approach=(0, 0, -1),
        constrain_orientation=True,
        n_starts=20,
        pos_tolerance=0.01,
        seed_rng=0,
    )
    assert ok
    assert pos_err < 0.005
    T = fk(q)
    np.testing.assert_allclose(T[:3, 3], target, atol=0.01)


def test_solve_ik_solutions_stay_within_the_safety_margin():
    # Every returned solution -- reachable or not -- must stay inside the
    # margined limits (JOINT_LIMITS shrunk by _LIMIT_MARGIN), never just
    # the raw URDF limits: that margin is what leaves controller
    # tracking/settling error room to not overshoot into a hard limit.
    for target in ([0.20, 0.0, 0.095], [-0.20, 0.0, 0.105], [5.0, 0.0, 5.0]):
        q, _, _, _ = solve_ik(target, n_starts=10, seed_rng=0)
        for qi, (lo, hi) in zip(q, _SOLVE_LIMITS):
            assert lo - 1e-6 <= qi <= hi + 1e-6


def test_solve_ik_reports_failure_for_an_unreachable_target():
    target = [5.0, 0.0, 5.0]
    _, _, _, ok = solve_ik(target, n_starts=10, pos_tolerance=0.01, seed_rng=0)
    assert not ok


def test_solve_ik_unconstrained_orientation_is_never_worse_than_constrained():
    # Position-only IK has two extra redundant DOF to work with, so for
    # the same target it should never end up with *more* position error
    # than the orientation-constrained solve -- if it ever did, that
    # would point at a scoring/weighting bug in solve_ik.
    target = [0.25, 0.0, 0.10]
    _, pos_err_constrained, _, _ = solve_ik(
        target, constrain_orientation=True, n_starts=20, seed_rng=1
    )
    _, pos_err_unconstrained, _, _ = solve_ik(
        target, constrain_orientation=False, n_starts=20, seed_rng=1
    )
    assert pos_err_unconstrained <= pos_err_constrained + 1e-6


def test_solve_pose_ik_is_a_no_op_when_seed_already_matches_target():
    seed = [0.1, 0.5, -0.5, 0.2, 0.0]
    T = fk(seed)
    q, pos_err, rot_err = solve_pose_ik(T[:3, 3], T[:3, :3], seed)
    assert pos_err < 1e-4
    assert rot_err < 1e-4
    np.testing.assert_allclose(q, seed, atol=1e-4)
