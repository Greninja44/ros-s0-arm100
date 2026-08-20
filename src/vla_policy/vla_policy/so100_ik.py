"""Self-contained analytic-chain numerical IK for the SO-100 arm.

MoveIt's KDLKinematicsPlugin only solves this chain in ``position_only_ik``
mode (5 joints < 6 required for a full pose), which leaves the gripper's
approach orientation arbitrary on every solve -- fine for reaching a point,
useless for grasping, since the jaws end up pointing every which way.

This module computes the same forward kinematics by hand (values taken
directly from urdf/so100.urdf.xacro, and cross-checked against MoveIt's own
/compute_fk service) and solves a damped-least-squares IK for position
*and* a chosen approach direction (e.g. straight down) with multi-start
search to escape local minima / joint-limit lock. No PyKDL/kdl_parser
dependency -- kdl_parser_py isn't available on Jazzy.
"""

import numpy as np

ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# (origin_xyz, origin_rpy, joint_axis) for each joint, base -> gripper.
_CHAIN = [
    ((0, -0.0452, 0.0165), (1.57079, 0, 0), (0, 1, 0)),   # shoulder_pan
    ((0, 0.1025, 0.0306), (-1.8, 0, 0), (1, 0, 0)),       # shoulder_lift
    ((0, 0.11257, 0.028), (1.57079, 0, 0), (1, 0, 0)),    # elbow_flex
    ((0, 0.0052, 0.1349), (-1, 0, 0), (1, 0, 0)),         # wrist_flex
    ((0, -0.0601, 0), (0, 1.57079, 0), (0, 1, 0)),        # wrist_roll
]

# gripper -> gripper_tcp fixed offset (pinch point, see urdf/so100.urdf.xacro)
_TCP_OFFSET = np.array([-0.019, -0.095, 0.0])

# gripper_tcp's local "pointing" axis (the direction the jaws face)
_LOCAL_APPROACH = np.array([0.0, -1.0, 0.0])

# A solution that lands exactly on a URDF joint limit leaves no room for
# controller tracking/settling error: by the time the arm is *at* the
# target and something else reads its state back, it can be a hair past
# the limit and get rejected as "out of bounds" on the next plan. Solving
# against a slightly tighter range keeps every solution a small margin
# inside the true physical limits.
_LIMIT_MARGIN = 0.02

JOINT_LIMITS = [
    (-2.0, 2.0),
    (-0.001, 3.5),
    (-3.14158, 0.001),
    (-2.5, 1.2),
    (-3.14158, 3.14158),
]

# Used to clip solutions during optimization; seeds are still drawn from
# the full JOINT_LIMITS range above.
_SOLVE_LIMITS = [(lo + _LIMIT_MARGIN, hi - _LIMIT_MARGIN) for lo, hi in JOINT_LIMITS]


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot_rpy(r, p, y):
    return _rot_z(y) @ _rot_y(p) @ _rot_x(r)


def _rot_axis(axis, theta):
    if axis == (1, 0, 0):
        return _rot_x(theta)
    if axis == (0, 1, 0):
        return _rot_y(theta)
    if axis == (0, 0, 1):
        return _rot_z(theta)
    raise ValueError(axis)


def fk(q):
    """Forward kinematics: joint values -> 4x4 pose of gripper_tcp in base frame."""
    T = np.eye(4)
    for (xyz, rpy, axis), theta in zip(_CHAIN, q):
        Ti = np.eye(4)
        Ti[:3, :3] = _rot_rpy(*rpy) @ _rot_axis(axis, theta)
        Ti[:3, 3] = xyz
        T = T @ Ti
    Ttcp = np.eye(4)
    Ttcp[:3, 3] = _TCP_OFFSET
    return T @ Ttcp


def _error(q, target_pos, target_approach, orient_weight):
    T = fk(q)
    pos_err = T[:3, 3] - target_pos
    approach_world = T[:3, :3] @ _LOCAL_APPROACH
    orient_err = orient_weight * (approach_world - target_approach)
    return np.concatenate([pos_err, orient_err])


def _solve_once(
    target_pos, target_approach, orient_weight, seed, max_iter=150, damping=0.01
):
    q = np.array(seed, dtype=float)
    for _ in range(max_iter):
        e = _error(q, target_pos, target_approach, orient_weight)
        if np.linalg.norm(e) < 1e-5:
            break
        J = np.zeros((6, 5))
        eps = 1e-6
        for i in range(5):
            dq = q.copy()
            dq[i] += eps
            J[:, i] = (
                _error(dq, target_pos, target_approach, orient_weight) - e
            ) / eps
        step = np.linalg.solve(
            J.T @ J + damping * np.eye(5), -J.T @ e
        )
        q = q + np.clip(step, -0.5, 0.5)
        for i, (lo, hi) in enumerate(_SOLVE_LIMITS):
            q[i] = np.clip(q[i], lo, hi)
    e = _error(q, target_pos, target_approach, orient_weight)
    return q, np.linalg.norm(e[:3]), np.linalg.norm(e[3:])


def solve_ik(
    target_pos,
    target_approach=(0.0, 0.0, -1.0),
    constrain_orientation=True,
    n_starts=60,
    pos_tolerance=0.005,
    seed_rng=0,
):
    """Multi-start damped-least-squares IK for position (+ approach direction).

    With ``constrain_orientation`` (the default), also solves for
    ``gripper_tcp``'s local pointing axis to align with ``target_approach``
    -- needed wherever the gripper actually has to touch something. Pass
    ``constrain_orientation=False`` for transit waypoints where any
    orientation is fine; the 5-DOF arm has much more reach that way, and
    plenty of tests only need the position anyway.

    Returns (joint_values, pos_error, orient_error, ok) where ``ok`` is
    True only if the best solution found is within ``pos_tolerance``
    metres of the target position.
    """
    target_pos = np.asarray(target_pos, dtype=float)
    target_approach = np.asarray(target_approach, dtype=float)
    target_approach = target_approach / np.linalg.norm(target_approach)
    orient_weight = 1.0 if constrain_orientation else 0.0

    rng = np.random.default_rng(seed_rng)
    seeds = [[0.0, 0.0, 0.0, 0.0, 0.0]] + [
        [rng.uniform(lo, hi) for lo, hi in JOINT_LIMITS]
        for _ in range(n_starts - 1)
    ]

    best = None
    for seed in seeds:
        q, pos_err, orient_err = _solve_once(
            target_pos, target_approach, orient_weight, seed
        )
        score = pos_err + 0.3 * orient_err
        if best is None or score < best[3]:
            best = (q, pos_err, orient_err, score)

    q, pos_err, orient_err, _ = best
    return list(q), pos_err, orient_err, pos_err <= pos_tolerance
