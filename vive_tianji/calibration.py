"""Geometry-only calibration helpers for a two-wrist Vive teleoperator.

No device or robot connection is made here.  All positions are in metres.
``basis_from_pair`` returns a matrix that maps SteamVR Standing vectors into
robot coordinates (+X forward, +Y left, +Z up).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PivotResult:
    """Tracker-local wrist-centre offset and fit diagnostics.

    ``samples`` counts accepted inliers; rejected outliers are excluded.
    """

    offset_tracker_m: np.ndarray
    pivot_world_m: np.ndarray
    rms_m: float
    max_m: float
    rotation_span_deg: float
    condition_number: float
    samples: int


def _fit_pivot(rotations: np.ndarray, positions: np.ndarray,
               weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit ``p_i + R_i offset = pivot`` by weighted least squares."""
    count = len(positions)
    design = np.empty((3 * count, 6), dtype=np.float64)
    design[:, :3] = rotations.reshape(-1, 3)
    design[:, 3:] = np.tile(-np.eye(3), (count, 1))
    rhs = -positions.reshape(-1)
    if weights is not None:
        root_weights = np.repeat(np.sqrt(weights), 3)
        design = design * root_weights[:, None]
        rhs = rhs * root_weights
    singular_values = np.linalg.svd(design, compute_uv=False)
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf")
    estimate, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    return estimate[:3], estimate[3:], condition


def _rotation_span_deg(rotations: np.ndarray) -> float:
    # tr(R_i.T @ R_j) is the dot product of the flattened rotation matrices.
    flat = rotations.reshape(len(rotations), 9)
    cos_angles = np.clip(((flat @ flat.T) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angles)).max())


def solve_pivot(poses: list[np.ndarray]) -> PivotResult:
    """Find a fixed wrist centre from tracker rotations about that centre.

    Capture at least eight diverse, settled orientations while keeping the
    anatomical wrist centre still.  A single-axis sweep cannot resolve the
    offset component along that axis and is rejected.  Up to 20% isolated
    outliers may be trimmed; a generally noisy or slipping mount is rejected.
    The returned residual describes fit repeatability, not absolute accuracy.
    """
    try:
        transforms = np.asarray(poses, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("pivot poses must be finite 4x4 transforms") from exc
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or len(transforms) < 8:
        raise ValueError("pivot calibration requires at least eight 4x4 poses")
    if not np.isfinite(transforms).all():
        raise ValueError("pivot poses contain non-finite values")
    if not np.allclose(transforms[:, 3, :], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError("pivot poses have invalid homogeneous rows")

    rotations = transforms[:, :3, :3]
    positions = transforms[:, :3, 3]
    identity = np.eye(3)
    orthogonality = np.matmul(rotations.transpose(0, 2, 1), rotations)
    if (np.max(np.abs(orthogonality - identity)) > 1e-3
            or np.max(np.abs(np.linalg.det(rotations) - 1.0)) > 1e-3):
        raise ValueError("pivot poses require proper rotation matrices")
    if _rotation_span_deg(rotations) < 20.0:
        raise ValueError("pivot calibration needs at least 20 degrees of rotation span")

    # Huber-style reweighting prevents one bad sample from skewing the fit
    # enough that the good samples are misclassified as outliers.
    weights = np.ones(len(transforms), dtype=np.float64)
    for _ in range(12):
        offset, pivot, _ = _fit_pivot(rotations, positions, weights)
        residuals = np.linalg.norm(positions + rotations @ offset - pivot, axis=1)
        cutoff = max(0.004, 1.5 * float(np.median(residuals)))
        new_weights = np.minimum(1.0, cutoff / np.maximum(residuals, 1e-12))
        if np.max(np.abs(new_weights - weights)) < 1e-3:
            break
        weights = new_weights

    residuals = np.linalg.norm(positions + rotations @ offset - pivot, axis=1)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    outlier_limit = max(0.008, median + 3.0 * max(1.4826 * mad, 0.001))
    inliers = residuals <= outlier_limit
    rejected = len(inliers) - int(inliers.sum())
    if rejected > len(inliers) // 5 or int(inliers.sum()) < 8:
        raise ValueError("too many inconsistent pivot samples")

    good_rotations, good_positions = rotations[inliers], positions[inliers]
    span = _rotation_span_deg(good_rotations)
    if span < 20.0:
        raise ValueError("accepted pivot samples do not span 20 degrees")
    offset, pivot, condition = _fit_pivot(good_rotations, good_positions)
    if not np.isfinite(condition) or condition > 100.0:
        raise ValueError("pivot rotations are ill-conditioned; vary at least two axes")
    if np.linalg.norm(offset) > 0.25:
        raise ValueError("fitted wrist-centre offset exceeds 25 cm; check the mount and samples")
    errors = np.linalg.norm(good_positions + good_rotations @ offset - pivot, axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))
    max_error = float(np.max(errors))
    if rms > 0.008 or max_error > 0.020:
        raise ValueError(f"pivot fit is too noisy: RMS {rms * 1000:.1f} mm, max {max_error * 1000:.1f} mm")

    return PivotResult(
        offset_tracker_m=offset.copy(),
        pivot_world_m=pivot.copy(),
        rms_m=rms,
        max_m=max_error,
        rotation_span_deg=span,
        condition_number=condition,
        samples=len(good_positions),
    )


def measured_offset_from_mm(values: list[float] | tuple[float, float, float]) -> np.ndarray:
    """Validate a *measured* tracker-origin to wrist-centre vector.

    Components are signed OpenVR tracker-local X/Y/Z millimetres, from the
    Tracker 3.0 mounting-screw datum to the anatomical wrist centre.  A zero
    vector is usually an unmeasured placeholder for a tracker on the hand back.
    These physical bounds catch unit mistakes; they do not certify the actual
    measurement or replace a good pivot fit.
    """
    try:
        offset_mm = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('measured offset needs three numeric tracker-local millimetre values') from exc
    if offset_mm.shape != (3,) or not np.isfinite(offset_mm).all():
        raise ValueError('measured offset needs three finite tracker-local millimetre values')
    distance_mm = float(np.linalg.norm(offset_mm))
    if distance_mm < 5.0 or distance_mm > 250.0:
        raise ValueError('measured wrist-centre offset must be 5–250 mm from tracker origin')
    return offset_mm / 1000.0


def basis_from_pair(left_position: np.ndarray, right_position: np.ndarray) -> np.ndarray:
    """Return SteamVR Standing displacement to robot-base direction mapping.

    At calibration, stand facing the robot with uncrossed hands at roughly the
    same height.  The left-to-right separation determines the lateral axis;
    SteamVR Standing +Y supplies gravity-up.  As with any two-point alignment,
    the hand posture must be deliberate and repeatable.
    """
    left = np.asarray(left_position, dtype=np.float64)
    right = np.asarray(right_position, dtype=np.float64)
    if left.shape != (3,) or right.shape != (3,) or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("left and right positions must be finite 3D vectors")
    difference = left - right
    horizontal = difference.copy()
    horizontal[1] = 0.0
    separation = float(np.linalg.norm(horizontal))
    if separation < 0.18:
        raise ValueError("wrist separation must be at least 18 cm for axis alignment")
    if abs(float(difference[1])) > 0.12:
        raise ValueError("keep both wrists within 12 cm of the same height")

    left_axis = horizontal / separation
    up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward_axis = np.cross(left_axis, up_axis)
    forward_axis /= np.linalg.norm(forward_axis)
    # Rows are robot +X forward, +Y left, +Z up expressed in SteamVR.
    return np.stack((forward_axis, left_axis, up_axis), axis=0)


def basis_from_forward_motion(
    neutral_left: np.ndarray,
    neutral_right: np.ndarray,
    moved_left: np.ndarray,
    moved_right: np.ndarray,
) -> np.ndarray:
    """Estimate robot axes from a deliberate, simultaneous two-wrist forward move.

    The operator first holds uncrossed wrists at similar height, then moves both
    wrists straight toward the robot.  The averaged horizontal displacement is
    robot +X.  SteamVR Standing +Y supplies up, and the right-handed cross
    product supplies robot +Y left.  The wrist separation is used only as a
    consistency check, so a modest fore/aft wrist offset cannot bias yaw.

    Returns a 3x3 rotation mapping SteamVR Standing vectors to robot vectors.
    Geometry alone cannot tell whether the operator actually moved *toward*
    the robot; that direction is an explicit calibration instruction.
    """
    inputs = (neutral_left, neutral_right, moved_left, moved_right)
    points = [np.asarray(point, dtype=np.float64) for point in inputs]
    if any(point.shape != (3,) or not np.isfinite(point).all() for point in points):
        raise ValueError("forward-motion calibration requires four finite 3D positions")
    left0, right0, left1, right1 = points

    baseline = left0 - right0
    lateral = baseline.copy()
    lateral[1] = 0.0
    lateral_size = float(np.linalg.norm(lateral))
    if lateral_size < 0.18:
        raise ValueError("neutral wrists must be horizontally separated by at least 18 cm")
    if abs(float(baseline[1])) > 0.12:
        raise ValueError("neutral wrists must be within 12 cm of the same height")

    movements = np.stack((left1 - left0, right1 - right0))
    if np.max(np.abs(movements[:, 1])) > 0.025:
        raise ValueError("each forward movement must have at most 2.5 cm vertical drift")
    horizontal = movements.copy()
    horizontal[:, 1] = 0.0
    lengths = np.linalg.norm(horizontal, axis=1)
    if np.min(lengths) < 0.08:
        raise ValueError("both wrists must move forward at least 8 cm")
    if abs(float(lengths[0] - lengths[1])) / float(np.mean(lengths)) > 0.25:
        raise ValueError("left and right forward distances differ by more than 25 percent")
    cosine_between = float(np.dot(horizontal[0], horizontal[1]) / (lengths[0] * lengths[1]))
    angle_between = float(np.degrees(np.arccos(np.clip(cosine_between, -1.0, 1.0))))
    if angle_between > 10.0:
        raise ValueError("left and right motion directions differ by more than 10 degrees")

    forward = np.mean(horizontal, axis=0)
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)

    # Do not let a crossed-hand or strongly fore/aft pose silently reverse or
    # contaminate the intended robot yaw.  Small fore/aft mismatch is allowed.
    neutral_left_direction = lateral / lateral_size
    left_agreement = float(np.dot(neutral_left_direction, left))
    if left_agreement < np.cos(np.deg2rad(25.0)):
        raise ValueError("neutral wrist line is not approximately lateral (within 25 degrees)")

    return np.stack((forward, left, up), axis=0)
