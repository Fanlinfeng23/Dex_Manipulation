"""Small, deterministic constrained IK steps for continuous virtual teleoperation.

Joint coordinates are radians here; the public SDK adapter uses degrees.
Each quadratic subproblem starts from the previous feasible joint state.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class IKResult:
    joints_deg: np.ndarray
    converged: bool
    position_error_m: float
    rotation_error_rad: float
    iterations: int
    limited: bool = False


def pose_error(target, actual):
    return np.r_[target[:3, 3] - actual[:3, 3],
                 Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()]


def bounded_quadratic_step(hessian, gradient, constraints, slack):
    """Primal active-set QP: min .5*x'H*x-g'x, A*x <= b, with feasible x=0."""
    x = np.zeros(7)
    active = []
    inverse_products = np.linalg.solve(hessian, np.column_stack((gradient, constraints.T)))
    free, columns = inverse_products[:, 0], inverse_products[:, 1:]
    # At most a handful of the 7 joint bounds/wrist faces can be active at once.
    for _ in range(40):
        direction = free - x
        if active:
            rows = constraints[active]
            inverse_rows = columns[:, active]
            multipliers = np.linalg.lstsq(rows @ inverse_rows, rows @ direction, rcond=1e-10)[0]
            direction = direction - inverse_rows @ multipliers
        else:
            multipliers = np.empty(0)
        if np.linalg.norm(direction) < 1e-9:
            if not active or np.min(multipliers) >= -1e-8:
                break
            active.pop(int(np.argmin(multipliers)))
            continue
        projection = constraints @ direction
        candidates = np.flatnonzero(projection > 1e-10)
        candidates = [i for i in candidates if i not in active]
        alpha, blocker = 1., None
        for i in candidates:
            fraction = max(0., float((slack[i] - constraints[i] @ x) / projection[i]))
            if fraction < alpha:
                alpha, blocker = fraction, i
        x += alpha * direction
        if blocker is not None:
            active.append(blocker)
    return x


def solve_local(target, reference_deg, fk_jacobian, constraints_at, valid,
                limits_deg, *, max_delta_deg, max_iterations, position_tolerance_m,
                rotation_tolerance_rad, rotation_weight_m, damping, centering):
    reference = np.deg2rad(reference_deg)
    q = reference.copy()
    lower = np.maximum(np.deg2rad(limits_deg[:, 0]), reference - np.deg2rad(max_delta_deg))
    upper = np.minimum(np.deg2rad(limits_deg[:, 1]), reference + np.deg2rad(max_delta_deg))
    span = np.deg2rad(limits_deg[:, 1] - limits_deg[:, 0])
    middle = np.deg2rad(limits_deg.mean(axis=1))
    weights = np.r_[np.ones(3), np.full(3, rotation_weight_m)]
    actual, jacobian = fk_jacobian(q)
    error = pose_error(target, actual)
    score = float(np.linalg.norm(weights * error))
    iterations = 0
    converged = False
    for iterations in range(max_iterations + 1):
        if (np.linalg.norm(error[:3]) <= position_tolerance_m and
                np.linalg.norm(error[3:]) <= rotation_tolerance_rad):
            converged = True
            break
        if iterations == max_iterations:
            break
        j = weights[:, None] * jacobian
        e = weights * error
        # More damping as the smallest task-space singular value approaches zero.
        sigma = float(np.linalg.svd(j, compute_uv=False)[-1])
        regularizer = damping + .02 * max(0., 1. - sigma / .04) ** 2
        hessian = j.T @ j + regularizer ** 2 * np.eye(7)
        # Prefer room on narrow wrist joints; the task remains the primary objective.
        center_step = np.clip(centering * (middle - q) / np.maximum(span, .1) ** 2, -.04, .04)
        gradient = j.T @ e + regularizer ** 2 * center_step
        extra_a, extra_b = constraints_at(q)
        a = np.vstack((np.eye(7), -np.eye(7), extra_a))
        b = np.r_[upper - q, q - lower, extra_b]
        step = bounded_quadratic_step(hessian, gradient, a, np.maximum(b, 0.))
        length = float(np.max(np.abs(step)))
        if length < 1e-9:
            break
        if length > .12:
            step *= .12 / length
        accepted = False
        for fraction in (1., .5, .25, .125, .0625):
            trial = q + fraction * step
            if not valid(np.rad2deg(trial)):
                continue
            trial_actual, trial_jacobian = fk_jacobian(trial)
            trial_error = pose_error(target, trial_actual)
            trial_score = float(np.linalg.norm(weights * trial_error))
            if trial_score < score - 1e-12:
                q, actual, jacobian, error, score = trial, trial_actual, trial_jacobian, trial_error, trial_score
                accepted = True
                break
        if not accepted:
            break
    return IKResult(np.rad2deg(q), converged, float(np.linalg.norm(error[:3])),
                    float(np.linalg.norm(error[3:])), iterations, not converged)
