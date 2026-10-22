"""Synthetic geometry tests; no SteamVR or robot hardware required."""

from __future__ import annotations

import unittest

import numpy as np

from vive_tianji.calibration import (basis_from_forward_motion, basis_from_pair,
                                     measured_offset_from_mm, solve_pivot)


def _rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(degrees)
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def _pivot_samples(count: int = 30, noise_m: float = 0.0005) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(104)
    offset = np.array([0.026, -0.039, 0.058])
    pivot = np.array([1.1, 0.8, -0.4])
    poses: list[np.ndarray] = []
    for _ in range(count):
        matrix = np.eye(4)
        matrix[:3, :3] = _rotation(rng.normal(size=3), rng.uniform(-55, 55))
        matrix[:3, 3] = pivot - matrix[:3, :3] @ offset + rng.normal(scale=noise_m, size=3)
        poses.append(matrix)
    return poses, offset, pivot


class PivotCalibrationTests(unittest.TestCase):
    def test_recovers_offset_and_pivot(self) -> None:
        poses, offset, pivot = _pivot_samples()
        result = solve_pivot(poses)
        self.assertLess(np.linalg.norm(result.offset_tracker_m - offset), 0.002)
        self.assertLess(np.linalg.norm(result.pivot_world_m - pivot), 0.002)
        self.assertLess(result.rms_m, 0.002)
        self.assertEqual(result.samples, len(poses))

    def test_trims_one_bad_translation(self) -> None:
        poses, offset, _ = _pivot_samples()
        poses[5] = poses[5].copy()
        poses[5][:3, 3] += np.array([0.08, -0.12, 0.05])
        result = solve_pivot(poses)
        self.assertEqual(result.samples, len(poses) - 1)
        self.assertLess(np.linalg.norm(result.offset_tracker_m - offset), 0.003)

    def test_rejects_single_axis_rotation(self) -> None:
        offset = np.array([0.02, 0.03, 0.04])
        pivot = np.array([1.0, 0.5, 0.0])
        poses = []
        for angle in np.linspace(-60, 60, 20):
            matrix = np.eye(4)
            matrix[:3, :3] = _rotation(np.array([0.0, 0.0, 1.0]), angle)
            matrix[:3, 3] = pivot - matrix[:3, :3] @ offset
            poses.append(matrix)
        with self.assertRaisesRegex(ValueError, "ill-conditioned"):
            solve_pivot(poses)

    def test_rejects_noisy_and_low_motion_sets(self) -> None:
        poses, _, _ = _pivot_samples(noise_m=0.025)
        with self.assertRaises(ValueError):
            solve_pivot(poses)
        poses, _, _ = _pivot_samples()
        for matrix in poses:
            matrix[:3, :3] = np.eye(3)
        with self.assertRaisesRegex(ValueError, "rotation span"):
            solve_pivot(poses)

    def test_rejects_invalid_poses(self) -> None:
        poses, _, _ = _pivot_samples()
        poses[0][:3, :3] *= 1.1
        with self.assertRaisesRegex(ValueError, "proper rotation"):
            solve_pivot(poses)


class MeasuredOffsetTests(unittest.TestCase):
    def test_converts_signed_tracker_local_millimetres(self) -> None:
        np.testing.assert_allclose(measured_offset_from_mm([12.5, -37., 61.]),
                                   [.0125, -.037, .061])

    def test_rejects_missing_nan_units_and_placeholder_values(self) -> None:
        for invalid in ([0., 0., 0.], [4., 0., 0.], [251., 0., 0.],
                        [float('nan'), 10., 0.], [10., 20.],
                        [10., 'abc', 0.]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                measured_offset_from_mm(invalid)


class PairBasisTests(unittest.TestCase):
    def test_standard_steamvr_axes(self) -> None:
        mapping = basis_from_pair(np.array([-0.3, 1.2, 0.0]), np.array([0.3, 1.2, 0.0]))
        # SteamVR +X right, +Y up, +Z back -> robot +Y left, +Z up, +X forward.
        np.testing.assert_allclose(mapping @ [0.0, 0.0, -1.0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(mapping @ [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(mapping @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(mapping @ mapping.T, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(mapping), 1.0)

    def test_rejects_ambiguous_hand_pose(self) -> None:
        with self.assertRaisesRegex(ValueError, "separation"):
            basis_from_pair([0.01, 1.0, 0.0], [0.0, 1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "same height"):
            basis_from_pair([-0.3, 1.0, 0.0], [0.3, 1.2, 0.0])


class ForwardMotionBasisTests(unittest.TestCase):
    def test_avoids_yaw_bias_from_fore_aft_wrist_offset(self) -> None:
        left0 = np.array([-0.3, 1.2, 0.10])
        right0 = np.array([0.3, 1.2, 0.0])
        forward = np.array([0.0, 0.0, -1.0])
        mapping = basis_from_forward_motion(left0, right0, left0 + 0.12 * forward, right0 + 0.12 * forward)
        np.testing.assert_allclose(mapping @ forward, [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(mapping @ mapping.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(mapping), 1.0)

        pair_mapping = basis_from_pair(left0, right0)
        pair_forward = pair_mapping[0]
        pair_yaw_error = np.degrees(np.arccos(np.clip(np.dot(pair_forward, forward), -1.0, 1.0)))
        self.assertGreater(pair_yaw_error, 9.0)

    def test_recovers_rotated_work_area_heading(self) -> None:
        yaw = np.deg2rad(32.0)
        forward = np.array([-np.sin(yaw), 0.0, -np.cos(yaw)])
        left_axis = np.cross([0.0, 1.0, 0.0], forward)
        left0 = np.array([0.0, 1.2, 0.0]) + 0.3 * left_axis
        right0 = np.array([0.0, 1.2, 0.0]) - 0.3 * left_axis
        mapping = basis_from_forward_motion(left0, right0, left0 + 0.14 * forward, right0 + 0.13 * forward)
        np.testing.assert_allclose(mapping @ forward, [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(mapping @ left_axis, [0.0, 1.0, 0.0], atol=1e-12)

    def test_rejects_short_or_inconsistent_forward_motion(self) -> None:
        left0 = np.array([-0.3, 1.2, 0.0])
        right0 = np.array([0.3, 1.2, 0.0])
        with self.assertRaisesRegex(ValueError, "at least 8 cm"):
            basis_from_forward_motion(left0, right0, left0 + [0.0, 0.0, -0.07], right0 + [0.0, 0.0, -0.12])
        with self.assertRaisesRegex(ValueError, "vertical drift"):
            basis_from_forward_motion(left0, right0, left0 + [0.0, 0.03, -0.12], right0 + [0.0, 0.0, -0.12])
        with self.assertRaisesRegex(ValueError, "distances differ"):
            basis_from_forward_motion(left0, right0, left0 + [0.0, 0.0, -0.10], right0 + [0.0, 0.0, -0.16])
        with self.assertRaisesRegex(ValueError, "directions differ"):
            basis_from_forward_motion(left0, right0, left0 + [0.0, 0.0, -0.12], right0 + [0.035, 0.0, -0.12])

    def test_rejects_ambiguous_neutral_wrist_geometry(self) -> None:
        left0 = np.array([-0.3, 1.2, 0.0])
        right0 = np.array([0.3, 1.2, 0.0])
        move = np.array([0.0, 0.0, -0.12])
        with self.assertRaisesRegex(ValueError, "horizontally separated"):
            basis_from_forward_motion([0.0, 1.2, 0.0], [0.1, 1.2, 0.0], [0.0, 1.2, -0.12], [0.1, 1.2, -0.12])
        with self.assertRaisesRegex(ValueError, "approximately lateral"):
            # The wrist line has a large fore/aft component despite good common motion.
            skewed_left = left0 + [0.0, 0.0, 0.35]
            basis_from_forward_motion(skewed_left, right0, skewed_left + move, right0 + move)
        with self.assertRaisesRegex(ValueError, "approximately lateral"):
            basis_from_forward_motion(right0, left0, right0 + move, left0 + move)


if __name__ == "__main__":
    unittest.main()
