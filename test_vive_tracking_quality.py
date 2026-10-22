"""Offline tests for the read-only Vive quality metrics."""

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import vive_tracking_quality as quality
from vive_tracking_quality import analyze


def pose(x=0.0, angle_deg=0.0):
    radians = math.radians(angle_deg)
    matrix = np.eye(4)
    matrix[:2, :2] = [[math.cos(radians), -math.sin(radians)],
                      [math.sin(radians), math.cos(radians)]]
    matrix[0, 3] = x
    return matrix


_DEFAULT = object()


def sample(index, *, left=_DEFAULT, right=_DEFAULT, baseline=1.7, base_count=2,
           changed=False, time_s=None):
    left = pose() if left is _DEFAULT else left
    right = pose() if right is _DEFAULT else right
    return SimpleNamespace(
        time_s=index / 90 if time_s is None else time_s,
        poses={"left": left, "right": right},
        states={
            side: {"connected": True, "valid": matrix is not None,
                   "tracking_result": 200}
            for side, matrix in (("left", left), ("right", right))
        },
        reference_changed=changed,
        base_station_count=base_count,
        base_station_baseline_m=baseline,
    )


class TrackingQualityTests(unittest.TestCase):
    def test_perfect_static_tracking_passes_and_has_zero_jitter(self):
        rows = [sample(index) for index in range(900)]
        report = analyze(rows, "static", duration_s=10)
        self.assertTrue(report["calibration_readiness"]["pass"])
        self.assertEqual(report["simultaneous"]["both_trackers_valid_ratio"], 1)
        self.assertEqual(report["simultaneous"]["valid_ratio"], 1)
        self.assertAlmostEqual(report["base_stations"]["baseline_m"]["median"], 1.7)
        self.assertAlmostEqual(report["trackers"]["left"]["static_jitter"]["position_rms_mm"], 0)
        self.assertAlmostEqual(report["trackers"]["right"]["static_jitter"]["orientation_p95_deg"], 0)
        self.assertAlmostEqual(report["sample_interval_ms"]["p95"], 1000 / 90)
        json.dumps(report)

    def test_static_jitter_uses_3d_displacement_and_geodesic_angle(self):
        rows = [sample(index,
                       left=pose(.001 if index % 2 else -.001,
                                 1.0 if index % 2 else -1.0))
                for index in range(900)]
        metrics = analyze(rows, "static", duration_s=10)["trackers"]["left"]["static_jitter"]
        self.assertAlmostEqual(metrics["position_rms_mm"], 1, places=5)
        self.assertAlmostEqual(metrics["position_p95_mm"], 1, places=5)
        self.assertAlmostEqual(metrics["orientation_rms_deg"], 1, places=5)
        self.assertAlmostEqual(metrics["orientation_p95_deg"], 1, places=5)

    def test_three_consecutive_dropouts_fail_even_above_98_percent(self):
        rows = [sample(index, right=None if 100 <= index <= 102 else pose())
                for index in range(900)]
        report = analyze(rows, "sweep", duration_s=10)
        self.assertAlmostEqual(report["simultaneous"]["valid_ratio"], 897 / 900)
        self.assertEqual(report["trackers"]["right"]["longest_invalid_streak_frames"], 3)
        self.assertEqual(report["simultaneous"]["longest_invalid_streak_frames"], 3)
        self.assertFalse(report["calibration_readiness"]["pass"])
        self.assertNotIn("static_jitter", report["trackers"]["left"])

    def test_old_ten_centimeter_baseline_fails(self):
        rows = [sample(index, baseline=.129) for index in range(900)]
        report = analyze(rows, "sweep", duration_s=10)
        self.assertEqual(report["simultaneous"]["valid_frames"], 0)
        self.assertEqual(report["simultaneous"]["both_trackers_valid_frames"], 900)
        self.assertFalse(report["calibration_readiness"]["pass"])

    def test_explicit_close_layout_is_read_only_and_keeps_other_gates(self):
        limits = (.22, .26)
        rows = [sample(index, baseline=.240) for index in range(900)]
        default = analyze(rows, "sweep", duration_s=10)
        self.assertFalse(default["calibration_readiness"]["pass"])

        report = analyze(rows, "sweep", duration_s=10, baseline_limits_m=limits)
        self.assertTrue(report["calibration_readiness"]["pass"])
        self.assertEqual(report["base_stations"]["required_baseline_m"], [.22, .26])
        self.assertTrue(report["base_stations"]["close_layout_read_only_test"])
        self.assertIn("不保证之后整个工作区", report["calibration_readiness"]["meaning"])

        for changed_rows in (
            [sample(index, baseline=.129) for index in range(900)],
            [sample(index, baseline=None if index == 50 else .240)
             for index in range(900)],
            [sample(index, baseline=.240, base_count=1 if index == 50 else 2)
             for index in range(900)],
            [sample(index, baseline=.240, changed=index == 50)
             for index in range(900)],
        ):
            with self.subTest(bad_frame=changed_rows[50]):
                failed = analyze(changed_rows, "sweep", duration_s=10,
                                 baseline_limits_m=limits)
                self.assertFalse(failed["calibration_readiness"]["pass"])

    def test_cli_close_layout_records_test_range_in_json(self):
        rows = [sample(index, baseline=.240) for index in range(900)]

        class Pair:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "quality.json"
            stdout = io.StringIO()
            with (patch.object(quality, "_user_vrserver_running", return_value=True),
                  patch("vive_tianji.tracker_input.TrackerPair", return_value=Pair()),
                  patch.object(quality, "collect", return_value=rows),
                  contextlib.redirect_stdout(stdout)):
                code = quality.main(["--mode", "sweep", "--seconds", "10",
                                     "--expected-baseline-m", "0.240",
                                     "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertIn("只读采样追踪质量通过", stdout.getvalue())
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["base_stations"]["required_baseline_m"],
                             [.22, .26])
            self.assertTrue(report["base_stations"]["close_layout_read_only_test"])

    def test_baseline_missing_one_frame_fails_conservatively(self):
        rows = [sample(index, baseline=None if index == 50 else 1.7)
                for index in range(900)]
        report = analyze(rows, "sweep", duration_s=10)
        self.assertEqual(report["base_stations"]["both_named_valid_frames"], 899)
        self.assertFalse(report["calibration_readiness"]["pass"])

    def test_reference_reset_and_interval_spike_fail(self):
        rows = [sample(index, changed=index == 100,
                       time_s=(index / 90 + (.1 if index >= 200 else 0)))
                for index in range(900)]
        report = analyze(rows, "sweep", duration_s=10)
        self.assertGreater(report["sample_interval_ms"]["max"], 50)
        self.assertEqual(report["reference_changed_frames"], 1)
        self.assertFalse(report["calibration_readiness"]["pass"])

    def test_invalid_end_tail_fails(self):
        rows = [sample(index, left=None if index == 899 else pose())
                for index in range(900)]
        report = analyze(rows, "sweep", duration_s=10)
        self.assertEqual(report["simultaneous"]["valid_tail_frames"], 0)
        self.assertFalse(report["calibration_readiness"]["pass"])


if __name__ == "__main__":
    unittest.main()
