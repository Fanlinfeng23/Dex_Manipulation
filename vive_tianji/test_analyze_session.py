"""Offline fixtures for the Vive session-quality report; no hardware is opened."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from .analyze_session import analyze_stream, format_report


class AnalyzeSessionTest(unittest.TestCase):
    TRACKER = {"position_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}

    def _session(self, rows):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "stream.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _control(fk_mm, measured_mm, joint_error, frame_age, queue_delay):
        return {
            "following": True,
            "frame_age_ms": frame_age,
            "queue_delay_ms": queue_delay,
            "target_position_m": {"left": [0, 0, 0], "right": [0, 0, 0]},
            "fk_position_m": {"left": [fk_mm / 1000, 0, 0],
                              "right": [fk_mm / 1000, 0, 0]},
            "measured_position_m": {"left": ([measured_mm / 1000, 0, 0]
                                           if measured_mm is not None else None),
                                    "right": None},
            "tracking_error_deg": {"left": joint_error, "right": None},
        }

    def test_tracking_and_following_error_are_separate(self):
        rows = [
            {"t": 0.00, "seq": 1, "valid": True,
             "tracker": {"left": self.TRACKER, "right": self.TRACKER},
             "control": self._control(1, 2, 1, 5, 2)},
            {"t": 0.01, "seq": 2, "valid": False,
             "tracker": {"left": self.TRACKER, "right": None}},
            {"t": 0.02, "seq": 3, "valid": False,
             "tracker": {"left": self.TRACKER, "right": None}},
            {"t": 0.03, "seq": 4, "valid": True,
             "tracker": {"left": self.TRACKER, "right": self.TRACKER}},
            # The controller gate may be false for reasons other than a lost wrist.
            {"t": 0.04, "seq": 6, "valid": False,
             "tracker": {"left": self.TRACKER, "right": self.TRACKER},
             "control": self._control(3, None, 3, 15, 6)},
        ]
        report = analyze_stream(self._session(rows))
        self.assertEqual(report["tracking"]["left"]["valid_samples"], 5)
        self.assertEqual(report["tracking"]["right"]["valid_samples"], 3)
        self.assertEqual(report["tracking"]["both"]["valid_samples"], 3)
        self.assertEqual(report["tracking"]["gate"]["valid_samples"], 2)
        self.assertEqual(report["tracking"]["right"]["longest_loss_frames"], 2)
        self.assertAlmostEqual(report["tracking"]["right"]["longest_loss_ms_approx"], 20)
        self.assertEqual(report["missing_sequence_rows"], 1)
        self.assertEqual(report["following_control_samples"], 2)
        self.assertAlmostEqual(report["timing"]["frame_age_ms"]["p95"], 14.5)
        left = report["following_internal_error"]["left"]
        self.assertAlmostEqual(left["target_to_fk_mm"]["p50"], 2)
        self.assertEqual(left["target_to_measured_mm"]["samples"], 1)
        self.assertAlmostEqual(left["target_to_measured_mm"]["p50"], 2)
        self.assertAlmostEqual(left["joint_tracking_error_deg"]["p50"], 2)
        self.assertIn("不等于外部测量", format_report(report))

    def test_preview_without_following_or_hardware_feedback(self):
        rows = [{"t": 1.0, "seq": 1, "valid": True,
                 "tracker": {"left": self.TRACKER, "right": self.TRACKER},
                 "control": {"following": False, "frame_age_ms": 4.0,
                             "queue_delay_ms": 2.0}}]
        report = analyze_stream(self._session(rows))
        self.assertEqual(report["following_control_samples"], 0)
        self.assertEqual(report["following_internal_error"]["left"]
                         ["target_to_measured_mm"]["samples"], 0)
        self.assertIsNone(report["following_internal_error"]["left"]
                          ["target_to_measured_mm"]["p95"])

    def test_rejects_corrupt_record_instead_of_silently_skipping_it(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "stream.jsonl"
        path.write_text('{"t": 1, "seq": 1}\n{broken}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "JSON 行损坏"):
            analyze_stream(path)


if __name__ == "__main__":
    unittest.main()
