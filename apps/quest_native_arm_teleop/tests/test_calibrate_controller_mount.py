"""Offline wrist-pivot fitting tests; no Quest or robot hardware is used."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from calibrate_controller_mount import CalibrationError, fit_pivot, load_samples, main


def capture(count=150, mode="multi", wrist_motion=0.0, offset_scale=1.0):
    random = np.random.default_rng(42)
    offsets = {
        "left": np.array([0.08, -0.07, 0.05]) * offset_scale,
        "right": np.array([-0.06, 0.09, 0.04]) * offset_scale,
    }
    pivots = {"left": np.array([-0.25, 1.3, -0.4]),
              "right": np.array([0.25, 1.3, -0.4])}
    samples = {side: [] for side in offsets}
    rows = []
    for index in range(count):
        angle = 2 * np.pi * index / count
        controllers = {}
        for side, phase in (("left", 0.0), ("right", 0.6)):
            if mode == "single":
                rotation = Rotation.from_euler("z", 45 * np.sin(angle + phase), degrees=True)
            else:
                rotation = Rotation.from_euler(
                    "xyz", [32 * np.sin(angle + phase), 27 * np.sin(1.7 * angle + phase),
                            18 * np.cos(1.3 * angle + phase)], degrees=True)
            pivot = pivots[side] + np.array([wrist_motion * np.sin(2.1 * angle),
                                              wrist_motion * np.cos(2.7 * angle), 0])
            position = pivot - rotation.as_matrix() @ offsets[side] + random.normal(0, .0007, 3)
            controllers[side] = {"position": position.tolist(), "quaternion": rotation.as_quat().tolist(),
                                 "tracked": True, "grip": 0.0}
            samples[side].append((position, rotation.as_matrix()))
        rows.append({"kind": "sample", "new_input": True, "input_fresh": True,
                     "state": {"source": "quest"},
                     "input": {"session": "synthetic-session", "reference": 1, "seq": index,
                               "captured_s": 10 + index / 25, "focused": True,
                               "controllers": controllers}})
    return rows, samples, offsets


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PivotCalibrationTests(unittest.TestCase):
    def test_recovers_both_controller_local_offsets_and_filters_nonfresh_rows(self):
        rows, _, expected = capture()
        stale = copy.deepcopy(rows[0]); stale["input"]["seq"] = 1000; stale["new_input"] = False
        lost = copy.deepcopy(rows[1]); lost["input"]["seq"] = 1001
        lost["input"]["controllers"]["right"]["tracked"] = False
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "motion.jsonl"
            write_rows(path, [stale, lost, *rows])
            samples, info = load_samples(path)
            self.assertEqual(info["eligible_rows"], len(rows))
            for side in ("left", "right"):
                result = fit_pivot(samples[side], side)
                np.testing.assert_allclose(result["controller_to_wrist_m"], expected[side], atol=.002)
                self.assertLess(result["pivot_rmse_mm"], 3)

    def test_single_axis_and_low_excitation_are_rejected(self):
        _, samples, _ = capture(mode="single")
        for side in ("left", "right"):
            with self.assertRaisesRegex(CalibrationError, "multi-axis"):
                fit_pivot(samples[side], side)
        _, samples, _ = capture(mode="multi")
        nearly_static = [(p, Rotation.from_rotvec(Rotation.from_matrix(r).as_rotvec() * .03).as_matrix())
                         for p, r in samples["right"]]
        with self.assertRaisesRegex(CalibrationError, "multi-axis"):
            fit_pivot(nearly_static, "right")

    def test_moving_wrist_and_unreasonable_mount_are_rejected(self):
        _, samples, _ = capture(wrist_motion=.04)
        with self.assertRaisesRegex(CalibrationError, "pivot moved"):
            fit_pivot(samples["left"], "left")
        _, samples, _ = capture(offset_scale=5)
        with self.assertRaisesRegex(CalibrationError, "exceeds"):
            fit_pivot(samples["right"], "right")

    def test_mixed_reference_space_and_too_few_frames_are_rejected(self):
        rows, _, _ = capture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "motion.jsonl"
            mixed = copy.deepcopy(rows)
            mixed[80]["input"]["reference"] = 2
            write_rows(path, mixed)
            with self.assertRaisesRegex(CalibrationError, "reference-space reset"):
                load_samples(path)
            write_rows(path, rows[:20])
            with self.assertRaisesRegex(CalibrationError, "need at least"):
                load_samples(path)

    def test_cli_dry_run_then_backup_atomic_config_write_preserves_orientation(self):
        rows, _, expected = capture()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            recording = folder / "motion.jsonl"
            config = folder / "robot.json"
            write_rows(recording, rows)
            original = {"model": "test", "arms": {
                "left": {"controller_to_wrist_m": [0, 0, 0],
                         "controller_to_wrist_xyzw": [0, 0, 0, 1], "other": 42},
                "right": {"controller_to_wrist_m": [0, 0, 0],
                          "controller_to_wrist_xyzw": [0.1, 0.2, 0.3, 0.9], "other": 43}}}
            original_bytes = json.dumps(original, indent=2).encode()
            config.write_bytes(original_bytes)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([str(recording), "--config", str(config)]), 0)
            self.assertFalse(json.loads(output.getvalue())["written"])
            self.assertEqual(config.read_bytes(), original_bytes)
            self.assertFalse(list(folder.glob("*.bak-*")))

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([str(recording), "--config", str(config),
                                       "--side", "right", "--write-config"]), 0)
            report = json.loads(output.getvalue())
            self.assertTrue(report["written"])
            self.assertEqual(Path(report["backup"]).read_bytes(), original_bytes)
            changed = json.loads(config.read_text())
            self.assertEqual(changed["arms"]["left"], original["arms"]["left"])
            self.assertEqual(changed["arms"]["right"]["controller_to_wrist_xyzw"],
                             original["arms"]["right"]["controller_to_wrist_xyzw"])
            self.assertEqual(changed["arms"]["right"]["other"], 43)
            np.testing.assert_allclose(changed["arms"]["right"]["controller_to_wrist_m"],
                                       expected["right"], atol=.002)

    def test_cli_rejects_bad_capture_before_any_config_write(self):
        rows, _, _ = capture(mode="single")
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            recording = folder / "single_axis.jsonl"
            config = folder / "robot.json"
            write_rows(recording, rows)
            original = {"arms": {side: {"controller_to_wrist_m": [0, 0, 0],
                                         "controller_to_wrist_xyzw": [0, 0, 0, 1]}
                                 for side in ("left", "right")}}
            original_bytes = json.dumps(original).encode()
            config.write_bytes(original_bytes)
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                self.assertEqual(main([str(recording), "--config", str(config), "--write-config"]), 2)
            self.assertIn("multi-axis", errors.getvalue())
            self.assertEqual(config.read_bytes(), original_bytes)
            self.assertFalse(list(folder.glob("*.bak-*")))


if __name__ == "__main__":
    unittest.main()
