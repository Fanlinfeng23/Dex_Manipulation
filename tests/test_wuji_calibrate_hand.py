"""Offline checks for the single-side Wuji calibration wrapper."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import wuji_calibrate_hand as calibration


class FakeGlove:
    def __init__(self, side: str, result=None, error=None):
        self.side = side
        self.result = result
        self.error = error
        self.calibrate_calls = 0

    def hand_side(self):
        return SimpleNamespace(get=lambda: self.side)

    async def calibrate(self, *, skip_constraints, timeout_s, on_feedback):
        self.calibrate_calls += 1
        if skip_constraints or timeout_s != 900.0:
            raise AssertionError("calibration safety settings changed")
        on_feedback({
            "step_index": 0, "step_total": 6,
            "step_name": "pinch_index", "state": "waiting_stable",
        })
        if self.error is not None:
            raise self.error
        return self.result


class FakeManager:
    def __init__(self, glove, *, user_id="u_alice", is_default=False,
                 expected_sn="GLOVE_LEFT", expected_side="left"):
        self.glove = glove
        self.user = {"user_id": user_id, "display_name": "alice", "is_default": is_default}
        self.expected_sn = expected_sn
        self.expected_side = expected_side
        self.connected = False
        self.disconnected = False

    def current_user(self):
        return self.user

    def connect(self, *, sn, device_name):
        self.connected = True
        if sn != self.expected_sn or device_name != f"calibration_{self.expected_side}":
            raise AssertionError("wrong glove selection")
        return self.glove

    def disconnect_all(self):
        self.disconnected = True


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.root = Path(home.name)
        home_patch = patch.object(Path, "home", return_value=self.root)
        home_patch.start()
        self.addCleanup(home_patch.stop)
        self.model = calibration.model_path("u_alice", "left")
        self.model.parent.mkdir(parents=True)
        self.model.write_text('<robot name="left"/>', encoding="utf-8")
        self.output = self.root / "result.json"

    def result(self):
        return {
            "handedness": "left",
            "calibrated_urdf": str(self.model),
            "sdk_user": {"user_id": "u_alice", "display_name": "alice", "is_default": False},
            "poses_collected": 6,
            "frames_per_pose": {"pinch_index": 42},
        }

    def run_wrapper(self, manager, *, sn="GLOVE_LEFT", side="left"):
        with (
            patch.object(calibration, "SdkManager", SimpleNamespace(instance=lambda: manager)),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            status = calibration.main([
                "--sn", sn, "--side", side,
                "--user-id", "u_alice", "--output", str(self.output),
            ])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_success_writes_checked_sdk_result_and_disconnects(self):
        glove = FakeGlove("left", self.result())
        manager = FakeManager(glove)
        status, stdout, stderr = self.run_wrapper(manager)
        self.assertEqual(status, 0, stderr)
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), self.result())
        self.assertIn("当前标定：左手", stdout)
        self.assertEqual(glove.calibrate_calls, 1)
        self.assertTrue(manager.disconnected)
        self.assertEqual(manager.user["user_id"], "u_alice")
        self.assertFalse(list(self.output.parent.glob(".result.json.*.tmp")))

    def test_right_side_uses_right_model_and_result(self):
        right_model = calibration.model_path("u_alice", "right")
        right_model.write_text('<robot name="right"/>', encoding="utf-8")
        result = self.result()
        result.update(handedness="right", calibrated_urdf=str(right_model))
        manager = FakeManager(
            FakeGlove("right", result),
            expected_sn="GLOVE_RIGHT", expected_side="right",
        )
        status, stdout, stderr = self.run_wrapper(
            manager, sn="GLOVE_RIGHT", side="right",
        )
        self.assertEqual(status, 0, stderr)
        self.assertIn("当前标定：右手", stdout)
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), result)

    def test_wrong_or_default_user_fails_before_connect(self):
        for kwargs in ({"user_id": "other"}, {"is_default": True}):
            with self.subTest(kwargs=kwargs):
                manager = FakeManager(FakeGlove("left", self.result()), **kwargs)
                status, _, _ = self.run_wrapper(manager)
                self.assertEqual(status, 1)
                self.assertFalse(manager.connected)
                self.assertTrue(manager.disconnected)
                self.assertFalse(self.output.exists())

    def test_wrong_physical_glove_side_fails_before_calibration(self):
        glove = FakeGlove("right", self.result())
        manager = FakeManager(glove)
        status, _, stderr = self.run_wrapper(manager)
        self.assertEqual(status, 1)
        self.assertIn("本次需要 'left'", stderr)
        self.assertEqual(glove.calibrate_calls, 0)
        self.assertTrue(manager.disconnected)
        self.assertFalse(self.output.exists())

    def test_mismatched_sdk_success_summary_never_writes_result(self):
        variants = (
            ("handedness", "right"),
            ("calibrated_urdf", "/tmp/other.urdf"),
            ("poses_collected", 0),
        )
        for key, value in variants:
            with self.subTest(key=key):
                result = self.result()
                result[key] = value
                manager = FakeManager(FakeGlove("left", result))
                status, _, _ = self.run_wrapper(manager)
                self.assertEqual(status, 1)
                self.assertFalse(self.output.exists())
                self.assertTrue(manager.disconnected)

        result = self.result()
        result["sdk_user"]["user_id"] = "other"
        manager = FakeManager(FakeGlove("left", result))
        status, _, _ = self.run_wrapper(manager)
        self.assertEqual(status, 1)
        self.assertFalse(self.output.exists())
        self.assertTrue(manager.disconnected)

    def test_missing_saved_model_is_rejected(self):
        self.model.unlink()
        manager = FakeManager(FakeGlove("left", self.result()))
        status, _, _ = self.run_wrapper(manager)
        self.assertEqual(status, 1)
        self.assertFalse(self.output.exists())

    def test_sdk_timeout_is_nonzero_and_leaves_no_result(self):
        glove = FakeGlove("left", error=TimeoutError("pose timed out"))
        manager = FakeManager(glove)
        status, _, stderr = self.run_wrapper(manager)
        self.assertEqual(status, 1)
        self.assertIn("pose timed out", stderr)
        self.assertTrue(manager.disconnected)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
