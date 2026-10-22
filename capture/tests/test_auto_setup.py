"""Offline safety checks for the optional Tianji Quest startup adapter.

These tests stub ADB and the PC Service recorder. They never contact a Quest,
start the service, or run robot code.
"""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_test as native  # noqa: E402


OFFICIAL_SHA = "a" * 64
CUSTOM_SHA = "b" * 64
CUSTOM_PACKAGE = "com.xrobotoolkit.client.quest.tianji"
CUSTOM_COMPONENT = CUSTOM_PACKAGE + "/com.tianji.xrobotoolkit.AutoSetupActivity"


def custom_profile(**changes):
    profile = {
        "package": CUSTOM_PACKAGE,
        "component": CUSTOM_COMPONENT,
        "sha256": CUSTOM_SHA,
        "upstream_sha256": OFFICIAL_SHA,
        "version": "1.0.1-tianji.1",
    }
    profile.update(changes)
    return profile


def fake_adb_for_apk(remote_sha=CUSTOM_SHA, version="1.0.1-tianji.1"):
    def fake_adb(_serial, *args, **_kwargs):
        if args[:2] == ("shell", "sha256sum"):
            return SimpleNamespace(stdout=f"{remote_sha}  /data/app/base.apk\n", returncode=0)
        if args[:2] == ("shell", "dumpsys"):
            return SimpleNamespace(stdout=f"versionName={version}\n", returncode=0)
        raise AssertionError(f"unexpected ADB command: {args!r}")
    return fake_adb


def tracking_row(clock, *, head=True, left=True, right=True,
                 left_pose="0,0,0,0,0,0,1", right_pose="1,0,0,0,0,0,1"):
    frame = {"predictTime": clock, "Controller": {}}
    if head:
        frame["Head"] = {"status": 3, "pose": "0,0,0,0,0,0,1"}
    if left:
        frame["Controller"]["left"] = {"pose": left_pose}
    if right:
        frame["Controller"]["right"] = {"pose": right_pose}
    return json.dumps({
        "event_type": "PXREADeviceStateJson",
        "payload": json.dumps({"functionName": "Tracking", "value": json.dumps(frame)}),
    })


class AutoSetupTests(unittest.TestCase):
    def test_changed_local_custom_apk_is_rejected_before_adb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apk = root / "custom.apk"
            apk.write_bytes(b"altered-custom-apk")
            profile = custom_profile(apk=str(apk), sha256="0" * 64)
            (root / "tianji-client.json").write_text(json.dumps(profile))
            with patch.object(native, "ROOT", root), patch.dict(
                    "os.environ", {"TIANJI_QUEST_CLIENT": "auto"}), patch.object(
                    native, "adb") as adb:
                with self.assertRaisesRegex(RuntimeError, "校验失败"):
                    native.use_tianji_client()
                adb.assert_not_called()

    def test_changed_upstream_sha_rejects_custom_client(self):
        profile = custom_profile(upstream_sha256="0" * 64)
        with patch.object(native, "CUSTOM_PROFILE", profile), patch.object(
                native, "official_apk_reference", return_value=(OFFICIAL_SHA, None)), patch.object(
                native, "adb", side_effect=fake_adb_for_apk()):
            result = native.verify_installed_apk("serial", "package:/data/app/base.apk\n")
        self.assertFalse(result["approved_client"])
        self.assertFalse(result["official_apk_unmodified"])

    def test_approved_custom_client_is_never_labeled_official(self):
        with patch.object(native, "CUSTOM_PROFILE", custom_profile()), patch.object(
                native, "official_apk_reference", return_value=(OFFICIAL_SHA, None)), patch.object(
                native, "adb", side_effect=fake_adb_for_apk()):
            result = native.verify_installed_apk("serial", "package:/data/app/base.apk\n")
        self.assertTrue(result["approved_client"])
        self.assertEqual(result["client_kind"], "tianji_auto_setup")
        self.assertFalse(result["official_apk_unmodified"])

    def test_custom_mode_cannot_claim_official_even_if_hash_fields_match(self):
        # A semantic guard: package choice, not only a hash comparison, defines
        # whether the active client may be described as the official APK.
        profile = custom_profile(sha256=OFFICIAL_SHA, version="1.0.1")
        with patch.object(native, "CUSTOM_PROFILE", profile), patch.object(
                native, "official_apk_reference", return_value=(OFFICIAL_SHA, None)), patch.object(
                native, "adb", side_effect=fake_adb_for_apk(OFFICIAL_SHA, "1.0.1")):
            result = native.verify_installed_apk("serial", "package:/data/app/base.apk\n")
        self.assertEqual(result["client_kind"], "tianji_auto_setup")
        self.assertFalse(result["official_apk_unmodified"])

    def test_probe_may_retry_but_defaults_are_requested_only_once(self):
        calls = []
        probe = None
        seen = 0

        def fake_adb(_serial, *args, **_kwargs):
            nonlocal probe
            calls.append(args)
            if args[:3] == ("shell", "pidof", CUSTOM_PACKAGE):
                return SimpleNamespace(stdout="1234\n", returncode=0)
            if args[:3] == ("shell", "am", "start"):
                if "tianji_probe" in args:
                    probe = args[-1]
                return SimpleNamespace(stdout="", returncode=0)
            raise AssertionError(f"unexpected ADB command: {args!r}")

        def fake_log(_serial, _pid):
            nonlocal seen
            seen += 1
            if seen < 3:
                return ""
            if seen == 3:
                return "AndroidCall:" + probe
            return "AndroidCall:TianjiAutoConfigured|v1"

        with patch.object(native, "PACKAGE", CUSTOM_PACKAGE), patch.object(
                native, "COMPONENT", CUSTOM_COMPONENT), patch.object(
                native, "adb", side_effect=fake_adb), patch.object(
                native, "unity_log", side_effect=fake_log), patch.object(
                native.time, "sleep", return_value=None):
            result = native.configure_tianji_client("serial", timeout_s=2)

        starts = [call for call in calls if call[:3] == ("shell", "am", "start")]
        probes = [call for call in starts if "tianji_probe" in call]
        defaults = [call for call in starts if "tianji_defaults" in call]
        self.assertEqual(len(probes), 3)
        self.assertEqual(len(defaults), 1)
        self.assertLess(calls.index(probes[-1]), calls.index(defaults[0]))
        self.assertEqual(result["pid"], 1234)
        self.assertTrue(result["defaults_requested"])

    def test_stream_requires_head_both_controllers_and_new_clock(self):
        cases = {
            "valid": [tracking_row(t) for t in (1, 2, 3)],
            "no_head": [tracking_row(t, head=False) for t in (1, 2, 3)],
            "no_left": [tracking_row(t, left=False) for t in (1, 2, 3)],
            "no_right": [tracking_row(t, right=False) for t in (1, 2, 3)],
            "frozen_clock": [tracking_row(1) for _ in range(3)],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            for name, rows in cases.items():
                with self.subTest(name=name):
                    def fake_run(command, **_kwargs):
                        output = Path(command[command.index("--output") + 1])
                        output.write_text("\n".join(rows) + "\n")
                        return SimpleNamespace(returncode=0)

                    with patch.object(native, "ROOT", root), patch.object(
                            native.subprocess, "run", side_effect=fake_run):
                        if name == "valid":
                            result = native.verify_tianji_stream()
                            self.assertTrue(result["head_controller_send_verified"])
                            self.assertGreaterEqual(result["distinct_app_frames"], 3)
                        else:
                            with self.assertRaisesRegex(RuntimeError, "完整 Head / Controller 新帧"):
                                native.verify_tianji_stream()

    def test_stream_rejects_reverse_or_nonfinite_clock_and_empty_pose(self):
        cases = {
            "reverse_clock": [tracking_row(t) for t in (3, 2, 1)],
            "nonfinite_clock": [tracking_row(t) for t in (1, float("nan"), 3)],
            "empty_pose": [tracking_row(t, left_pose="") for t in (1, 2, 3)],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            for name, rows in cases.items():
                with self.subTest(name=name):
                    def fake_run(command, **_kwargs):
                        output = Path(command[command.index("--output") + 1])
                        output.write_text("\n".join(rows) + "\n")
                        return SimpleNamespace(returncode=0)

                    with patch.object(native, "ROOT", root), patch.object(
                            native.subprocess, "run", side_effect=fake_run):
                        with self.assertRaisesRegex(RuntimeError, "完整 Head / Controller 新帧"):
                            native.verify_tianji_stream()


if __name__ == "__main__":
    unittest.main()
