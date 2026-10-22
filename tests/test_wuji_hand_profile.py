"""Offline safety checks for the Wuji model selection and glove stream."""

from __future__ import annotations

import contextlib
import io
import os
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import wuji_hand_profile as profile
from wuji_joint_hand import glove_frame_stale, latest_keypoints, limited_resume_target


class FakeManager:
    def __init__(self):
        self.users = [
            {"user_id": "", "display_name": "Default", "is_default": True},
            {"user_id": "u_alice", "display_name": "alice", "is_default": False},
        ]
        self.current = self.users[0]

    def list_users(self):
        return list(self.users)

    def current_user(self):
        return self.current

    def switch_user(self, user_id):
        self.current = next(user for user in self.users if user["user_id"] == user_id)
        return self.current

    def switch_to_default_user(self):
        self.current = self.users[0]
        return self.current

    def create_user(self, name):
        user = {"user_id": f"u_{name}", "display_name": name, "is_default": False}
        self.users.append(user)
        return user


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        self.home_patch = patch.object(Path, "home", return_value=Path(self.home.name))
        self.home_patch.start()
        self.addCleanup(self.home_patch.stop)
        self.manager = FakeManager()

    def write_model(self, user, side="right"):
        path = profile.user_model_path(user, side)
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix(".staged")
        staged.write_text(f'<robot name="{side}_hand"/>', encoding="utf-8")
        staged.replace(path)
        return path

    def calibration_success(self, command, check=False):
        side = command[command.index("--side") + 1]
        user = profile.find_user(self.manager, command[command.index("--user-id") + 1])
        model = self.write_model(user, side)
        Path(command[command.index("--output") + 1]).write_text(json.dumps({
            "handedness": side, "sdk_user": user, "calibrated_urdf": str(model),
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    def test_existing_profile_requires_valid_models_on_both_sides(self):
        user = self.manager.users[1]
        with self.assertRaisesRegex(ValueError, "没有左手标定模型"):
            profile.select_user(self.manager, user)
        self.assertEqual(self.manager.current_user()["user_id"], "")
        path = self.write_model(user)
        self.assertEqual(profile.checked_model_path(user), path.resolve())
        with self.assertRaisesRegex(ValueError, "没有左手标定模型"):
            profile.select_user(self.manager, user)
        self.write_model(user, "left")
        profile.select_user(self.manager, user)
        self.assertEqual(self.manager.current_user()["user_id"], "u_alice")
        path.write_text("<not_a_robot/>", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "模型不可用"):
            profile.checked_model_path(user)

    def test_failed_calibration_restores_previous_user(self):
        with patch.object(profile.subprocess, "run", return_value=SimpleNamespace(returncode=1)):
            with self.assertRaisesRegex(RuntimeError, "标定未完成"):
                profile.calibrate_user(self.manager, "alice", "WG_LEFT", "WG_RIGHT")
        self.assertEqual(self.manager.current_user()["user_id"], "")
        self.assertTrue(profile.calibration_pending_path(self.manager.users[1]).exists())

    def test_success_exit_without_calibration_result_is_rejected(self):
        with patch.object(profile.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
            with self.assertRaisesRegex(RuntimeError, "没有返回成功结果"):
                profile.calibrate_user(self.manager, "alice", "WG_LEFT", "WG_RIGHT")
        self.assertEqual(self.manager.current_user()["user_id"], "")

    def test_recalibration_must_update_model_of_expected_side(self):
        model = self.write_model(self.manager.users[1], "left")
        def unchanged_model(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_text(json.dumps({
                "handedness": "left", "sdk_user": self.manager.users[1],
                "calibrated_urdf": str(model),
            }), encoding="utf-8")
            return SimpleNamespace(returncode=0)
        with patch.object(profile.subprocess, "run", side_effect=unchanged_model):
            with self.assertRaisesRegex(RuntimeError, "未更新左手模型"):
                profile.calibrate_user(self.manager, "alice", "WG_LEFT", "WG_RIGHT")
        self.assertEqual(self.manager.current_user()["user_id"], "")

    def test_both_calibrations_finish_before_selection_is_usable(self):
        with patch.object(profile.subprocess, "run", side_effect=self.calibration_success) as run:
            user = profile.calibrate_user(self.manager, "alice", "WG_LEFT", "WG_RIGHT")
        self.assertEqual([c.args[0][c.args[0].index("--sn") + 1] for c in run.call_args_list],
                         ["WG_LEFT", "WG_RIGHT"])
        self.assertEqual(set(profile.checked_model_paths(user)), {"left", "right"})
        self.assertFalse(profile.calibration_pending_path(user).exists())
        self.assertEqual(self.manager.current_user()["user_id"], "")
        output = Path(self.home.name) / "selection.json"
        profile.write_selection(output, user)
        self.assertEqual(set(json.loads(output.read_text())["model_paths"]), {"left", "right"})

    def test_partial_recalibration_cannot_use_old_right_model_to_start(self):
        user = self.manager.users[1]
        for side in ("left", "right"):
            self.write_model(user, side)
        def fail_right(command, **kwargs):
            if command[command.index("--side") + 1] == "right":
                return SimpleNamespace(returncode=1)
            return self.calibration_success(command, **kwargs)
        with patch.object(profile.subprocess, "run", side_effect=fail_right):
            with self.assertRaisesRegex(RuntimeError, "右手手型标定未完成"):
                profile.calibrate_user(self.manager, "alice", "WG_LEFT", "WG_RIGHT")
        with self.assertRaisesRegex(ValueError, "上次双手标定未完成"):
            profile.checked_model_paths(user)

    def test_c_is_entered_at_the_model_menu_then_operator_name(self):
        self.write_model(self.manager.users[1])
        with patch("builtins.input", side_effect=["c", "alice"]) as read, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(profile.interactive_choice(self.manager), ("calibrate", "alice"))
        self.assertEqual(read.call_args_list[0].args[0], "选择编号 / c：")
        self.assertIn("输入要标定的用户名称", read.call_args_list[1].args[0])
        self.assertNotIn("[1]", output.getvalue())
        self.assertIn("没有左手标定模型", output.getvalue())

    def test_default_cannot_bypass_bilateral_calibration(self):
        user = profile.find_user(self.manager, "Default")
        with self.assertRaisesRegex(ValueError, "不能使用 Default"):
            profile.select_user(self.manager, user)


class GloveStreamTests(unittest.TestCase):
    def test_stale_stream_never_reuses_old_frame_indefinitely(self):
        self.assertTrue(glove_frame_stale(None, 10.0))
        self.assertFalse(glove_frame_stale(10.0, 10.5))
        self.assertTrue(glove_frame_stale(10.0, 10.7))

    def test_invalid_skeleton_frame_is_rejected(self):
        joint = SimpleNamespace(pose=SimpleNamespace(position=[0.0, 0.0, 0.0]))
        frame = SimpleNamespace(joints=[joint] * 20)
        samples = iter([frame, None])
        subscription = SimpleNamespace(recv=lambda: next(samples))
        with self.assertRaisesRegex(RuntimeError, "骨架帧无效"):
            latest_keypoints(subscription)

    def test_valid_frame_carries_sequence_for_freshness_check(self):
        joint = SimpleNamespace(pose=SimpleNamespace(position=[0.0, 0.0, 0.0]))
        frame = SimpleNamespace(
            joints=[joint] * 21,
            header=SimpleNamespace(seq=22, timestamp_us=1_000_000),
        )
        samples = iter([frame, None])
        subscription = SimpleNamespace(recv=lambda: next(samples))
        keypoints, identity = latest_keypoints(subscription)
        self.assertEqual(keypoints.shape, (21, 3))
        self.assertEqual(identity, (22, 1_000_000))

    def test_reconnect_limits_each_joint_step_until_caught_up(self):
        previous = [0.0] * 20
        target = [1.01] * 20
        for _ in range(40):
            command, caught_up = limited_resume_target(target, previous)
            self.assertLessEqual(max(abs(a - b) for a, b in zip(command, previous)),
                                 0.025000001)
            previous = command
        self.assertFalse(caught_up)
        command, caught_up = limited_resume_target(target, previous)
        self.assertTrue(caught_up)
        self.assertAlmostEqual(command[0], 1.01)

class JointStartupTests(unittest.TestCase):
    """The public shell entries preserve arguments when routing to native launcher."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        dex = Path(__file__).resolve().parents[1]
        for script in ("tianji_quest.sh", "joint_teleop.sh"):
            shutil.copy2(dex / script, self.root / script)
        self.native_args = self.root / "native-args.json"
        native = self.root / "tianji_native.sh"
        native.write_text(
            '#!/usr/bin/env python3\n'
            'import json, os, sys\n'
            'from pathlib import Path\n'
            'Path(os.environ["TEST_NATIVE_ARGS"]).write_text('
            'json.dumps(sys.argv[1:], ensure_ascii=False), encoding="utf-8")\n',
            encoding="utf-8",
        )
        native.chmod(0o755)
        self.env = os.environ.copy()
        self.env["TEST_NATIVE_ARGS"] = str(self.native_args)

    def route(self, script, *args):
        self.native_args.unlink(missing_ok=True)
        result = subprocess.run(
            [str(self.root / script), *args], capture_output=True, text=True,
            env=self.env, timeout=5, check=False,
        )
        forwarded = (json.loads(self.native_args.read_text(encoding="utf-8"))
                     if self.native_args.exists() else None)
        return result, forwarded

    def test_joint_preserves_operator_name_with_spaces_and_all_serials(self):
        arguments = [
            "--model-user", "Operator Alice", "--left-hand-sn", "HL",
            "--right-hand-sn", "HR", "--left-glove-sn", "GL",
            "--right-glove-sn", "GR", "--manual",
        ]
        result, forwarded = self.route("tianji_quest.sh", "joint", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(forwarded, ["joint", *arguments])

    def test_joint_teleop_uses_the_same_native_joint_route(self):
        arguments = ["--calibrate-user", "New Operator", "--left-glove-sn", "GL",
                     "--right-glove-sn", "GR"]
        result, forwarded = self.route("joint_teleop.sh", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(forwarded, ["joint", *arguments])

    def test_check_always_requests_read_only_hand_probe(self):
        result, forwarded = self.route("tianji_quest.sh", "check", "--robot-ip", "192.0.2.1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(forwarded, ["check", "--check-hands", "--robot-ip", "192.0.2.1"])

    def test_other_commands_and_aliases_route_to_native_lifecycle(self):
        expected = {"home": "home", "stop": "stop", "capture": "capture",
                    "usb": "capture", "dual-arm": "start", "standby": "standby",
                    "recover": "recover", "tracking": "tracking", "status": "status"}
        for command, native_command in expected.items():
            with self.subTest(command=command):
                result, forwarded = self.route("tianji_quest.sh", command, "--probe")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(forwarded, [native_command, "--probe"])

    def test_unknown_command_does_not_call_native_launcher(self):
        result, forwarded = self.route("tianji_quest.sh", "unknown")
        self.assertEqual(result.returncode, 2)
        self.assertIsNone(forwarded)


if __name__ == "__main__":
    unittest.main()
