"""Offline checks for the Vive + bilateral Wuji process supervisor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import joint_launcher as launcher


def args(**changes):
    values = dict(model_user=None, calibrate_user=None, scale=.5, seconds=60.,
                  expected_baseline_m=.240, left_hand_sn='', right_hand_sn='',
                  left_glove_sn='', right_glove_sn='')
    values.update(changes)
    return argparse.Namespace(**values)


class HandPreparationTests(unittest.TestCase):
    def test_reuses_bilateral_discovery_and_both_models(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            models = folder / 'models'
            models.mkdir()
            for side in launcher.SIDES:
                (models / f'{side}_hand.urdf').write_text('<robot/>')

            commands = []

            def fake_run(command, **_kwargs):
                commands.append([str(part) for part in command])
                if '--check-only' in command:
                    (folder / 'hand-devices.json').write_text(json.dumps({
                        'left': {'hand_sn': 'HL', 'glove_sn': 'GL'},
                        'right': {'hand_sn': 'HR', 'glove_sn': 'GR'},
                    }))
                else:
                    (folder / 'hand-profile.json').write_text(json.dumps({
                        'user_id': 'operator-1', 'is_default': False,
                        'model_paths': {side: str(models / f'{side}_hand.urdf')
                                        for side in launcher.SIDES},
                    }))

            with mock.patch.object(launcher, '_run_checked', side_effect=fake_run):
                serials, user = launcher.prepare_hands(args(model_user='operator-1'), folder)
            self.assertEqual(user, 'operator-1')
            self.assertEqual(serials['--left-hand-sn'], 'HL')
            self.assertEqual(serials['--right-glove-sn'], 'GR')
            self.assertIn('--user', commands[1])
            self.assertNotIn('--calibrate-user', commands[1])

    def test_rejects_incomplete_bilateral_calibration(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            models = folder / 'models'
            models.mkdir()
            (models / 'left_hand.urdf').write_text('<robot/>')

            def fake_run(command, **_kwargs):
                if '--check-only' in command:
                    (folder / 'hand-devices.json').write_text(json.dumps({
                        'left': {'hand_sn': 'HL', 'glove_sn': 'GL'},
                        'right': {'hand_sn': 'HR', 'glove_sn': 'GR'},
                    }))
                else:
                    (folder / 'hand-profile.json').write_text(json.dumps({
                        'user_id': 'operator-1', 'is_default': False,
                        'model_paths': {side: str(models / f'{side}_hand.urdf')
                                        for side in launcher.SIDES},
                    }))

            with mock.patch.object(launcher, '_run_checked', side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, 'right 手模型'):
                    launcher.prepare_hands(args(), folder)


class SessionTests(unittest.TestCase):
    def test_default_scale_matches_official_one_to_one_translation(self):
        with mock.patch.object(launcher, 'run_joint', return_value=0) as run:
            self.assertEqual(launcher.main(['--model-user', 'operator']), 0)
        self.assertEqual(run.call_args.args[0].scale, 1.)

    def test_arm_command_requires_live_both_full_range_and_hand_gate(self):
        command = launcher._arm_command(args(), Path('/tmp/gate'), Path('/tmp/ready'))
        self.assertEqual(command[1:5], ['live', '--side', 'both', '--scale'])
        self.assertIn('--full-range', command)
        self.assertEqual(command[command.index('--joint-gate') + 1], '/tmp/gate')
        self.assertEqual(command[command.index('--joint-ready-file') + 1], '/tmp/ready')

    def test_hand_failure_closes_gate_before_arm_shutdown(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / 'logs').mkdir()
            hand_python = folder / 'python'
            hand_python.touch()
            arm_launcher = folder / 'arm.sh'
            arm_launcher.touch()
            (folder / 'vive_tianji_calibration.json').touch()

            class FakeProcess:
                def __init__(self, pid):
                    self.pid = pid
                    self.returncode = None

                def poll(self):
                    return self.returncode

            hand = FakeProcess(1001)
            arm = FakeProcess(1002)
            stopped = []

            def fake_stop(process, *, label, timeout_s=30.):
                sessions = list((folder / 'logs').glob('vive-joint-*'))
                self.assertEqual(len(sessions), 1)
                self.assertFalse((sessions[0] / 'hand.follow').exists())
                stopped.append(label)
                process.returncode = 0

            with (mock.patch.object(launcher, 'ROOT', folder),
                  mock.patch.object(launcher, 'HAND_PYTHON', hand_python),
                  mock.patch.object(launcher, 'ARM_LAUNCHER', arm_launcher),
                  mock.patch.object(launcher, '_quest_live', return_value=False),
                  mock.patch.object(launcher, '_other_vive_live', return_value=False),
                  mock.patch.object(launcher.sys.stdin, 'isatty', return_value=True),
                  mock.patch.object(launcher, 'prepare_hands', return_value=({'--left-hand-sn': 'HL'}, 'operator')),
                  mock.patch.object(launcher, '_await_hand_ready'),
                  mock.patch.object(launcher, '_worker_ready', side_effect=RuntimeError('hand died')),
                  mock.patch.object(launcher, '_stop_process', side_effect=fake_stop),
                  mock.patch.object(launcher.subprocess, 'Popen', side_effect=[hand, arm])):
                with self.assertRaisesRegex(RuntimeError, 'hand died'):
                    launcher.run_joint(args())
            self.assertEqual(stopped, ['Vive 双臂', 'Wuji 双手'])

    def test_clean_arm_exit_keeps_hands_gated_until_worker_stops(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / 'logs').mkdir()
            hand_python = folder / 'python'
            hand_python.touch()
            arm_launcher = folder / 'arm.sh'
            arm_launcher.touch()
            (folder / 'vive_tianji_calibration.json').touch()
            hand = mock.Mock(pid=1001, returncode=None)
            hand.poll.return_value = None
            arm = mock.Mock(pid=1002, returncode=0)
            arm.poll.return_value = 0
            stopped = []

            def fake_stop(process, *, label, timeout_s=30.):
                session = next((folder / 'logs').glob('vive-joint-*'))
                self.assertFalse((session / 'hand.follow').exists())
                stopped.append(label)

            with (mock.patch.object(launcher, 'ROOT', folder),
                  mock.patch.object(launcher, 'HAND_PYTHON', hand_python),
                  mock.patch.object(launcher, 'ARM_LAUNCHER', arm_launcher),
                  mock.patch.object(launcher, '_quest_live', return_value=False),
                  mock.patch.object(launcher, '_other_vive_live', return_value=False),
                  mock.patch.object(launcher.sys.stdin, 'isatty', return_value=True),
                  mock.patch.object(launcher, 'prepare_hands', return_value=({'--left-hand-sn': 'HL'}, 'operator')),
                  mock.patch.object(launcher, '_await_hand_ready'),
                  mock.patch.object(launcher, '_worker_ready', return_value=True),
                  mock.patch.object(launcher, '_stop_process', side_effect=fake_stop),
                  mock.patch.object(launcher.subprocess, 'Popen', side_effect=[hand, arm])):
                self.assertEqual(launcher.run_joint(args()), 0)
            self.assertEqual(stopped, ['Vive 双臂', 'Wuji 双手'])


if __name__ == '__main__':
    unittest.main()
