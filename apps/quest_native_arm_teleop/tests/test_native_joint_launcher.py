"""Offline lifecycle checks for the XRoboToolkit + bilateral Wuji launcher."""
from __future__ import annotations

import json
import signal
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import native_launcher as launcher


SERIALS = {
    '--left-hand-sn': 'LH', '--left-glove-sn': 'LG',
    '--right-hand-sn': 'RH', '--right-glove-sn': 'RG',
}


def joint_args():
    return SimpleNamespace(action='joint', seconds=None, port=18766, serial=None,
                           robot_ip='192.168.1.190', interface='fake0',
                           model_user='calibrated-user', calibrate_user=None,
                           no_browser=True, manual=False, keep_capture=False,
                           left_hand_sn='', left_glove_sn='',
                           right_hand_sn='', right_glove_sn='')


class FakeProcess:
    def __init__(self, kind):
        self.kind = kind
        self.returncode = 0 if kind == 'keyboard' else None

    def poll(self):
        return self.returncode


class NativeJointLauncherTests(unittest.TestCase):
    def test_preflight_needs_four_devices_and_both_model_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / 'models'
            models.mkdir()
            left = models / 'left_hand.urdf'
            right = models / 'right_hand.urdf'
            left.write_text('<robot name="left"/>')
            right.write_text('<robot name="right"/>')
            cases = (
                ({'right': {'hand_sn': 'RH'}}, None, 'right glove'),
                (None, {'left': str(left)}, '模型'),
            )
            for index, (device_override, model_override, expected) in enumerate(cases):
                with self.subTest(expected=expected):
                    log_dir = root / f'case-{index}'
                    log_dir.mkdir()
                    commands = []

                    def fake_run(command, **_):
                        commands.append(command)
                        if '--check-only' in command:
                            devices = {'left': {'hand_sn': 'LH', 'glove_sn': 'LG'},
                                       'right': {'hand_sn': 'RH', 'glove_sn': 'RG'}}
                            if device_override:
                                devices.update(device_override)
                            Path(command[command.index('--device-output') + 1]).write_text(json.dumps(devices))
                        else:
                            selection = {'user_id': 'calibrated-user', 'is_default': False,
                                         'model_paths': model_override or
                                                        {'left': str(left), 'right': str(right)}}
                            Path(command[command.index('--output') + 1]).write_text(json.dumps(selection))

                    with patch.object(launcher, 'run', side_effect=fake_run), \
                         patch.dict(launcher.os.environ, {'WUJI_DEFAULT_MODEL': '0'}):
                        with self.assertRaisesRegex(RuntimeError, expected):
                            launcher.prepare_hands(joint_args(), log_dir)
                    self.assertIn('--check-only', commands[0])
                    if device_override:
                        self.assertEqual(len(commands), 1)  # No model selector after failed scan.

    def mocked_launch(self, *, failure=None, action='joint'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = []
            commands = {}
            gate = None
            ready = None
            keyboard_started = False
            args = joint_args()
            args.action = action
            capture = Mock()
            capture.service_env.return_value = {}

            def prepare(_args, _directory):
                events.append('preflight')
                if failure == 'preflight':
                    raise RuntimeError('四设备或双模型校验失败')
                return {'serials': dict(SERIALS), 'user_id': 'calibrated-user'}

            def diagnose(_args, _config, _directory):
                events.append('diagnose')

            def confirm(_prompt):
                events.append('confirm')
                return 'JOINT-MOVE' if action == 'joint' else 'NATIVE-MOVE'

            def capture_start(_capture, _serial):
                events.append('capture')
                return {'serial': 'fake-quest',
                        'apk_verification': {'official_apk_unmodified': True}}

            def spawn(command, _log, **_kwargs):
                nonlocal gate, ready, keyboard_started
                words = list(map(str, command))
                if 'wuji_joint_hand.py' in ' '.join(words):
                    kind = 'hand'
                    gate = Path(words[words.index('--gate') + 1])
                    ready = Path(words[words.index('--ready-file') + 1])
                    if failure != 'hand_start':
                        ready.write_text('ready\n')
                elif 'quest_teleop.native_server' in words:
                    kind = 'server'
                elif 'quest_teleop.native_bridge' in words:
                    kind = 'bridge'
                elif 'quest_teleop.keyboard' in words:
                    kind = 'keyboard'
                    keyboard_started = True
                    if gate is not None:
                        gate.touch()  # Simulate an already active hand-follow gate.
                else:
                    self.fail(f'unexpected child: {words}')
                events.append(kind)
                commands[kind] = words
                return FakeProcess(kind), {'pid': 100 + len(commands),
                                           'start_ticks': 1, 'exe': 'fake'}

            original_ready = launcher.await_hand_ready

            def await_ready(process, ready_file):
                if failure == 'hand_start':
                    raise RuntimeError('左右 Glove 未就绪')
                original_ready(process, ready_file, timeout=.2)
                events.append('hand_ready')

            original_hand_alive = launcher.hand_alive

            def check_hand(process, ready_file=None):
                if failure == 'hand_runtime' and keyboard_started:
                    raise RuntimeError('左右手通道已退出')
                original_hand_alive(process, ready_file)

            def finish(process, name, *_args, **_kwargs):
                if process is not None:
                    self.assertFalse(gate.exists() if gate else False)
                    events.append('finish:' + process.kind)

            ready_health = {'ok': True, 'hardware': True, 'input_backend': 'xrobotoolkit',
                            'source': 'quest', 'hardware_fault': None,
                            'hardware_connected': True, 'motion_scope': 'full',
                            'control_sides': ['left', 'right']}
            patches = (
                patch.object(launcher, 'WORKSPACE', root),
                patch.object(launcher, 'RUNTIME', root / 'runtime'),
                patch.object(launcher, 'STATE', root / 'runtime/session.json'),
                patch.object(launcher, 'CAPTURE_OWNER', root / 'runtime/capture-owner.json'),
                patch.object(launcher, 'load_capture', return_value=capture),
                patch.object(launcher, 'require_idle'),
                patch.object(launcher, 'require_port_free'),
                patch.object(launcher, 'prepare_hands', side_effect=prepare),
                patch.object(launcher, 'diagnose', side_effect=diagnose),
                patch('builtins.input', side_effect=confirm),
                patch.object(launcher, 'start_capture', side_effect=capture_start),
                patch.object(launcher, 'child', side_effect=spawn),
                patch.object(launcher, 'await_hand_ready', side_effect=await_ready),
                patch.object(launcher, 'hand_alive', side_effect=check_hand),
                patch.object(launcher, 'finish_child', side_effect=finish),
                patch.object(launcher, 'health', return_value=ready_health),
                patch.object(launcher, 'stop_owned_capture', return_value=True),
                patch.object(launcher.sys.stdin, 'isatty', return_value=True),
            )
            with ExitStack() as stack:
                for change in patches:
                    stack.enter_context(change)
                try:
                    result, error = launcher.launch(args), None
                except RuntimeError as exc:
                    result, error = None, exc
                state_exists = launcher.STATE.exists()
                session_files = list(root.glob('logs/*/session.json'))
                session = json.loads(session_files[0].read_text()) if session_files else None
                gate_exists = gate.exists() if gate else False
                ready_exists = ready.exists() if ready else False
            return SimpleNamespace(result=result, error=error, events=events, commands=commands,
                                   gate_exists=gate_exists, ready_exists=ready_exists,
                                   state_exists=state_exists, session=session)

    def test_joint_start_order_keyboard_arguments_and_cleanup(self):
        observed = self.mocked_launch()
        self.assertIsNone(observed.error)
        self.assertEqual(observed.result, 0)
        sequence = observed.events
        self.assertLess(sequence.index('preflight'), sequence.index('diagnose'))
        self.assertLess(sequence.index('diagnose'), sequence.index('confirm'))
        self.assertLess(sequence.index('confirm'), sequence.index('capture'))
        self.assertLess(sequence.index('capture'), sequence.index('hand'))
        self.assertLess(sequence.index('hand_ready'), sequence.index('server'))
        self.assertLess(sequence.index('server'), sequence.index('bridge'))
        self.assertLess(sequence.index('bridge'), sequence.index('keyboard'))
        self.assertEqual(observed.commands['hand'][observed.commands['hand'].index('--gate-timeout') + 1],
                         '0.5')
        for option, serial in SERIALS.items():
            self.assertEqual(observed.commands['hand'][observed.commands['hand'].index(option) + 1],
                             serial)
        keyboard = observed.commands['keyboard']
        for option in ('--native-arm-mode', '--joint-gate', '--token-file', '--hardware',
                       '--full-arm', '--auto-calibrate'):
            self.assertIn(option, keyboard)
        self.assertNotIn('--auto-follow-on-ready', keyboard)
        hand = observed.commands['hand']
        bridge = observed.commands['bridge']
        server = observed.commands['server']
        self.assertEqual(keyboard[keyboard.index('--joint-gate') + 1],
                         hand[hand.index('--gate') + 1])
        token = keyboard[keyboard.index('--token-file') + 1]
        self.assertEqual(token, bridge[bridge.index('--token-file') + 1])
        self.assertEqual(token, server[server.index('--token-file') + 1])
        self.assertFalse(observed.gate_exists)
        self.assertFalse(observed.ready_exists)
        self.assertFalse(observed.state_exists)
        self.assertEqual(observed.session['exit_code'], 0)

    def test_old_auto_start_environment_cannot_enable_hardware_auto_follow(self):
        with patch.dict(launcher.os.environ,
                        {'TIANJI_NATIVE_AUTO_START': '1', 'WUJI_JOINT_AUTO_START': '1'}):
            for action in ('joint', 'start'):
                with self.subTest(action=action):
                    observed = self.mocked_launch(action=action)
                    self.assertIsNone(observed.error)
                    self.assertEqual(observed.result, 0)
                    keyboard = observed.commands['keyboard']
                    self.assertIn('--auto-calibrate', keyboard)
                    self.assertNotIn('--auto-follow-on-ready', keyboard)

    def test_preflight_or_hand_start_failure_never_starts_arm(self):
        for failure in ('preflight', 'hand_start'):
            with self.subTest(failure=failure):
                observed = self.mocked_launch(failure=failure)
                self.assertIsInstance(observed.error, RuntimeError)
                self.assertNotIn('server', observed.commands)
                self.assertNotIn('bridge', observed.commands)
                self.assertNotIn('keyboard', observed.commands)
                self.assertFalse(observed.gate_exists)
                self.assertFalse(observed.state_exists)

    def test_hand_runtime_failure_closes_gate_then_stops_arm(self):
        observed = self.mocked_launch(failure='hand_runtime')
        self.assertIsInstance(observed.error, RuntimeError)
        self.assertIn('server', observed.commands)
        self.assertIn('keyboard', observed.commands)
        self.assertFalse(observed.gate_exists)
        self.assertFalse(observed.state_exists)
        self.assertLess(observed.events.index('finish:keyboard'),
                        observed.events.index('finish:server'))
        self.assertLess(observed.events.index('finish:server'),
                        observed.events.index('finish:hand'))

    def test_stop_recovers_owned_orphan_hand_but_ignores_reused_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / 'hand.follow'
            gate.touch()
            hand = {'pid': 1201, 'start_ticks': 22, 'exe': '/usr/bin/python3'}
            stale_server = {'pid': 1202, 'start_ticks': 10, 'exe': '/usr/bin/python3'}
            current_server = {**stale_server, 'start_ticks': 99}
            state_file = root / 'session.json'
            state_file.write_text(json.dumps({'hand': hand, 'server': stale_server,
                                              'hand_gate': str(gate)}))
            alive_hand = True
            killed = []

            def identity(pid):
                if pid == hand['pid']:
                    return hand if alive_hand else None
                if pid == stale_server['pid']:
                    return current_server
                return None

            def kill(pid, sig):
                nonlocal alive_hand
                self.assertFalse(gate.exists())
                killed.append((pid, sig))
                alive_hand = False

            with patch.object(launcher, 'STATE', state_file), \
                 patch.object(launcher, 'process_identity', side_effect=identity), \
                 patch.object(launcher.os, 'kill', side_effect=kill), \
                 patch.object(launcher, 'load_capture'), \
                 patch.object(launcher, 'stop_owned_capture', return_value=False):
                launcher.stop()
            self.assertEqual(killed, [(hand['pid'], signal.SIGTERM)])
            self.assertFalse(state_file.exists())
            self.assertFalse(gate.exists())


if __name__ == '__main__':
    unittest.main()
