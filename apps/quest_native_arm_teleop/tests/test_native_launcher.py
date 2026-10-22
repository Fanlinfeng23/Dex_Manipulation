import json
import socket
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import native_launcher as launcher


class NativeLauncherTests(unittest.TestCase):
    def test_vive_live_process_blocks_quest_hardware_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary) / '123456'
            proc.mkdir()
            (proc / 'cmdline').write_bytes(
                b'/usr/bin/python3\0-m\0vive_tianji.teleop\0live\0--side\0both\0')
            self.assertTrue(launcher.vive_live_running(temporary))
            (proc / 'cmdline').write_bytes(
                b'/usr/bin/python3\0-m\0vive_tianji.teleop\0preview\0')
            self.assertFalse(launcher.vive_live_running(temporary))
        with patch.object(launcher, 'vive_live_running', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'Vive 双腕实机遥操'):
                launcher.require_idle(Mock())

    def test_recently_closed_tcp_connection_does_not_block_restart(self):
        with socket.socket() as server, socket.socket() as client:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('127.0.0.1', 0))
            port = server.getsockname()[1]
            server.listen(1)
            client.settimeout(1)
            client.connect(('127.0.0.1', port))
            accepted, _ = server.accept()
            accepted.close()  # The server endpoint performs the active close.
            self.assertEqual(client.recv(1), b'')
            client.close()
            server.close()
        launcher.require_port_free(port)

    def test_running_listener_remains_protected(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('127.0.0.1', 0))
            server.listen(1)
            with self.assertRaises(RuntimeError):
                launcher.require_port_free(server.getsockname()[1])

    def test_joint_motion_profile_is_preserved(self):
        expected = json.loads((launcher.APP / 'configs/hardware_dual_arm.json').read_text())
        expected.update(position_scale=1.0, keyboard_auto_resume_tracking=True,
                        heading_source='fixed', initial_source='quest')
        self.assertEqual(launcher.profile({}), expected)
        self.assertEqual(expected['hardware']['control_sides'], ['left', 'right'])
        self.assertTrue(expected['hardware']['startup_move_enabled'])

    def test_joint_heading_overrides_are_shared(self):
        config = launcher.profile({'TIANJI_JOINT_HEADING_SOURCE': 'head',
                                   'TIANJI_JOINT_OPERATOR_YAW_DEG': '22.5'})
        self.assertEqual(config['heading_source'], 'head')
        self.assertEqual(config['operator_yaw_deg'], 22.5)
        with self.assertRaises(ValueError):
            launcher.profile({'TIANJI_JOINT_OPERATOR_YAW_DEG': 'nan'})

    def test_hardware_without_terminal_stops_before_capture_or_robot(self):
        args = SimpleNamespace(action='start')
        with patch.object(launcher.sys.stdin, 'isatty', return_value=False), \
                patch.object(launcher, 'load_capture') as capture, \
                patch.object(launcher, 'diagnose') as robot:
            with self.assertRaises(RuntimeError):
                launcher.launch(args)
        capture.assert_not_called()
        robot.assert_not_called()

    def test_preview_failure_closes_partial_session_without_robot_diagnosis(self):
        args = SimpleNamespace(action='preview', seconds=1, port=18966, serial=None,
                               no_browser=True, manual=False, robot_ip='192.168.1.190')
        capture = Mock()
        capture.start.return_value = {'serial': 'fake', 'pid': 42, 'start_ticks': 43,
                                      'apk_verification': {'official_apk_unmodified': True}}
        capture.read_state.side_effect = [{}, capture.start.return_value]
        capture.service_alive.return_value = False
        capture.active_recorder_pids.return_value = []
        process = Mock()
        process.poll.return_value = 1
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(launcher, 'WORKSPACE', root), \
                    patch.object(launcher, 'RUNTIME', root / 'runtime'), \
                    patch.object(launcher, 'STATE', root / 'runtime/session.json'), \
                    patch.object(launcher, 'CAPTURE_OWNER', root / 'runtime/capture-owner.json'), \
                    patch.object(launcher, 'load_capture', return_value=capture), \
                    patch.object(launcher, 'require_idle'), \
                    patch.object(launcher, 'require_port_free'), \
                    patch.object(launcher, 'diagnose') as diagnosis, \
                    patch.object(launcher, 'child', return_value=(process, None)) as spawn:
                with self.assertRaises(RuntimeError):
                    launcher.launch(args)
                diagnosis.assert_not_called()
                self.assertNotIn('--hardware', spawn.call_args.args[0])
                capture.stop.assert_called_once_with('fake')
                self.assertFalse(launcher.STATE.exists())
                self.assertFalse(list(root.rglob('session.token')))
                session = json.loads(next(root.glob('logs/*/session.json')).read_text())
                self.assertEqual(session['exit_code'], 2)

    def test_borrowed_capture_service_is_not_stopped(self):
        capture = Mock()
        capture.read_state.return_value = {'pid': 12, 'start_ticks': 14, 'serial': 'borrowed'}
        capture.service_alive.return_value = True
        capture.start.return_value = capture.read_state.return_value
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(launcher, 'CAPTURE_OWNER', Path(temporary) / 'owner.json'):
            with self.assertRaises(RuntimeError):
                launcher.start_capture(capture, None)
            self.assertFalse(launcher.stop_owned_capture(capture))
        capture.start.assert_not_called()
        capture.stop.assert_not_called()

    def test_identity_does_not_accept_wrong_process_start_time(self):
        current = launcher.process_identity(launcher.os.getpid())
        self.assertTrue(launcher.alive(current))
        self.assertFalse(launcher.alive({**current, 'start_ticks': current['start_ticks'] + 1}))

    def test_home_delegate_has_no_timeout_and_keeps_robot_network(self):
        with patch.object(launcher, 'load_capture'), patch.object(launcher, 'require_idle'), \
                patch.object(launcher, 'run') as execute:
            self.assertEqual(launcher.main(['home', '--robot-ip', '192.168.1.191']), 0)
        self.assertEqual(execute.call_args.args[0][-1], 'home')
        self.assertIsNone(execute.call_args.kwargs['timeout'])
        self.assertEqual(execute.call_args.kwargs['env']['TIANJI_ROBOT_IP'], '192.168.1.191')

    def test_refused_home_returns_failure_without_starting_another_action(self):
        with patch.object(launcher, 'load_capture'), patch.object(launcher, 'require_idle'), \
                patch.object(launcher, 'run', side_effect=subprocess.CalledProcessError(2, ['home'])) as execute:
            self.assertEqual(launcher.main(['home']), 2)
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[0][-1], 'home')


if __name__ == '__main__':
    unittest.main()
