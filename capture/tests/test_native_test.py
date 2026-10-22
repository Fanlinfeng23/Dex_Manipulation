"""Offline checks for the isolated Quest capture launcher."""

from contextlib import redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_test as native  # noqa: E402


class NativeTest(unittest.TestCase):
    def test_device_reports_usb_permissions_and_headset_authorization_separately(self):
        for status, expected in [('no permissions', 'USB 权限'),
                                 ('unauthorized', '头显中允许')]:
            with self.subTest(status=status), patch.object(
                    native, 'run', return_value=SimpleNamespace(
                        stdout=f'List of devices attached\nQUEST123\t{status}\n')):
                with self.assertRaisesRegex(RuntimeError, expected):
                    native.device()
                with self.assertRaisesRegex(RuntimeError, expected):
                    native.device('QUEST123')

    def test_device_requires_serial_only_for_multiple_ready_devices(self):
        with patch.object(native, 'run', return_value=SimpleNamespace(
                stdout='List of devices attached\nQUEST1\tdevice\nQUEST2\tdevice\n')):
            with self.assertRaisesRegex(RuntimeError, '--serial'):
                native.device()
            self.assertEqual(native.device('QUEST2'), 'QUEST2')

    def test_existing_service_state_is_migrated_with_executable_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'service-state.json'
            path.write_text(json.dumps({'pid': os.getpid(), 'serial': 'test'}))
            with patch.object(native, 'STATE', path), patch.object(
                    native, 'EXE', Path(sys.executable).resolve()):
                state = native.read_state()
                self.assertEqual(state['start_ticks'], native.proc_stat(os.getpid())[1])
                self.assertEqual(json.loads(path.read_text())['start_ticks'], state['start_ticks'])
                self.assertTrue(native.service_alive(state))
                state['start_ticks'] += 1
                self.assertFalse(native.service_alive(state))

    def test_record_preflight_runs_before_device_or_service_start(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                native, 'RECORDER', Path(directory) / 'missing.py'), patch.object(
                native, 'start') as start, patch.object(sys, 'argv',
                ['native_test.py', 'record', '--output', str(Path(directory) / 'out.jsonl')]):
            with redirect_stderr(io.StringIO()):
                self.assertEqual(native.main(), 2)
            start.assert_not_called()

    def test_record_preflight_respects_actual_summary_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = root / 'raw_recorder.py'
            sdk = root / 'libPXREARobotSDK.so'
            recorder.touch()
            sdk.touch()
            output = root / 'out.jsonl'
            (root / 'out.jsonl.summary.json').touch()
            args = SimpleNamespace(output=output, seconds=60.0, wait_seconds=15.0)
            with patch.object(native, 'RECORDER', recorder), patch.object(native, 'SDK', sdk):
                with self.assertRaisesRegex(RuntimeError, '输出已存在'):
                    native.prepare_record(args)

    def test_installed_apk_truth_is_based_on_remote_hash_and_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apk = root / 'XRoboToolkit-Quest-1.0.1.apk'
            apk.write_bytes(b'offline-test-apk')
            digest = hashlib.sha256(apk.read_bytes()).hexdigest()
            assets = root / 'official-assets.json'
            assets.write_text(json.dumps({'assets': [{
                'file': str(apk),
                'url': 'https://github.com/XR-Robotics/XRoboToolkit-Unity-Client-Quest/'
                       'releases/download/v1.0.1/XRoboToolkit-Quest-1.0.1.apk',
                'sha256': digest,
            }]}))

            def fake_adb(_serial, *args, **_kwargs):
                if args[1] == 'sha256sum':
                    return SimpleNamespace(stdout=f'{digest}  /data/app/base.apk\n', returncode=0)
                return SimpleNamespace(stdout='versionName=1.0.1\n', returncode=0)

            with patch.object(native, 'ASSETS', assets), patch.object(native, 'adb', fake_adb):
                verified = native.verify_installed_apk('test', 'package:/data/app/base.apk\n')
                self.assertTrue(verified['official_apk_unmodified'])
                self.assertEqual(verified['installed_apk_path'], '/data/app/base.apk')

            def wrong_hash(_serial, *args, **_kwargs):
                if args[1] == 'sha256sum':
                    return SimpleNamespace(stdout=f'{"0" * 64}  /data/app/base.apk\n', returncode=0)
                return SimpleNamespace(stdout='versionName=1.0.1\n', returncode=0)

            with patch.object(native, 'ASSETS', assets), patch.object(native, 'adb', wrong_hash):
                self.assertFalse(native.verify_installed_apk(
                    'test', 'package:/data/app/base.apk\n')['official_apk_unmodified'])

    def test_serial_mismatch_does_not_stop_anything(self):
        with patch.object(native, 'read_state', return_value={'serial': 'expected'}), patch.object(
                native, 'read_record_state', return_value={}), patch.object(
                native, 'stop_collector') as stop_collector, patch.object(native, 'adb') as adb:
            with self.assertRaisesRegex(RuntimeError, '不符'):
                native.stop('different')
            stop_collector.assert_not_called()
            adb.assert_not_called()

    def test_stop_finishes_recorder_before_stopping_service(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            state_path = Path(directory) / 'service-state.json'
            record_path = Path(directory) / 'record-state.json'
            state_path.write_text('{}')
            state = {'pid': 123, 'start_ticks': 10, 'serial': 'test'}
            recording = {'collector_pid': 456, 'collector_start_ticks': 20,
                         'serial': 'test'}

            def fake_adb(_serial, *_args, **_kwargs):
                events.append('adb')
                return SimpleNamespace(stdout='', returncode=0)

            def fake_kill(pid, _signal):
                events.append(f'kill:{pid}')

            with patch.object(native, 'STATE', state_path), patch.object(
                    native, 'RECORD_STATE', record_path), patch.object(
                    native, 'read_state', return_value=state), patch.object(
                    native, 'read_record_state', return_value=recording), patch.object(
                    native, 'stop_collector', side_effect=lambda _state: events.append('collector') or True), patch.object(
                    native, 'collector_alive', return_value=False), patch.object(
                    native, 'logcat_alive', return_value=False), patch.object(
                    native, 'service_alive', side_effect=[True, False]), patch.object(
                    native, 'wait_until_gone', return_value=True), patch.object(
                    native, 'adb', side_effect=fake_adb), patch.object(
                    native.os, 'kill', side_effect=fake_kill):
                native.stop('test')
            self.assertEqual(events[0], 'collector')
            self.assertIn('kill:123', events)
            self.assertLess(events.index('collector'), events.index('kill:123'))

    def test_listener_check_does_not_connect(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(('127.0.0.1', 0))
            server.listen(1)
            self.assertTrue(native.listener(server.getsockname()[1]))


if __name__ == '__main__':
    unittest.main()
