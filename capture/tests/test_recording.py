"""Offline-only tests: no native library, PC service, or Quest connection."""

import ctypes
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_recording import analyze, parse_value, summary_path, write_summary
from raw_recorder import CALLBACK_MASK, DeviceStateJson, RawRecorder, main, record


def state_event(app_time, left='0,0,0,0,0,0,1', right='0,0,0,0,0,0,1',
                *, wall_ns=1_800_000_000_000_000_000, recv_ns=1_000_000_000,
                focus=True):
    value = {
        'predictTime': app_time, 'timeStampNs': wall_ns,
        'appState': {'focus': focus},
        'Head': {'status': 3},
        'Input': 2,
        'Controller': {'left': {'pose': left}, 'right': {'pose': right}},
    }
    return {
        'recv_monotonic_ns': recv_ns, 'recv_wall_ns': 1_800_000_000_000_000_000,
        'event_type': 'PXREADeviceStateJson', 'status': 0,
        'device_id': 'quest',
        'payload': json.dumps({'functionName': 'Tracking', 'value': json.dumps(value)}),
    }


def write_events(path, events):
    path.write_text(''.join(json.dumps(event) + '\n' for event in events),
                    encoding='utf-8')


class AnalyzerTests(unittest.TestCase):
    def test_quest_wrapper(self):
        event = state_event(1000000)
        value = parse_value(event['payload'])
        self.assertEqual(value['Controller']['left']['pose'], '0,0,0,0,0,0,1')

    def test_dedup_gaps_jumps_restart_focus_and_wall_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'raw.jsonl'
            events = [
                state_event(1_000_000),
                state_event(1_000_000, recv_ns=1_001_000_000),
                state_event(1_010_000, recv_ns=1_010_000_000),
                state_event(1_310_000, left='1,0,0,0,0,0,1',
                            recv_ns=1_310_000_000, focus=False),
                state_event(500_000, wall_ns=638_000_000_000_000_000,
                            recv_ns=1_320_000_000),
            ]
            write_events(path, events)
            result = analyze(path)
            left = next(stream for stream in result['streams'] if stream['side'] == 'left')
            self.assertEqual(left['duplicate_timestamps'], 1)
            self.assertEqual(left['unique_samples'], 4)
            self.assertEqual(left['app_uptime_regressions'], 1)
            self.assertEqual(left['app_uptime_epochs'], 2)
            self.assertEqual(left['app_frame_gap_count'], 1)
            self.assertEqual(left['receive_gap_count'], 1)
            self.assertEqual(left['pose_jump_count'], 1)
            self.assertEqual(left['different_pose_count'], 2)
            self.assertEqual(left['longest_identical_pose_duration_ms'], 10.0)
            self.assertEqual(left['longest_identical_pose_run_frames'], 2)
            right = next(stream for stream in result['streams'] if stream['side'] == 'right')
            self.assertEqual(right['different_pose_count'], 1)
            self.assertEqual(right['longest_identical_pose_duration_ms'], 310.0)
            self.assertEqual(result['app_focus_events']['false'], 1)
            self.assertEqual(result['top_level_input_counts'], {'2': 5})
            self.assertEqual(result['head_status_events']['3'], 5)
            self.assertEqual(result['top_level_timestamp_domains'][
                'possible_dotnet_ticks_fallback'], 1)
            self.assertEqual(result['controller_sides_with_numeric_pose'], ['left', 'right'])
            self.assertFalse(result['true_6dof_proven'])

    def test_reanalysis_preserves_matching_capture_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'raw.jsonl'
            write_events(path, [state_event(1_000_000)])
            sidecar = summary_path(path)
            capture = {'sdk_init_result': 0, 'stop_reason': 'duration_complete'}
            sidecar.write_text(json.dumps({'input': str(path), 'capture': capture}),
                               encoding='utf-8')
            result, destination = write_summary(path)
            self.assertEqual(destination, sidecar)
            self.assertEqual(result['capture'], capture)
            self.assertEqual(json.loads(sidecar.read_text())['capture'], capture)
            custom = Path(temp) / 'custom.json'
            custom.write_text(json.dumps({'input': str(Path(temp) / 'other.jsonl'),
                                          'capture': {'wrong': True}}), encoding='utf-8')
            custom_result, _ = write_summary(path, custom)
            self.assertEqual(custom_result['capture'], capture)
            self.assertEqual(result['optical_tracking_validity'], 'unknown')
            self.assertNotIn('transport_check', result)

    def test_empty_or_invalid_pose_never_proves_tracking(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'raw.jsonl'
            write_events(path, [])
            result = analyze(path)
            self.assertFalse(result['has_state_frames'])
            self.assertFalse(result['has_both_controller_pose_fields'])
            event = state_event(1_000_000, left='nan,0,0,0,0,0,1')
            write_events(path, [event])
            result = analyze(path)
            left = next(stream for stream in result['streams'] if stream['side'] == 'left')
            self.assertEqual(left['invalid_pose_samples'], 1)
            self.assertEqual(result['controller_sides_with_numeric_pose'], ['right'])
            self.assertFalse(result['true_6dof_proven'])


class FakeFunction:
    def __init__(self, invoke):
        self.invoke = invoke

    def __call__(self, *args):
        return self.invoke(*args)


class FakeSDK:
    def __init__(self, send_state=False):
        self.calls = []
        self.send_state = send_state
        self.PXREAInit = FakeFunction(self.init)
        self.PXREADeinit = FakeFunction(self.deinit)

    def init(self, _context, callback, mask):
        self.calls.append('init')
        assert mask == CALLBACK_MASK
        if self.send_state:
            state = DeviceStateJson()
            state.devID = b'quest-test'
            state.stateJson = state_event(1_000_000)['payload'].encode()
            callback(None, 1 << 2, 0, None)
            callback(None, 1 << 25, 0, ctypes.cast(ctypes.pointer(state), ctypes.c_void_p))
        return 0

    def deinit(self):
        self.calls.append('deinit')
        return 0


class RecorderTests(unittest.TestCase):
    def test_sdk_state_struct_matches_header(self):
        self.assertEqual(ctypes.sizeof(DeviceStateJson), 32 + 16352)
        self.assertEqual(DeviceStateJson.stateJson.offset, 32)

    def test_callback_copies_full_json_and_deinits(self):
        with tempfile.TemporaryDirectory() as temp:
            sdk = FakeSDK(send_state=True)
            output = Path(temp) / 'raw.jsonl'
            summary, destination = record(
                output=output, sdk_lib='fake', seconds=0.003,
                wait_seconds=0.003, sdk_loader=lambda _: sdk)
            self.assertEqual(sdk.calls, ['init', 'deinit'])
            self.assertTrue(destination.exists())
            self.assertTrue(summary['has_state_frames'])
            self.assertTrue(summary['capture']['server_connect_observed'])
            lines = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(lines[1]['payload'], state_event(1_000_000)['payload'])
            self.assertEqual(lines[1]['device_id'], 'quest-test')
            self.assertIn('recv_monotonic_ns', lines[1])
            self.assertIn('recv_wall_ns', lines[1])

    def test_no_first_frame_still_writes_summary_and_deinits(self):
        with tempfile.TemporaryDirectory() as temp:
            sdk = FakeSDK()
            output = Path(temp) / 'raw.jsonl'
            summary, destination = record(
                output=output, sdk_lib='fake', seconds=0.003,
                wait_seconds=0.003, sdk_loader=lambda _: sdk)
            self.assertEqual(sdk.calls, ['init', 'deinit'])
            self.assertTrue(destination.exists())
            self.assertFalse(summary['has_state_frames'])
            self.assertEqual(summary['capture']['stop_reason'], 'first_state_timeout')
            self.assertFalse(summary['capture']['server_connect_observed'])

    def test_existing_raw_file_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'raw.jsonl'
            output.write_text('original\n', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                RawRecorder(output)
            self.assertEqual(output.read_text(encoding='utf-8'), 'original\n')

    def test_cli_exit_requires_complete_clean_bilateral_numeric_frames(self):
        bilateral = [{'side': side, 'valid_pose_samples': 2, 'unique_samples': 2}
                     for side in ('left', 'right')]
        base = {'device_state_events': 2, 'parse_error_count': 0,
                'streams': bilateral,
                'capture': {'stop_reason': 'duration_complete', 'callback_errors': [],
                            'sdk_deinit_result': 0}}
        cases = [('complete', {}, 0),
                 ('callback_error', {'capture': {**base['capture'],
                                                'callback_errors': ['bad callback']}}, 1),
                 ('service_disconnect', {'capture': {**base['capture'],
                                                     'stop_reason': 'server_disconnected'}}, 1),
                 ('exception', {'capture': {**base['capture'],
                                           'stop_reason': 'exception'}}, 1),
                 ('interrupted', {'capture': {**base['capture'],
                                             'stop_reason': 'interrupted'}}, 130),
                 ('one_side', {'streams': bilateral[:1]}, 1)]
        for name, changes, expected in cases:
            with self.subTest(name=name), patch('raw_recorder.record') as fake_record:
                fake_record.return_value = ({**base, **changes}, Path('/tmp/summary.json'))
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main(['--output', '/tmp/test-unused-recorder.jsonl'])
                self.assertEqual(code, expected)


if __name__ == '__main__':
    unittest.main()
