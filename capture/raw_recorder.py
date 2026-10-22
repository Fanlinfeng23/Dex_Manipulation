#!/usr/bin/env python3
"""Record raw XRoboToolkit SDK callbacks. Read-only: no device control calls."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import signal
import threading
import time
from pathlib import Path

from analyze_recording import summary_path, write_summary


EVENTS = {
    1 << 2: 'PXREAServerConnect',
    1 << 3: 'PXREAServerDisconnect',
    1 << 4: 'PXREADeviceFind',
    1 << 5: 'PXREADeviceMissing',
    1 << 9: 'PXREADeviceConnect',
    1 << 25: 'PXREADeviceStateJson',
}
CALLBACK_MASK = sum(EVENTS)
STATE_EVENT = 1 << 25
CONNECT_EVENT = 1 << 2
CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                            ctypes.c_int, ctypes.c_void_p)


class DeviceStateJson(ctypes.Structure):
    _fields_ = [('devID', ctypes.c_char * 32),
                ('stateJson', ctypes.c_char * 16352)]


class RawRecorder:
    def __init__(self, output):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.output.open('x', encoding='utf-8', buffering=1)
        self.lock = threading.Lock()
        self.first_state = threading.Event()
        self.server_connected = threading.Event()
        self.server_disconnected = threading.Event()
        self.stop = threading.Event()
        self.first_state_monotonic_ns = None
        self.state_count = 0
        self.callback_errors = []
        # Keep the C callback alive until PXREADeinit has returned.
        self.callback = CALLBACK(self._on_callback)

    def _on_callback(self, _context, event_code, status, user_data):
        monotonic_ns = time.monotonic_ns()
        wall_ns = time.time_ns()
        try:
            event_type = EVENTS.get(event_code, f'PXREAUnknown_{event_code}')
            entry = {
                'recv_monotonic_ns': monotonic_ns,
                'recv_wall_ns': wall_ns,
                'event_type': event_type,
                'status': int(status),
            }
            if event_code == STATE_EVENT:
                if not user_data:
                    raise ValueError('state callback has NULL userData')
                state = ctypes.cast(user_data, ctypes.POINTER(DeviceStateJson)).contents
                # The SDK owns this stack object only for the callback duration.
                entry['device_id'] = bytes(state.devID).split(b'\0', 1)[0].decode(
                    'utf-8', errors='replace')
                entry['payload'] = bytes(state.stateJson).split(b'\0', 1)[0].decode(
                    'utf-8', errors='replace')
            elif user_data:
                entry['device_id'] = ctypes.string_at(user_data).decode(
                    'utf-8', errors='replace')
            with self.lock:
                self.file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                if event_code == STATE_EVENT:
                    self.state_count += 1
                    if self.first_state_monotonic_ns is None:
                        self.first_state_monotonic_ns = monotonic_ns
                        self.first_state.set()
                elif event_code == CONNECT_EVENT:
                    self.server_connected.set()
                elif event_code == (1 << 3):
                    self.server_disconnected.set()
        except Exception as exc:  # C callback exceptions must never escape to ctypes.
            with self.lock:
                self.callback_errors.append(f'{type(exc).__name__}: {exc}')

    def close(self):
        with self.lock:
            if not self.file.closed:
                self.file.flush()
                self.file.close()


def configure_sdk(sdk):
    sdk.PXREAInit.argtypes = [ctypes.c_void_p, CALLBACK, ctypes.c_uint]
    sdk.PXREAInit.restype = ctypes.c_int
    sdk.PXREADeinit.argtypes = []
    sdk.PXREADeinit.restype = ctypes.c_int


def record(*, output, sdk_lib, seconds, wait_seconds, sdk_loader=ctypes.CDLL):
    """Record for `seconds` after the first state frame; wait at most `wait_seconds`."""
    recorder = RawRecorder(output)
    metadata = {
        'sdk_lib': str(sdk_lib), 'requested_seconds': seconds,
        'wait_seconds': wait_seconds, 'sdk_init_result': None,
        'sdk_deinit_result': None, 'stop_reason': None,
        'server_connect_observed': False,
    }
    sdk = None
    initialized = False
    previous_handlers = {}
    try:
        def request_stop(_signum, _frame):
            recorder.stop.set()

        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, request_stop)
        sdk = sdk_loader(str(sdk_lib))
        configure_sdk(sdk)
        init_result = sdk.PXREAInit(None, recorder.callback, CALLBACK_MASK)
        metadata['sdk_init_result'] = int(init_result)
        if init_result != 0:
            metadata['stop_reason'] = 'sdk_init_failed'
        else:
            initialized = True
            deadline_ns = time.monotonic_ns() + round(wait_seconds * 1e9)
            while (not recorder.stop.is_set() and not recorder.first_state.is_set()
                   and not recorder.server_disconnected.is_set()):
                remaining = (deadline_ns - time.monotonic_ns()) / 1e9
                if remaining <= 0:
                    metadata['stop_reason'] = 'first_state_timeout'
                    break
                recorder.stop.wait(min(remaining, 0.05))
            if (recorder.first_state.is_set() and not recorder.stop.is_set()
                    and not recorder.server_disconnected.is_set()):
                print(f'已收到首个状态帧，开始记录 {seconds:g} 秒。', flush=True)
                target_ns = recorder.first_state_monotonic_ns + round(seconds * 1e9)
                while (not recorder.stop.is_set()
                       and not recorder.server_disconnected.is_set()):
                    remaining = (target_ns - time.monotonic_ns()) / 1e9
                    if remaining <= 0:
                        metadata['stop_reason'] = 'duration_complete'
                        break
                    recorder.stop.wait(min(remaining, 0.1))
            if recorder.server_disconnected.is_set():
                metadata['stop_reason'] = 'server_disconnected'
            elif recorder.stop.is_set():
                metadata['stop_reason'] = 'interrupted'
    except Exception as exc:
        metadata['stop_reason'] = 'exception'
        metadata['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if initialized:
            try:
                metadata['sdk_deinit_result'] = int(sdk.PXREADeinit())
            except Exception as exc:
                metadata['deinit_error'] = f'{type(exc).__name__}: {exc}'
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        metadata['server_connect_observed'] = recorder.server_connected.is_set()
        metadata['raw_state_events_in_callback'] = recorder.state_count
        metadata['callback_errors'] = recorder.callback_errors[:20]
        recorder.close()
        try:
            summary, destination = write_summary(output)
            summary['capture'] = metadata
            destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                                   encoding='utf-8')
        except Exception as exc:
            metadata['analysis_error'] = f'{type(exc).__name__}: {exc}'
            destination = summary_path(output)
            summary = {'capture': metadata, 'true_6dof_proven': False,
                       'has_state_frames': False, 'analysis_error': metadata['analysis_error']}
            destination.write_text(json.dumps(summary,
                                              ensure_ascii=False, indent=2) + '\n',
                                   encoding='utf-8')
    return summary, destination


def main(argv=None):
    parser = argparse.ArgumentParser(description='只读记录 PXREA SDK 原始 JSON 回调')
    parser.add_argument('--seconds', type=float, default=30.0,
                        help='收到首个状态帧后记录秒数')
    parser.add_argument('--wait-seconds', type=float, default=30.0,
                        help='等待首个状态帧的最长秒数')
    parser.add_argument('--output', type=Path, required=True, help='原始 JSONL 输出路径')
    parser.add_argument('--sdk-lib', type=Path, default=Path(os.environ.get(
        'TIANJI_XROBOT_SDK', str(Path(os.environ.get('TIANJI_XROBOT_SERVICE_DIR',
            str(Path(__file__).parent / 'native/opt/apps/roboticsservice'))) /
            'SDK/x64/libPXREARobotSDK.so'))))
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.seconds, args.wait_seconds)):
        parser.error('--seconds 和 --wait-seconds 必须为有限正数')
    try:
        summary, destination = record(
            output=args.output, sdk_lib=args.sdk_lib,
            seconds=args.seconds, wait_seconds=args.wait_seconds)
    except FileExistsError:
        parser.error(f'输出已存在，拒绝覆盖：{args.output}')
    print(f"原始记录：{args.output}；分析：{destination}；"
          f"状态事件 {summary.get('device_state_events', 0)}；"
          f"停止原因 {summary['capture']['stop_reason']}；真实 6DoF 未经证明")
    capture = summary['capture']
    if capture.get('stop_reason') == 'interrupted':
        return 130
    if (capture.get('stop_reason') != 'duration_complete' or
            capture.get('callback_errors') or capture.get('error') or
            capture.get('analysis_error') or capture.get('deinit_error') or
            capture.get('sdk_deinit_result') != 0 or
            summary.get('parse_error_count') or
            not all(any(stream.get('side') == side and
                             stream.get('valid_pose_samples', 0) >= 2 and
                             stream.get('unique_samples', 0) >= 2
                             for stream in summary.get('streams', []))
                     for side in ('left', 'right'))):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
