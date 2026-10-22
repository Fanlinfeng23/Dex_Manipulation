"""Read-only PXREA Quest callback to local protocol-v1 WebSocket bridge.

Only PXREAInit/PXREADeinit are used. This process never sends SDK device
controls, calibration commands, or robot commands. A local WebSocket drop
causes the arm server's existing watchdog/disconnect pause.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import math
import os
import signal
import tempfile
import threading
import time
from pathlib import Path

from aiohttp import ClientSession, WSMsgType

from .native_adapter import NativeAdapter


SERVER_CONNECT = 1 << 2
SERVER_DISCONNECT = 1 << 3
DEVICE_FIND = 1 << 4
DEVICE_MISSING = 1 << 5
DEVICE_STATE_JSON = 1 << 25
EVENT_NAMES = {
    SERVER_CONNECT: 'sdk_server_connect',
    SERVER_DISCONNECT: 'sdk_server_disconnect',
    DEVICE_FIND: 'sdk_device_find',
    DEVICE_MISSING: 'sdk_device_missing',
    DEVICE_STATE_JSON: 'sdk_device_state_json',
}
CALLBACK_MASK = sum(EVENT_NAMES)
CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                            ctypes.c_int, ctypes.c_void_p)
CAPTURE_ROOT = Path(os.environ.get('TIANJI_QUEST_CAPTURE_DIR',
                                   str(Path(__file__).resolve().parents[3] / 'capture')))
SERVICE_ROOT = Path(os.environ.get('TIANJI_XROBOT_SERVICE_DIR',
                                   str(CAPTURE_ROOT / 'native/opt/apps/roboticsservice')))
DEFAULT_SDK = Path(os.environ.get('TIANJI_XROBOT_SDK',
                                  str(SERVICE_ROOT / 'SDK/x64/libPXREARobotSDK.so')))


def loss_edges(frame):
    safety = frame.get('safety') or {}
    focus_loss = bool(safety.get('focus_lost')) or frame.get('focused') is False
    tracking_loss = (bool(safety.get('tracking_lost')) or focus_loss or
                     any(not controller.get('tracked', False)
                         for controller in frame.get('controllers', {}).values()))
    return focus_loss, tracking_loss


def carry_loss_edges(previous, current):
    """Keep a lost-tracking/focus sample through newest-frame coalescing."""
    focus_loss, tracking_loss = loss_edges(previous)
    if not focus_loss and not tracking_loss:
        return current
    merged = dict(current)
    merged['safety'] = {**current.get('safety', {}),
                        'focus_lost': focus_loss or current.get('safety', {}).get('focus_lost', False),
                        'tracking_lost': True}
    if focus_loss:
        merged['focused'] = False
    merged['controllers'] = {
        side: {**controller, 'tracked': False}
        for side, controller in current.get('controllers', {}).items()}
    return merged


class DeviceStateJson(ctypes.Structure):
    _fields_ = [('devID', ctypes.c_char * 32),
                ('stateJson', ctypes.c_char * 16352)]


class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.file = self.path.open('a', encoding='utf-8', buffering=1)

    def write(self, event, **fields):
        item = {'wall_ns': time.time_ns(), 'monotonic_s': time.monotonic(),
                'event': event, **fields}
        with self.lock:
            self.file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')

    def close(self):
        with self.lock:
            self.file.close()


def atomic_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8',
                                         dir=destination.parent,
                                         prefix=f'.{destination.name}.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class NativeSdkIngress:
    """Copy/parse SDK callbacks and coalesce them into one newest pending frame."""

    def __init__(self, loop, log):
        self.loop = loop
        self.log = log
        self.adapter = NativeAdapter()
        self.queue = asyncio.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.latest = None
        self.flush_scheduled = False
        self.sdk_connected = False
        self.fault_reason = None
        self.dropped = 0
        self.callback_errors = 0
        self._last_reason = self.adapter.reason
        self.callback = CALLBACK(self._on_callback)

    def snapshot(self):
        with self.lock:
            status = self.adapter.status
            return {**status, 'sdk_connected': self.sdk_connected,
                    'fault_reason': self.fault_reason,
                    'coalesced_frames': self.dropped,
                    'callback_errors': self.callback_errors}

    def _clear_queue(self):
        while not self.queue.empty():
            self.queue.get_nowait()

    def _flush(self):
        with self.lock:
            entry = self.latest
            self.latest = None
            self.flush_scheduled = False
            if entry is not None and entry[0]['session'] != self.adapter.session:
                entry = None
        if entry is None:
            return
        with self.lock:
            if entry[0]['session'] != self.adapter.session:
                return
            if self.queue.full():
                old, _ = self.queue.get_nowait()
                entry = (carry_loss_edges(old, entry[0]), entry[1])
                self.dropped += 1
            self.queue.put_nowait(entry)

    def _push_locked(self, frame, received_s):
        if self.latest is not None:
            self.dropped += 1
            frame = carry_loss_edges(self.latest[0], frame)
        self.latest = (frame, received_s)
        if not self.flush_scheduled:
            self.flush_scheduled = True
            self.loop.call_soon_threadsafe(self._flush)

    def reset(self, reason):
        with self.lock:
            self.adapter.reset_epoch(reason)
            self.latest = None
            self.fault_reason = None
            self._last_reason = reason
        self._clear_queue()
        self.log.write('input_epoch_reset', reason=reason)

    def check_stale(self, now_s):
        with self.lock:
            stale = self.adapter.check_stale(now_s)
            if stale:
                self.latest = None
                self.fault_reason = 'native_stream_timeout'
                self._last_reason = self.adapter.reason
        if stale:
            self._clear_queue()
            self.log.write('native_stream_timeout')
        return stale

    def _on_callback(self, _context, event_type, status, user_data):
        received_s = time.monotonic()
        try:
            if event_type == DEVICE_STATE_JSON:
                if not user_data:
                    raise ValueError('NULL PXREADevStateJson')
                state = ctypes.cast(user_data, ctypes.POINTER(DeviceStateJson)).contents
                device_id = bytes(state.devID).split(b'\0', 1)[0].decode(
                    'utf-8', errors='replace')
                payload = bytes(state.stateJson).split(b'\0', 1)[0].decode(
                    'utf-8', errors='replace')
                if status != 0:
                    raise ValueError(f'SDK state status {status}')
                with self.lock:
                    frame = self.adapter.consume(payload, device_id, received_s)
                    reason = self.adapter.reason
                    if reason != self._last_reason:
                        self._last_reason = reason
                        self.log.write('native_gate', reason=reason,
                                       session=self.adapter.session,
                                       eligible=self.adapter.eligible)
                    if frame is not None:
                        self._push_locked(frame, received_s)
                    elif reason in ('invalid_native_json', 'missing_app_uptime'):
                        self.fault_reason = reason
                return
            name = EVENT_NAMES.get(event_type, f'sdk_event_{event_type}')
            device_id = (ctypes.string_at(user_data).decode('utf-8', errors='replace')
                         if user_data else None)
            with self.lock:
                if event_type == SERVER_CONNECT:
                    self.sdk_connected = status == 0
                elif event_type == SERVER_DISCONNECT:
                    self.sdk_connected = False
                    self.adapter.reset_epoch('sdk_server_disconnected')
                    self.latest = None
                    self.fault_reason = 'sdk_server_disconnected'
                elif (event_type == DEVICE_MISSING and device_id == self.adapter.device_id):
                    self.adapter.reset_epoch('sdk_device_missing')
                    self.latest = None
                    self.fault_reason = 'sdk_device_missing'
            self.log.write(name, device_id=device_id, status=int(status))
        except Exception as exc:
            with self.lock:
                self.callback_errors += 1
                self.adapter.invalidate('sdk_callback_error', rotate=True)
                self.latest = None
                self.fault_reason = 'sdk_callback_error'
            self.log.write('sdk_callback_error', error=f'{type(exc).__name__}: {exc}')


class NativeWsSender:
    """One frame in flight; ACK, age and rate limits prevent old-pose replay."""

    MAX_RATE_HZ = 60.0
    MAX_AGE_S = 0.15
    ACK_TIMEOUT_S = 0.15

    def __init__(self, ws):
        self.ws = ws
        self.welcomed = False
        self.latest = None
        self.inflight = None
        self.last_send_s = -math.inf
        self.sent = 0
        self.acked = 0
        self.dropped_stale = 0
        self.focus_loss_pending = False
        self.tracking_loss_pending = False
        self.loss_generation = 0

    def offer(self, frame, received_s):
        focus_loss, tracking_loss = loss_edges(frame)
        if focus_loss or tracking_loss:
            self.focus_loss_pending |= focus_loss
            self.tracking_loss_pending |= tracking_loss
            self.loss_generation += 1
        if self.latest is not None:
            frame = carry_loss_edges(self.latest[0], frame)
        self.latest = (frame, received_s)

    def on_message(self, data, now_s):
        if data.get('type') == 'welcome':
            if data.get('control') is not True or data.get('frame_ack') is not True:
                raise RuntimeError('arm server did not grant quest frame control/ACK')
            self.welcomed = True
        elif data.get('type') == 'frame_ack':
            if self.inflight is None:
                raise RuntimeError('unexpected frame_ack')
            frame, sent_at, sent_loss_generation = self.inflight
            if (data.get('session'), data.get('seq')) != (frame['session'], frame['seq']):
                raise RuntimeError('frame_ack session/seq mismatch')
            if now_s - sent_at > self.ACK_TIMEOUT_S:
                raise RuntimeError('late frame_ack')
            if data.get('accepted') is not True:
                raise RuntimeError(f"frame rejected: {data.get('reason', 'unknown')}")
            if sent_loss_generation == self.loss_generation:
                self.focus_loss_pending = False
                self.tracking_loss_pending = False
            self.inflight = None
            self.acked += 1
        elif data.get('type') == 'error':
            raise RuntimeError(f"arm server error: {data.get('message', 'unknown')}")

    async def pump(self, now_s):
        if self.inflight is not None:
            if now_s - self.inflight[1] > self.ACK_TIMEOUT_S:
                raise RuntimeError('frame_ack_timeout')
            return
        if not self.welcomed or self.latest is None:
            return
        if now_s - self.last_send_s < 1 / self.MAX_RATE_HZ:
            return
        frame, received_s = self.latest
        self.latest = None
        if now_s - received_s > self.MAX_AGE_S:
            self.dropped_stale += 1
            return
        if self.focus_loss_pending or self.tracking_loss_pending:
            frame = carry_loss_edges({
                'focused': not self.focus_loss_pending,
                'controllers': {'left': {'tracked': not self.tracking_loss_pending},
                                'right': {'tracked': not self.tracking_loss_pending}},
                'safety': {'focus_lost': self.focus_loss_pending,
                           'tracking_lost': self.tracking_loss_pending},
            }, frame)
        await asyncio.wait_for(self.ws.send_json(frame), timeout=self.ACK_TIMEOUT_S)
        sent_at = time.monotonic()
        self.inflight = (frame, sent_at, self.loss_generation)
        self.last_send_s = sent_at
        self.sent += 1


class NativeBridge:
    def __init__(self, *, port, token_file, status_file, log_file, sdk_lib):
        self.port = port
        token = Path(token_file).read_text(encoding='utf-8').strip()
        if len(token) < 16 or any(character.isspace() for character in token):
            raise ValueError('token-file must contain one nonempty random token')
        self.token = token
        self.status_file = Path(status_file)
        self.log = EventLog(log_file)
        self.sdk_lib = Path(sdk_lib)
        self.stop = asyncio.Event()
        self.ingress = None
        self.sender = None
        self.ws_connected = False
        self.last_error = None
        self.sdk = None
        self.sdk_initialized = False

    def status(self):
        source = self.ingress.snapshot() if self.ingress else {
            'eligible': False, 'tracking_validity': 'unreported',
            'reason': 'starting', 'last_rx': None}
        ready = bool(self.sender and self.sender.welcomed and self.ws_connected)
        eligible = bool(source.get('eligible') and source.get('sdk_connected') and ready)
        if not source.get('sdk_connected'):
            reason = 'sdk_server_disconnected' if self.ingress else 'starting'
            reason_zh = '等待 XRoboToolkit 原生服务连接' if self.ingress else '正在启动'
        elif not ready:
            reason = 'bridge_websocket_disconnected' if not self.ws_connected else 'bridge_waiting_for_welcome'
            reason_zh = '等待天机控制服务连接' if not self.ws_connected else '等待天机控制服务授权'
        else:
            reason = source.get('reason')
            reason_zh = source.get('reason_zh')
        return {
            'mode': 'XRoboToolkit native Quest to Tianji arms',
            'wall_ns': time.time_ns(), 'monotonic_s': time.monotonic(),
            **source, 'adapter_eligible': source.get('eligible', False),
            'eligible': eligible, 'reason': reason, 'reason_zh': reason_zh,
            'ws_connected': self.ws_connected, 'ws_ready': ready,
            'sent_frames': self.sender.sent if self.sender else 0,
            'acked_frames': self.sender.acked if self.sender else 0,
            'dropped_stale_frames': self.sender.dropped_stale if self.sender else 0,
            'last_error': self.last_error,
        }

    async def status_loop(self):
        while not self.stop.is_set():
            atomic_json(self.status_file, self.status())
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    async def websocket_loop(self):
        url = f'http://127.0.0.1:{self.port}/ws/quest'
        async with ClientSession() as client:
            while not self.stop.is_set():
                try:
                    async with client.ws_connect(url, heartbeat=5.0,
                                                 max_msg_size=1 << 20,
                                                 headers={'X-Teleop-Token': self.token}) as ws:
                        self.ws_connected = True
                        self.sender = NativeWsSender(ws)
                        self.ingress.reset('bridge_websocket_connected')
                        self.log.write('arm_websocket_connected', url=url)
                        reader = asyncio.create_task(self.read_websocket(ws, self.sender))
                        try:
                            while not self.stop.is_set():
                                if reader.done():
                                    reader.result()
                                    raise RuntimeError('arm websocket closed')
                                now_s = time.monotonic()
                                if self.ingress.check_stale(now_s):
                                    raise RuntimeError('native_stream_timeout')
                                ingress_status = self.ingress.snapshot()
                                fault = ingress_status['fault_reason']
                                if fault:
                                    raise RuntimeError(fault)
                                while not self.ingress.queue.empty():
                                    frame, received_s = self.ingress.queue.get_nowait()
                                    if frame['session'] == ingress_status['session']:
                                        self.sender.offer(frame, received_s)
                                if ingress_status['sdk_connected']:
                                    await self.sender.pump(now_s)
                                await asyncio.sleep(0.005)
                        finally:
                            reader.cancel()
                            try:
                                await reader
                            except asyncio.CancelledError:
                                pass
                except Exception as exc:
                    self.last_error = f'{type(exc).__name__}: {exc}'
                    self.log.write('arm_websocket_fault', error=self.last_error)
                finally:
                    self.ws_connected = False
                    self.ingress.reset('bridge_websocket_disconnected')
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass

    async def read_websocket(self, ws, sender):
        async for message in ws:
            if message.type == WSMsgType.TEXT:
                data = json.loads(message.data)
                if isinstance(data, dict):
                    sender.on_message(data, time.monotonic())
            elif message.type in (WSMsgType.ERROR, WSMsgType.CLOSED, WSMsgType.CLOSE):
                break

    def load_sdk(self):
        sdk = ctypes.CDLL(str(self.sdk_lib))
        sdk.PXREAInit.argtypes = [ctypes.c_void_p, CALLBACK, ctypes.c_uint]
        sdk.PXREAInit.restype = ctypes.c_int
        sdk.PXREADeinit.argtypes = []
        sdk.PXREADeinit.restype = ctypes.c_int
        self.sdk = sdk
        result = sdk.PXREAInit(None, self.ingress.callback, CALLBACK_MASK)
        if result != 0:
            raise RuntimeError(f'PXREAInit returned {result}')
        self.sdk_initialized = True
        self.log.write('sdk_initialized', sdk_lib=str(self.sdk_lib))

    async def run(self):
        self.ingress = NativeSdkIngress(asyncio.get_running_loop(), self.log)
        self.log.write('bridge_started', port=self.port)
        atomic_json(self.status_file, self.status())
        tasks = []
        stop_waiter = None
        try:
            self.load_sdk()
            tasks = [asyncio.create_task(self.status_loop()),
                     asyncio.create_task(self.websocket_loop())]
            stop_waiter = asyncio.create_task(self.stop.wait())
            done, _ = await asyncio.wait([stop_waiter, *tasks],
                                         return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task is not stop_waiter:
                    task.result()
                    if not self.stop.is_set():
                        raise RuntimeError('bridge background task exited unexpectedly')
        except Exception as exc:
            self.last_error = f'{type(exc).__name__}: {exc}'
            self.log.write('bridge_fatal', error=self.last_error)
            raise
        finally:
            self.stop.set()
            if stop_waiter is not None:
                stop_waiter.cancel()
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if stop_waiter is not None:
                await asyncio.gather(stop_waiter, return_exceptions=True)
            if self.sdk_initialized:
                try:
                    result = self.sdk.PXREADeinit()
                    self.log.write('sdk_deinitialized', result=int(result))
                except Exception as exc:
                    self.last_error = f'PXREADeinit: {exc}'
                    self.log.write('sdk_deinit_failed', error=self.last_error)
            self.ws_connected = False
            if self.ingress:
                self.ingress.reset('bridge_stopped')
            atomic_json(self.status_file, self.status())
            self.log.write('bridge_stopped')
            self.log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='只读 XRoboToolkit 原生 Quest 双臂输入桥')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--status-file', type=Path, required=True)
    parser.add_argument('--log-file', type=Path, required=True)
    parser.add_argument('--sdk-lib', type=Path, default=DEFAULT_SDK)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port 必须为 1–65535')
    bridge = NativeBridge(port=args.port, token_file=args.token_file,
                          status_file=args.status_file,
                          log_file=args.log_file, sdk_lib=args.sdk_lib)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, bridge.stop.set)
    try:
        loop.run_until_complete(bridge.run())
    finally:
        loop.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
