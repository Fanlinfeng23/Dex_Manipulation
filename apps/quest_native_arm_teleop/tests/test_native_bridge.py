"""Offline bridge tests; no Quest SDK library or arm server is opened."""

import asyncio
import ctypes
import json
import tempfile
import time
import unittest
from pathlib import Path

from quest_teleop.native_bridge import (
    CALLBACK_MASK, DEVICE_STATE_JSON, SERVER_CONNECT, SERVER_DISCONNECT,
    DeviceStateJson, EventLog, NativeBridge, NativeSdkIngress, NativeWsSender,
    atomic_json,
)


def payload(ms, shift):
    value = {
        'predictTime': ms * 1000, 'Input': 1, 'appState': {'focus': True},
        'Head': {'status': 3, 'pose': '0,1,0,0,0,0,1'},
        'Controller': {
            'left': {'pose': f'{shift},0,0,0,0,0,1'},
            'right': {'pose': f'{.2 + shift},0,0,0,0,0,1'},
        },
    }
    return json.dumps({'functionName': 'Tracking', 'value': json.dumps(value)})


class FakeWs:
    def __init__(self):
        self.sent = []

    async def send_json(self, frame):
        self.sent.append(frame)


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_frame_loss_survives_both_coalescing_stages(self):
        def frame(seq, *, lost=False):
            return {'type': 'frame', 'session': 'test-session', 'seq': seq,
                    'focused': not lost,
                    'controllers': {side: {'tracked': not lost}
                                    for side in ('left', 'right')},
                    'safety': {'focus_lost': lost, 'tracking_lost': lost}}

        with tempfile.TemporaryDirectory() as temp:
            log = EventLog(Path(temp) / 'events.jsonl')
            ingress = NativeSdkIngress(asyncio.get_running_loop(), log)
            ingress.reset('test')
            lost = frame(1, lost=True)
            good = frame(2)
            lost['session'] = good['session'] = ingress.snapshot()['session']
            with ingress.lock:
                ingress._push_locked(lost, time.monotonic())
                ingress._push_locked(good, time.monotonic())
            await asyncio.sleep(0)
            merged, _ = ingress.queue.get_nowait()
            self.assertFalse(merged['focused'])
            self.assertTrue(merged['safety']['focus_lost'])
            self.assertTrue(merged['safety']['tracking_lost'])
            self.assertFalse(merged['controllers']['left']['tracked'])
            log.close()

        ws = FakeWs()
        sender = NativeWsSender(ws)
        sender.on_message({'type': 'welcome', 'control': True, 'frame_ack': True},
                          time.monotonic())
        now = time.monotonic()
        sender.offer(frame(1), now)
        await sender.pump(now)
        sender.offer(frame(2, lost=True), time.monotonic())
        sender.offer(frame(3), time.monotonic())
        sender.on_message({'type': 'frame_ack', 'session': 'test-session',
                           'seq': 1, 'accepted': True}, time.monotonic())
        sender.last_send_s = time.monotonic() - .02
        await sender.pump(time.monotonic())
        sent = ws.sent[-1]
        self.assertEqual(sent['seq'], 3)
        self.assertFalse(sent['focused'])
        self.assertTrue(sent['safety']['tracking_lost'])
        self.assertFalse(sent['controllers']['right']['tracked'])

    async def test_sdk_callback_copies_state_and_disconnect_rotates_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            log = EventLog(Path(temp) / 'events.jsonl')
            ingress = NativeSdkIngress(asyncio.get_running_loop(), log)
            self.assertEqual(ctypes.sizeof(DeviceStateJson), 32 + 16352)
            self.assertTrue(CALLBACK_MASK & DEVICE_STATE_JSON)
            ingress.callback(None, SERVER_CONNECT, 0, None)
            state = DeviceStateJson()
            state.devID = b'QuestDevice'
            state.stateJson = payload(1000, .001).encode()
            ingress.callback(None, DEVICE_STATE_JSON, 0,
                             ctypes.cast(ctypes.pointer(state), ctypes.c_void_p))
            await asyncio.sleep(0)
            frame, received_s = ingress.queue.get_nowait()
            self.assertEqual(frame['controllers']['left']['position'][0], .001)
            self.assertTrue(received_s <= time.monotonic())
            first_session = frame['session']
            ingress.callback(None, SERVER_DISCONNECT, 0, None)
            self.assertEqual(ingress.snapshot()['fault_reason'], 'sdk_server_disconnected')
            self.assertNotEqual(ingress.snapshot()['session'], first_session)
            self.assertFalse(ingress.snapshot()['eligible'])
            log.close()

    async def test_ack_single_inflight_stale_drop_and_rejection(self):
        ws = FakeWs()
        sender = NativeWsSender(ws)
        frame = {'type': 'frame', 'session': 's', 'seq': 1}
        sender.on_message({'type': 'welcome', 'control': True, 'frame_ack': True},
                          time.monotonic())
        now = time.monotonic()
        sender.offer(frame, now)
        await sender.pump(now)
        self.assertEqual(len(ws.sent), 1)
        sender.offer({'type': 'frame', 'session': 's', 'seq': 2}, now)
        await sender.pump(now + .01)
        self.assertEqual(len(ws.sent), 1)
        sender.on_message({'type': 'frame_ack', 'session': 's', 'seq': 1,
                           'accepted': True}, time.monotonic())
        sender.last_send_s = time.monotonic() - .02
        await sender.pump(time.monotonic())
        self.assertEqual(len(ws.sent), 2)
        with self.assertRaisesRegex(RuntimeError, 'rejected'):
            sender.on_message({'type': 'frame_ack', 'session': 's', 'seq': 2,
                               'accepted': False, 'reason': 'inactive_source'},
                              time.monotonic())
        stale = NativeWsSender(FakeWs())
        stale.on_message({'type': 'welcome', 'control': True, 'frame_ack': True}, 1)
        stale.offer(frame, 1)
        await stale.pump(1.2)
        self.assertEqual(stale.sent, 0)
        self.assertEqual(stale.dropped_stale, 1)

    async def test_ack_timeout_is_fault(self):
        sender = NativeWsSender(FakeWs())
        sender.inflight = ({'session': 's', 'seq': 1}, 1.0)
        with self.assertRaisesRegex(RuntimeError, 'frame_ack_timeout'):
            await sender.pump(1.16)

    async def test_atomic_status_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'status.json'
            atomic_json(path, {'eligible': False})
            atomic_json(path, {'eligible': True})
            self.assertEqual(json.loads(path.read_text()), {'eligible': True})
            self.assertEqual(list(Path(temp).glob('*.tmp')), [])

    async def test_token_file_required_without_logging_secret(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            token_file = folder / 'token'
            token_file.write_text('secret-token-with-enough-entropy\n')
            bridge = NativeBridge(port=8766, token_file=token_file,
                                  status_file=folder / 'status.json',
                                  log_file=folder / 'events.jsonl', sdk_lib='fake')
            self.assertEqual(bridge.token, 'secret-token-with-enough-entropy')
            self.assertNotIn(bridge.token, json.dumps(bridge.status()))
            bridge.ingress = NativeSdkIngress(asyncio.get_running_loop(), bridge.log)
            bridge.ingress.adapter.eligible = True
            bridge.sender = NativeWsSender(FakeWs())
            bridge.sender.welcomed = True
            bridge.ws_connected = True
            self.assertFalse(bridge.status()['eligible'])  # SDK not connected.
            bridge.ingress.sdk_connected = True
            self.assertTrue(bridge.status()['eligible'])
            bridge.log.close()


if __name__ == '__main__':
    unittest.main()
