"""Native JSON to real vendor kinematics through WS/IPC; no robot connection."""
import asyncio
import contextlib
import json
from pathlib import Path
import tempfile
import time
import unittest

from aiohttp.test_utils import AioHTTPTestCase

from native_launcher import profile
from quest_teleop.keyboard import KeyboardClient
from quest_teleop.native_adapter import NativeAdapter
from quest_teleop.native_server import build_app


class Console:
    def __init__(self):
        self.actions = []

    def poll(self):
        actions, self.actions = self.actions, []
        return actions


class NativeRuntimeTests(AioHTTPTestCase):
    async def get_application(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.token = 'native-test-only-' * 3
        token_path = self.folder / 'token'
        token_path.write_text(self.token)
        config = profile({})
        # Match the completed joint startup pose, virtually.
        for side in ('left', 'right'):
            config['arms'][side]['initial_joints_deg'] = config['hardware']['startup_joints_deg'][side]
        config_path = self.folder / 'config.json'
        config_path.write_text(json.dumps(config))
        self.status_path = self.folder / 'native.json'
        return build_app(config_path, native_status_file=self.status_path, token_file=token_path)

    async def asyncTearDown(self):
        await super().asyncTearDown()
        self.temporary.cleanup()

    async def until(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            await asyncio.sleep(.02)
        self.fail('Native runtime condition timeout')

    async def test_auto_start_space_reanchor_and_stream_loss_with_vendor_kinematics(self):
        headers = {'X-Teleop-Token': self.token}
        bridge = await self.client.ws_connect('/ws/quest', headers=headers)
        keyboard_ws = await self.client.ws_connect('/ws/keyboard', headers=headers)
        self.assertEqual((await bridge.receive_json())['type'], 'welcome')
        self.assertEqual((await keyboard_ws.receive_json())['type'], 'welcome')
        console = Console()
        client = KeyboardClient(keyboard_ws, report=lambda _: None, hardware=False,
                                auto_calibrate=True, auto_follow_on_ready=True,
                                native_arm_mode=True)
        adapter = NativeAdapter(bridge_uuid='offline-integration')
        running = True
        offset = 0.0

        async def feed():
            index = 0
            while running:
                now = time.perf_counter()
                # Continuous submillimetre movement prevents ambiguity with a
                # deliberately frozen pose in the unmodified official format.
                jitter = (index % 2) * .0001
                value = {'predictTime': now * 1e6, 'Input': 1,
                         'appState': {'focus': True},
                         'Head': {'status': 3, 'pose': '0,1.5,0,0,0,0,1'},
                         'Controller': {s: {'pose': f'{x+jitter},1,-{.4+offset},0,0,0,1'}
                                        for s, x in (('left', -.2), ('right', .2))}}
                frame = adapter.consume(value, 'test-device', now)
                self.status_path.write_text(json.dumps({**adapter.status, 'wall_ns': time.time_ns(),
                                                        'ws_connected': True}))
                await bridge.send_json(frame)
                while True:
                    reply = await bridge.receive_json(timeout=1)
                    if reply.get('type') == 'frame_ack':
                        self.assertTrue(reply['accepted'], reply)
                        break
                index += 1
                await asyncio.sleep(.015)

        feeder = asyncio.create_task(feed())
        keyboard = asyncio.create_task(client.run(console))
        control = self.app['preview'].control
        try:
            await self.until(lambda: (control.state or {}).get('following'), timeout=6)
            self.assertFalse(control.state['hardware'])
            self.assertTrue(control.state['keyboard_follow'])
            before = {s: control.state['arms'][s]['actual']['position'] for s in ('left', 'right')}
            offset = .02
            await asyncio.sleep(.45)
            for side in ('left', 'right'):
                target = control.state['arms'][side]['target']['position']
                self.assertAlmostEqual(target[0] - before[side][0], .02, delta=.003)
            console.actions = ['keyboard_toggle']
            await self.until(lambda: not control.state['keyboard_follow'])
            held = {s: control.state['arms'][s]['actual']['position'] for s in ('left', 'right')}
            offset = .06
            await asyncio.sleep(.15)
            console.actions = ['keyboard_toggle']
            await self.until(lambda: control.state['keyboard_follow'])
            await asyncio.sleep(.15)
            for side in ('left', 'right'):
                current = control.state['arms'][side]['actual']['position']
                self.assertAlmostEqual(current[0], held[side][0], delta=.003)
            running = False
            await feeder
            await self.until(lambda: not control.state['enabled'], timeout=1)
            self.assertFalse(control.state['following'])
            console.actions = ['quit']
            await asyncio.wait_for(keyboard, 2)
        finally:
            running = False
            for task in (feeder, keyboard):
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await bridge.close()
            await keyboard_ws.close()


if __name__ == '__main__':
    unittest.main()
