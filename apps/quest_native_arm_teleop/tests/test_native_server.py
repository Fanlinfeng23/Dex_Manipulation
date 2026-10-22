"""Offline authentication and source isolation for the native arm snapshot."""
import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import WSServerHandshakeError
from aiohttp.test_utils import AioHTTPTestCase

from quest_teleop.config import APP_ROOT, SDK_ROOT
from quest_teleop.keyboard import (KeyboardClient, run as run_keyboard,
                                   quest_tracking_ready, StableQuestTracking)
from quest_teleop.native_server import build_app
from test_core import packet


@unittest.skipUnless((SDK_ROOT / 'SDK_PYTHON' /
                      ('libKine.dll' if sys.platform == 'win32' else 'libKine.so')).exists(),
                     'Platform vendor SDK required')
class NativeServerTests(AioHTTPTestCase):
    async def get_application(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.token = 't' * 40
        self.token_file = self.folder / 'token'
        self.token_file.write_text(self.token + '\n', encoding='utf-8')
        self.status_file = self.folder / 'status.json'
        self.status_file.write_text(json.dumps({
            'wall_ns': time.time_ns(), 'eligible': True, 'ws_connected': True,
            'tracking_validity': 'unreported', 'reason': 'native_numeric_gate_ready',
        }), encoding='utf-8')
        config = json.loads((APP_ROOT / 'configs/hardware_dual_arm.json').read_text(encoding='utf-8'))
        config['initial_source'] = 'quest'
        self.config_file = self.folder / 'config.json'
        self.config_file.write_text(json.dumps(config), encoding='utf-8')
        return build_app(self.config_file, native_status_file=self.status_file,
                         token_file=self.token_file)

    async def asyncTearDown(self):
        await super().asyncTearDown()
        self.temp.cleanup()

    async def next_type(self, ws, kind):
        for _ in range(50):
            result = await ws.receive_json(timeout=2)
            if result.get('type') == kind:
                return result
        self.fail(f'missing {kind}')

    async def test_only_authorized_bridge_and_keyboard_can_control(self):
        for role in ('quest', 'keyboard'):
            with self.assertRaises(WSServerHandshakeError) as rejected:
                await self.client.ws_connect('/ws/' + role)
            self.assertEqual(rejected.exception.status, 403)
            with self.assertRaises(WSServerHandshakeError):
                await self.client.ws_connect('/ws/' + role,
                                             headers={'X-Teleop-Token': 'wrong'})
        ui = await self.client.ws_connect('/ws/ui')
        self.assertFalse((await self.next_type(ui, 'welcome'))['control'])
        await ui.send_json(packet(captured_ms=time.perf_counter() * 1000))
        self.assertIn('只读', (await self.next_type(ui, 'error'))['message'])
        await ui.send_json({'type': 'command', 'action': 'source', 'value': 'simulation'})
        self.assertIn('只读', (await self.next_type(ui, 'error'))['message'])
        self.assertEqual(self.app['preview'].control.state['source'], 'quest')
        self.assertIsNone(self.app['preview'].control.frame)
        keyboard = await self.client.ws_connect('/ws/keyboard',
                                                headers={'X-Teleop-Token': self.token})
        self.assertTrue((await self.next_type(keyboard, 'welcome'))['control'])
        bridge = await self.client.ws_connect('/ws/quest',
                                              headers={'X-Teleop-Token': self.token})
        self.assertTrue((await self.next_type(bridge, 'welcome'))['control'])
        await bridge.send_json({'type': 'command', 'action': 'source', 'value': 'simulation'})
        self.assertIn('位姿帧', (await self.next_type(bridge, 'error'))['message'])
        await bridge.close()
        await keyboard.close()
        await ui.close()

    async def test_native_state_reports_unknown_optical_validity(self):
        self.assertEqual((await (await self.client.get('/api/health')).json())['input_backend'],
                         'xrobotoolkit')
        self.assertEqual((await self.client.get('/quest')).status, 404)
        ui = await self.client.ws_connect('/ws/ui')
        await self.next_type(ui, 'welcome')
        state = await self.next_type(ui, 'state')
        self.assertEqual(state['input_backend'], 'xrobotoolkit')
        self.assertEqual(state['native_tracking_validity'], 'unreported')
        self.assertEqual(state['arms']['left']['optical_tracking_validity'], 'unreported')
        self.assertIsNone(state['arms']['left']['tracked'])
        self.status_file.write_text(json.dumps({
            'wall_ns': time.time_ns() - 10_000_000_000,
            'eligible': True, 'ws_connected': True, 'tracking_validity': 'reported',
        }), encoding='utf-8')
        health = await (await self.client.get('/api/health')).json()
        self.assertFalse(health['native_eligible'])
        self.assertEqual(health['native_tracking_validity'], 'unreported')
        await ui.close()

    async def test_native_keyboard_skips_usb_and_uses_token(self):
        console = MagicMock()
        console.__enter__.return_value = console
        console.poll.return_value = ['quit']
        args = SimpleNamespace(port=self.server.port, adb='adb', serial='',
                               config=self.config_file, hardware=False, full_arm=True,
                               robot_ip=None, vel_ratio=None, acc_ratio=None,
                               auto_calibrate=False, auto_follow_on_ready=False,
                               joint_gate='', native_arm_mode=True,
                               token_file=self.token_file)
        console_name = 'WindowsConsole' if sys.platform == 'win32' else 'UnixConsole'
        with patch(f'quest_teleop.keyboard.{console_name}', return_value=console), \
             patch('quest_teleop.keyboard.run_usb', new=AsyncMock(side_effect=AssertionError('USB called'))), \
             patch('quest_teleop.keyboard.create_app', side_effect=AssertionError('legacy app started')):
            await asyncio.wait_for(run_keyboard(args), 5)


class NativeReadinessTests(unittest.TestCase):
    def state(self, **changes):
        state = dict(input_backend='xrobotoolkit', native_eligible=True, hardware=False,
                     source='quest', keyboard_mode=True, fresh=True, hardware_connected=False,
                     hardware_enabled=False, hardware_fault=None, tracking_loss_active=False,
                     tracking_recovery_pending=False, input_focus_lost=False, frame_age_ms=25,
                     control_sides=['left', 'right'], input_session='native-1', input_reference=0,
                     seq=0, arms={'left': {'tracked': None, 'numeric_pose_gate': True},
                                  'right': {'tracked': None, 'numeric_pose_gate': True}})
        state.update(changes)
        return state

    def test_preview_accepts_numeric_gate_but_not_optical_claim(self):
        self.assertTrue(quest_tracking_ready(self.state(), native_arm_mode=True))
        self.assertFalse(quest_tracking_ready(self.state()))
        for change in ({'native_eligible': False}, {'fresh': False},
                       {'input_backend': 'webxr'}, {'input_focus_lost': True},
                       {'arms': {'left': {'tracked': None, 'numeric_pose_gate': False},
                                 'right': {'tracked': None, 'numeric_pose_gate': True}}}):
            self.assertFalse(quest_tracking_ready(self.state(**change), native_arm_mode=True))

    def test_native_first_follow_requires_two_seconds_and_distinct_frames(self):
        window = StableQuestTracking(stable_s=2.0, min_frames=30, native_arm_mode=True)
        for index in range(29):
            self.assertFalse(window.observe(self.state(seq=index), index * .07))
        self.assertTrue(window.observe(self.state(seq=29), 2.03))
        self.assertFalse(window.observe(self.state(seq=30, native_eligible=False), 2.10))


class NativeManualCalibrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_pauses_active_follow_before_reanchoring(self):
        class Socket:
            async def __aiter__(self):
                await asyncio.Future()
                yield None

        notes = []
        client = KeyboardClient(Socket(), report=notes.append, native_arm_mode=True)
        actions = []

        async def command(action):
            actions.append(action)
            return {'keyboard_follow': action == 'keyboard_toggle'}

        client.command = command
        try:
            await client.manual_calibrate()
            self.assertEqual(actions, ['pause', 'calibrate', 'keyboard_toggle'])
            self.assertIn('自动继续', notes[-1])
        finally:
            client.receiver.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await client.receiver

    async def test_failed_a_remains_paused(self):
        class Socket:
            async def __aiter__(self):
                await asyncio.Future()
                yield None

        client = KeyboardClient(Socket(), report=lambda _: None, native_arm_mode=True)
        actions = []

        async def command(action):
            actions.append(action)
            if action == 'calibrate':
                raise ValueError('位姿不新鲜')
            return {'keyboard_follow': action == 'keyboard_toggle'}

        client.command = command
        try:
            with self.assertRaisesRegex(ValueError, '位姿不新鲜'):
                await client.manual_calibrate()
            self.assertEqual(actions, ['pause', 'calibrate'])
        finally:
            client.receiver.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await client.receiver
