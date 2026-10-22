import copy
import asyncio
import contextlib
import ctypes
from ctypes import wintypes
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import WSMsgType

import numpy as np

from quest_teleop.config import load_config
from quest_teleop.engine import TeleopEngine
from quest_teleop.keyboard import (KeyEdges, KeyboardClient, StableQuestTracking,
                                   WindowsConsole, joint_gate_ready, main as keyboard_main)
from quest_teleop.protocol import parse_frame
from test_core import LinearKine, packet


class KeyboardEngineTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.data = packet()
        self.engine = TeleopEngine(load_config(), LinearKine(), clock=lambda: self.now)
        self.engine.attach_keyboard()
        self.feed()

    def feed(self, dt=.02, heartbeat=True):
        self.now += dt
        self.data['seq'] += 1
        self.data['captured_ms'] = self.now * 1000
        if heartbeat:
            self.engine.keyboard_heartbeat()
        self.engine.receive(parse_frame(copy.deepcopy(self.data), self.now), 'quest')
        self.engine.step()

    def follow(self):
        self.engine.calibrate()
        self.engine.toggle_keyboard_follow()
        self.feed()
        for c in self.data['controllers'].values():
            c['position'][2] -= .04
        for _ in range(15):
            self.feed()
        self.assertTrue(all(self.engine.active.values()))

    def test_space_requires_a_then_stops_holds_and_reanchors_both_arms(self):
        with self.assertRaises(ValueError):
            self.engine.toggle_keyboard_follow()
        start = {s: q.copy() for s, q in self.engine.q.items()}
        self.follow()
        for side in start:
            self.assertFalse(np.allclose(self.engine.q[side], start[side]))
        with self.assertRaises(ValueError):
            self.engine.calibrate()
        revision = self.engine.calibration_revision
        self.engine.toggle_keyboard_follow()
        held = {s: q.copy() for s, q in self.engine.q.items()}
        self.assertFalse(any(self.engine.active.values()))
        for c in self.data['controllers'].values():
            c['position'][0] += .5
            c['grip'] = 1.  # Physical trigger cannot restart a keyboard-stopped arm.
        self.feed()
        self.engine.toggle_keyboard_follow()
        self.feed()
        for side in held:
            np.testing.assert_allclose(self.engine.q[side], held[side], atol=1e-9)
        self.assertEqual(self.engine.calibration_revision, revision)
        self.assertTrue(all(self.engine.active.values()))

    def test_keyboard_overrides_raw_triggers_without_rewriting_pose_frames(self):
        for c in self.data['controllers'].values():
            c['grip'] = 1.
        self.feed()
        self.engine.calibrate()
        self.engine.toggle_keyboard_follow()
        self.feed()
        anchors = dict(self.engine.anchors)
        for c in self.data['controllers'].values():
            c['grip'] = 0.
        self.data['safety'] = {'released': ['left', 'right']}
        self.feed()
        for side in anchors:
            self.assertIs(self.engine.anchors[side], anchors[side])
            self.assertEqual(self.engine.frame.controllers[side].grip, 0.)
            self.assertEqual(self.engine.snapshot()['arms'][side]['grip'], 1.)

    def test_b_and_detach_clear_latch_and_require_new_calibration(self):
        self.follow()
        self.engine.reset()
        self.assertFalse(self.engine.keyboard_follow)
        self.assertFalse(self.engine.enabled)
        with self.assertRaises(ValueError):
            self.engine.toggle_keyboard_follow()
        self.engine.detach_keyboard()
        self.assertIsNone(self.engine.reset_plan)
        self.assertFalse(self.engine.keyboard_mode)
        self.assertFalse(any(self.engine.active.values()))

    def test_tracking_focus_reference_and_session_loss_do_not_resume(self):
        for loss in ('tracking', 'focus', 'reference', 'session'):
            with self.subTest(loss=loss):
                self.setUp()
                self.follow()
                if loss == 'tracking':
                    self.data['controllers']['left']['tracked'] = False
                elif loss == 'focus':
                    self.data['focused'] = False
                elif loss == 'reference':
                    self.data['reference'] += 1
                else:
                    self.data['session'] = 'replacement'
                self.feed()
                self.assertFalse(self.engine.keyboard_follow)
                self.assertFalse(self.engine.enabled)
                self.data['focused'] = True
                self.data['controllers']['left']['tracked'] = True
                self.feed()
                self.assertFalse(self.engine.enabled)
                with self.assertRaises(ValueError):
                    self.engine.toggle_keyboard_follow()

    def test_input_timeout_clears_latch_even_when_keyboard_is_alive(self):
        self.follow()
        self.now += self.engine.config['input_timeout_s'] + .01
        self.engine.keyboard_heartbeat()
        self.engine.step()
        self.assertFalse(self.engine.enabled)
        self.assertFalse(self.engine.keyboard_follow)
        self.feed()
        self.assertFalse(self.engine.enabled)

    def test_keyboard_timeout_stops_even_with_fresh_quest_frames(self):
        self.follow()
        for _ in range(55):
            self.feed(heartbeat=False)
        self.assertFalse(self.engine.keyboard_mode)
        self.assertFalse(self.engine.keyboard_follow)
        self.assertFalse(self.engine.enabled)
        with self.assertRaises(ValueError):
            self.engine.keyboard_heartbeat()


class KeyboardKeyTests(unittest.TestCase):
    def test_keys_case_independent_and_auto_repeat_ignored_until_release(self):
        edges = KeyEdges()
        # Virtual-key A/B/Q are identical with Shift/CapsLock on or off.
        for key, action in ((0x41, 'calibrate'), (0x42, 'reset'), (0x20, 'keyboard_toggle'),
                            (0x51, 'quit'), (0x1B, 'quit')):
            self.assertEqual(edges.update(key, True), action)
            for _ in range(20):
                self.assertIsNone(edges.update(key, True))
            self.assertIsNone(edges.update(key, False))
            self.assertEqual(edges.update(key, True), action)
            edges.update(key, False)
        self.assertIsNone(edges.update(0x26, True))  # Up-arrow is not a control.


class JointGateTests(unittest.IsolatedAsyncioTestCase):
    def state(self, **changes):
        state = {'type': 'state', 'reason': 'test', 'keyboard_mode': True,
                 'keyboard_follow': True, 'enabled': True, 'following': True,
                 'fresh': True, 'hardware_connected': True, 'hardware_enabled': True,
                 'hardware_fault': None, 'tracking_loss_active': False,
                 'tracking_recovery_pending': False,
                 'source': 'quest', 'input_session': 'test-session',
                 'input_reference': 1, 'input_focus_lost': False,
                 'seq': 0, 'frame_age_ms': 20., 'calibration_revision': 0,
                 'control_sides': ['left', 'right'],
                 'arms': {'left': {'tracked': True}, 'right': {'tracked': True}}}
        state.update(changes)
        return state

    def native_state(self, **changes):
        state = self.state(hardware=True, input_backend='xrobotoolkit',
                           native_eligible=True, native_reason='native_numeric_gate_ready',
                           arms={'left': {'tracked': None, 'numeric_pose_gate': True},
                                 'right': {'tracked': None, 'numeric_pose_gate': True}})
        state.update(changes)
        return state

    async def test_gate_closes_during_tracking_loss_and_recovery(self):
        states = [self.state(), self.state(tracking_loss_active=True),
                  self.state(tracking_recovery_pending=True, keyboard_follow=False,
                             enabled=False, following=False), self.state()]
        class Socket:
            async def __aiter__(self):
                for state in states:
                    yield SimpleNamespace(type=WSMsgType.TEXT, json=lambda state=state: state)
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / 'joint.follow'
            observed = []
            client = KeyboardClient(Socket(), report=lambda _: observed.append(gate.exists()),
                                    hardware=True, joint_gate=gate)
            await client.receiver
            self.assertEqual(observed, [True, False, False, True])
            self.assertFalse(gate.exists())


    async def test_joint_manual_a_calibrates_then_resumes_without_space(self):
        class Socket:
            async def __aiter__(self):
                await asyncio.Future()
                yield None
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / 'joint.follow'
            gate.touch()
            notes = []
            client = KeyboardClient(Socket(), report=notes.append, hardware=True,
                                    joint_gate=gate)
            async def command(action):
                self.assertFalse(gate.exists())
                return {'keyboard_follow': action == 'keyboard_toggle'}
            client.command = AsyncMock(side_effect=command)
            try:
                await client.manual_calibrate()
                self.assertEqual([call.args[0] for call in client.command.await_args_list],
                                 ['calibrate', 'keyboard_toggle'])
                self.assertIn('自动继续', notes[-1])
            finally:
                client.receiver.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await client.receiver

    def test_gate_requires_real_arm_follow_and_healthy_input(self):
        self.assertTrue(joint_gate_ready(self.state()))
        for changes in ({'fresh': False}, {'following': False}, {'hardware_enabled': False},
                        {'hardware_fault': 'failure'}, {'tracking_loss_active': True},
                        {'tracking_recovery_pending': True},
                        {'arms': {'left': {'tracked': False}, 'right': {'tracked': True}}}):
            with self.subTest(changes=changes):
                self.assertFalse(joint_gate_ready(self.state(**changes)))

    def test_native_gate_uses_numeric_validity_and_all_arm_guards(self):
        self.assertTrue(joint_gate_ready(self.native_state(), native_arm_mode=True))
        self.assertFalse(joint_gate_ready(self.native_state()))  # tracked=None is not optical proof.
        bad_states = (
            {'hardware': False}, {'input_backend': 'webxr'}, {'native_eligible': False},
            {'fresh': False}, {'frame_age_ms': 180}, {'input_focus_lost': True},
            {'keyboard_mode': False}, {'keyboard_follow': False}, {'enabled': False},
            {'following': False}, {'hardware_connected': False}, {'hardware_enabled': False},
            {'hardware_fault': 'servo fault'}, {'tracking_loss_active': True},
            {'tracking_recovery_pending': True},
            {'arms': {'left': {'tracked': None, 'numeric_pose_gate': False},
                      'right': {'tracked': None, 'numeric_pose_gate': True}}},
        )
        for change in bad_states:
            with self.subTest(change=change):
                self.assertFalse(joint_gate_ready(self.native_state(**change), native_arm_mode=True))

    async def test_native_joint_gate_heartbeats_and_closes_on_any_bad_state(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / 'joint.follow'
            timestamps = []
            healthy = self.native_state()
            bad = self.native_state(native_eligible=False)

            class Socket:
                async def __aiter__(self):
                    for state in (healthy, healthy, healthy):
                        yield SimpleNamespace(type=WSMsgType.TEXT, json=lambda state=state: state)
                        await asyncio.sleep(.02)
                        timestamps.append(gate.stat().st_mtime_ns)
                    yield SimpleNamespace(type=WSMsgType.TEXT, json=lambda: bad)
                    await asyncio.sleep(.02)
                    timestamps.append(None if not gate.exists() else gate.stat().st_mtime_ns)

            client = KeyboardClient(Socket(), report=lambda _: None, hardware=True,
                                    native_arm_mode=True, joint_gate=gate)
            client.first_started = True
            await client.receiver
            self.assertEqual(len(timestamps), 4)
            self.assertLess(timestamps[0], timestamps[1])
            self.assertLess(timestamps[1], timestamps[2])
            self.assertIsNone(timestamps[3])
            self.assertFalse(gate.exists())

    async def test_native_hardware_glove_gate_stays_closed_before_first_space(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / 'joint.follow'
            observations = []
            state = self.native_state()

            class Socket:
                async def __aiter__(self):
                    yield SimpleNamespace(type=WSMsgType.TEXT, json=lambda: state)
                    await asyncio.sleep(0)
                    observations.append(gate.exists())
                    client.first_started = True
                    yield SimpleNamespace(type=WSMsgType.TEXT, json=lambda: state)
                    await asyncio.sleep(0)
                    observations.append(gate.exists())

            client = KeyboardClient(Socket(), report=lambda _: None, hardware=True,
                                    native_arm_mode=True, joint_gate=gate)
            await client.receiver
            self.assertEqual(observations, [False, True])
            self.assertFalse(gate.exists())

    async def test_native_hardware_first_a_prepares_without_follow_then_later_a_resumes(self):
        class Socket:
            async def __aiter__(self):
                await asyncio.Future()
                yield None

        notes = []
        client = KeyboardClient(Socket(), report=notes.append, hardware=True,
                                native_arm_mode=True)
        actions = []

        async def command(action):
            actions.append(action)
            return {'keyboard_follow': action == 'keyboard_toggle'}

        client.command = AsyncMock(side_effect=command)
        try:
            await client.manual_calibrate()
            self.assertEqual(actions, ['pause', 'calibrate'])
            self.assertTrue(client.initial_prepared)
            self.assertFalse(client.first_started)
            self.assertIn('按空格首次开始', notes[-1])
            client.first_started = True
            await client.manual_calibrate()
            self.assertEqual(actions[-3:], ['pause', 'calibrate', 'keyboard_toggle'])
            self.assertIn('自动继续', notes[-1])
        finally:
            client.receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await client.receiver

    async def test_native_hardware_wait_accepts_space_and_quit_without_deferred_start(self):
        class Socket:
            closed = False

            async def __aiter__(self):
                await asyncio.Future()
                yield None

            async def close(self):
                self.closed = True

        class Console:
            calls = 0

            def poll(self):
                self.calls += 1
                return ['keyboard_toggle'] if self.calls == 1 else ['quit']

        client = KeyboardClient(Socket(), report=lambda _: None, hardware=True,
                                native_arm_mode=True, auto_calibrate=True,
                                auto_follow_on_ready=True)
        actions = []

        async def command(action):
            actions.append(action)
            return {'keyboard_follow': action == 'keyboard_toggle'}

        async def waiting_for_pose(**_kwargs):
            await asyncio.Future()

        client.command = AsyncMock(side_effect=command)
        client.calibrate_when_ready = AsyncMock(side_effect=waiting_for_pose)
        client.auto_follow_once = AsyncMock(side_effect=AssertionError('must not auto follow'))
        await asyncio.wait_for(client.run(Console()), .5)
        self.assertEqual(actions, ['keyboard_attach', 'keyboard_detach'])
        client.calibrate_when_ready.assert_awaited_with(timeout_s=None)
        client.auto_follow_once.assert_not_awaited()
        self.assertFalse(client.first_started)

    async def test_native_hardware_auto_calibration_succeeds_but_waits_for_space(self):
        class Socket:
            closed = False

            async def __aiter__(self):
                await asyncio.Future()
                yield None

            async def close(self):
                self.closed = True

        class Console:
            calls = 0

            def poll(self):
                self.calls += 1
                return [] if self.calls == 1 else ['quit']

        client = KeyboardClient(Socket(), report=lambda _: None, hardware=True,
                                native_arm_mode=True, auto_calibrate=True,
                                auto_follow_on_ready=True)
        actions = []

        async def command(action):
            actions.append(action)
            return {'keyboard_follow': action == 'keyboard_toggle'}

        client.command = AsyncMock(side_effect=command)
        await asyncio.wait_for(client.run(Console()), .5)
        self.assertEqual(actions, ['keyboard_attach', 'calibrate', 'keyboard_detach'])
        self.assertTrue(client.initial_prepared)
        self.assertFalse(client.first_started)

    async def test_native_hardware_auto_calibration_waits_without_deadline(self):
        class Socket:
            async def __aiter__(self):
                await asyncio.Future()
                yield None

        client = KeyboardClient(Socket(), report=lambda _: None, hardware=True,
                                native_arm_mode=True)
        client.command = AsyncMock(side_effect=ValueError('没有新鲜的手柄数据'))
        task = asyncio.create_task(client.calibrate_when_ready(timeout_s=None))
        try:
            await asyncio.sleep(.12)
            self.assertFalse(task.done())
            self.assertGreaterEqual(client.command.await_count, 2)
        finally:
            task.cancel()
            client.receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            with contextlib.suppress(asyncio.CancelledError):
                await client.receiver

    async def test_native_hardware_auto_calibration_stops_on_reported_fault(self):
        class Socket:
            async def __aiter__(self):
                await asyncio.Future()
                yield None

        client = KeyboardClient(Socket(), report=lambda _: None, hardware=True,
                                native_arm_mode=True)
        client.latest_state = {'hardware_fault': 'servo fault'}
        client.command = AsyncMock()
        try:
            with self.assertRaisesRegex(RuntimeError, '实机故障已锁存'):
                await client.calibrate_when_ready(timeout_s=None)
            client.command.assert_not_awaited()
        finally:
            client.receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await client.receiver

    async def test_native_hardware_first_space_requires_current_ready_state(self):
        class Socket:
            closed = False

            async def __aiter__(self):
                await asyncio.Future()
                yield None

            async def close(self):
                self.closed = True

        client = KeyboardClient(Socket(), report=lambda _: None, hardware=True,
                                native_arm_mode=True, clock=lambda: 10.0)
        client.initial_prepared = True
        client.latest_state = self.native_state(keyboard_follow=False, following=False,
                                                native_eligible=False)
        client.latest_state_at = time.monotonic()
        actions = []

        async def command(action):
            actions.append(action)
            return {'keyboard_follow': action == 'keyboard_toggle'}

        client.command = AsyncMock(side_effect=command)

        class Console:
            calls = 0

            def poll(self):
                self.calls += 1
                if self.calls == 1:
                    return ['keyboard_toggle']
                if self.calls == 2:
                    client.latest_state = self_state
                    return []
                return ['quit']

        self_state = self.native_state(keyboard_follow=False, following=False)
        await asyncio.wait_for(client.run(Console()), .5)
        self.assertEqual(actions, ['keyboard_attach', 'keyboard_detach'])
        self.assertFalse(client.first_started)
        self.assertTrue(client.first_space_ready())

    async def test_native_hardware_ready_space_starts_exactly_once(self):
        class Socket:
            closed = False

            async def __aiter__(self):
                await asyncio.Future()
                yield None

            async def close(self):
                self.closed = True

        client = KeyboardClient(Socket(), report=lambda _: None, hardware=True,
                                native_arm_mode=True, clock=lambda: 10.0)
        client.initial_prepared = True
        client.latest_state = self.native_state(keyboard_follow=False, following=False)
        client.latest_state_at = time.monotonic()
        actions = []

        async def command(action):
            actions.append(action)
            return {'keyboard_follow': action == 'keyboard_toggle'}

        client.command = AsyncMock(side_effect=command)

        class Console:
            calls = 0

            def poll(self):
                self.calls += 1
                return ['keyboard_toggle'] if self.calls == 1 else ['quit']

        await asyncio.wait_for(client.run(Console()), .5)
        self.assertEqual(actions, ['keyboard_attach', 'keyboard_toggle', 'keyboard_detach'])
        self.assertTrue(client.first_started)

    def test_first_start_waits_for_distinct_frames_in_one_reference_space(self):
        window = StableQuestTracking()
        for seq in range(8):
            self.assertFalse(window.observe(self.state(seq=seq), seq * .04))
        self.assertFalse(window.observe(self.state(seq=8, input_reference=2), .32))
        for seq in range(9, 23):
            self.assertFalse(window.observe(self.state(seq=seq, input_reference=2), seq * .04))
        self.assertTrue(window.observe(self.state(seq=23, input_reference=2), .92))
        self.assertFalse(window.observe(self.state(seq=25, fresh=False, input_reference=2), 1.))
        self.assertFalse(window.observe(self.state(seq=26, input_reference=2), 1.04))

    async def test_first_auto_follow_calibrates_and_starts_once(self):
        states = [self.state(seq=i, keyboard_follow=False, following=False, enabled=False)
                  for i in range(18)]
        states += [self.state(seq=18, keyboard_follow=False, following=False,
                              calibration_revision=1)]
        class Socket:
            async def __aiter__(self):
                for state in states:
                    yield SimpleNamespace(type=WSMsgType.TEXT, json=lambda state=state: state)
                    await asyncio.sleep(0)
                await asyncio.Future()
        observations = []
        clock_ticks = iter(i * .04 for i in range(100))
        client = KeyboardClient(Socket(), report=observations.append, hardware=True,
                                clock=lambda: next(clock_ticks))
        actions = []
        async def command(action):
            actions.append(action)
            return {'ok': True, 'keyboard_follow': action == 'keyboard_toggle'}
        client.command = AsyncMock(side_effect=command)
        heartbeat = asyncio.create_task(asyncio.Event().wait())
        try:
            await asyncio.wait_for(client.auto_follow_once(heartbeat), 1.)
            self.assertEqual(actions, ['calibrate', 'keyboard_toggle'])
            self.assertTrue(any('已自动校准并开始联合跟随' in note for note in observations))
        finally:
            heartbeat.cancel(); client.receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            with contextlib.suppress(asyncio.CancelledError):
                await client.receiver

    async def test_reference_reset_during_auto_calibration_blocks_start(self):
        states = [self.state(seq=i, keyboard_follow=False, following=False, enabled=False)
                  for i in range(18)]
        states += [self.state(seq=18, input_reference=2, keyboard_follow=False,
                              following=False, calibration_revision=1)]
        class Socket:
            async def __aiter__(self):
                for state in states:
                    yield SimpleNamespace(type=WSMsgType.TEXT, json=lambda state=state: state)
                    await asyncio.sleep(0)
                await asyncio.Future()
        clock_ticks = iter(i * .04 for i in range(100))
        client = KeyboardClient(Socket(), report=lambda _: None, hardware=True,
                                clock=lambda: next(clock_ticks))
        actions = []
        async def command(action):
            actions.append(action)
            return {'ok': True, 'keyboard_follow': action == 'keyboard_toggle'}
        client.command = AsyncMock(side_effect=command)
        heartbeat = asyncio.create_task(asyncio.Event().wait())
        try:
            with self.assertRaisesRegex(RuntimeError, '追踪空间'):
                await asyncio.wait_for(client.auto_follow_once(heartbeat), 1.)
            self.assertEqual(actions, ['calibrate'])
        finally:
            heartbeat.cancel(); client.receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            with contextlib.suppress(asyncio.CancelledError):
                await client.receiver


class NativeKeyboardCliTests(unittest.TestCase):
    def test_native_joint_accepts_hardware_full_arm_and_gate(self):
        argv = ['keyboard', '--native-arm-mode', '--full-arm', '--hardware',
                '--joint-gate', '/tmp/native-joint.follow', '--token-file', '/tmp/native.token']
        with patch.object(sys, 'argv', argv), patch('quest_teleop.keyboard.asyncio.run') as launch:
            self.assertEqual(keyboard_main(), 0)
            coroutine = launch.call_args.args[0]
            self.assertTrue(coroutine.cr_frame.f_locals['args'].native_arm_mode)
            self.assertEqual(coroutine.cr_frame.f_locals['args'].joint_gate,
                             '/tmp/native-joint.follow')
            coroutine.close()

    def test_native_joint_rejects_preview_gate(self):
        argv = ['keyboard', '--native-arm-mode', '--full-arm', '--joint-gate',
                '/tmp/native-joint.follow', '--token-file', '/tmp/native.token']
        with patch.object(sys, 'argv', argv), patch.object(sys, 'stderr', io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                keyboard_main()
        self.assertEqual(error.exception.code, 2)


@unittest.skipUnless(sys.platform == 'win32' and sys.stdin.isatty(), 'Interactive Windows console required')
class WindowsConsoleTests(unittest.TestCase):
    def test_native_key_events_and_console_mode_restored(self):
        console = WindowsConsole()
        record_type = type(console.records)._type_
        write = console.api.WriteConsoleInputW
        write.argtypes = [wintypes.HANDLE, ctypes.POINTER(record_type), wintypes.DWORD,
                          ctypes.POINTER(wintypes.DWORD)]
        events = []
        for vk, char in ((0x41, 'a'), (0x41, 'A'), (0x42, 'b'), (0x42, 'B'),
                         (0x20, ' '), (0x51, 'q'), (0x51, 'Q'), (0x1B, '\x1b')):
            for down in (True, True, True, False):
                event = record_type()
                event.kind = 1
                event.event.key.down = down
                event.event.key.key = vk
                event.event.key.char = char
                event.event.key.repeat = 1
                events.append(event)
        with console:
            records = (record_type * len(events))(*events)
            written = wintypes.DWORD()
            self.assertTrue(write(console.handle, records, len(events), ctypes.byref(written)))
            self.assertEqual(written.value, len(events))
            self.assertEqual(console.poll(), ['calibrate', 'calibrate', 'reset', 'reset',
                                              'keyboard_toggle', 'quit', 'quit', 'quit'])
        restored = wintypes.DWORD()
        self.assertTrue(console.api.GetConsoleMode(console.handle, ctypes.byref(restored)))
        self.assertEqual(restored.value, console.mode.value)


if __name__ == '__main__':
    unittest.main()
