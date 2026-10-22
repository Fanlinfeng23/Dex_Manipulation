"""Spawn-process coverage of IPC isolation and dropped-frame safety."""
import asyncio
import copy
import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from quest_teleop.config import load_config
from quest_teleop.control_process import ControlProcess, _worker
from quest_teleop.protocol import parse_frame


class PollIntervalConfigTests(unittest.TestCase):
    def test_default_poll_interval_is_preserved_and_vive_can_opt_in(self):
        self.assertEqual(ControlProcess({}).ipc_poll_interval_s, .004)
        self.assertEqual(ControlProcess({'ipc_poll_interval_s': .001}).ipc_poll_interval_s,
                         .001)

    def test_poll_interval_rejects_busy_loop_and_invalid_values(self):
        for value in (0, .0001, .021, float('nan'), float('inf'), True, '1ms'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,
                                                                    'ipc_poll_interval_s'):
                ControlProcess({'ipc_poll_interval_s': value})


class ControlProcessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        config = load_config()
        config['visual_model'] = None
        self.runtime = ControlProcess(config)
        await self.runtime.start()
        self.seq = 0
        self.last_captured = 0.

    async def asyncTearDown(self):
        await self.runtime.close()

    def frame(self, grip=0., x=0.):
        now = time.perf_counter()
        self.last_captured = max(now, self.last_captured + .000001)
        self.seq += 1
        return parse_frame({
            'version': 1, 'type': 'frame', 'session': 'spawn-test', 'seq': self.seq,
            'reference': 0, 'captured_ms': self.last_captured * 1000, 'focused': True,
            'head': {'quaternion': [0, 0, 0, 1], 'tracked': True},
            'controllers': {side: {'position': [x, 1.2, -.4],
                'quaternion': [0, 0, 0, 1], 'tracked': True, 'grip': grip}
                for side in ('left', 'right')},
        }, now)

    async def wait_for(self, predicate, timeout=2.):
        deadline = time.perf_counter() + timeout
        while not predicate():
            if time.perf_counter() > deadline:
                self.fail(f'control state did not converge: {self.runtime.state}')
            await asyncio.sleep(.015)

    async def calibrate(self):
        frame = self.frame()
        self.assertTrue((await self.runtime.offer_frame(frame, 'simulation'))['accepted'])
        result = await self.runtime.command({'action': 'calibrate'}, frame=frame)
        self.assertTrue(result['ok'], result)
        self.assertTrue(self.runtime.state['enabled'])
        return frame

    async def consume(self, frame):
        result = await self.runtime.offer_frame(frame, 'simulation')
        self.assertTrue(result['accepted'], result)
        await self.wait_for(lambda: self.runtime.state['seq'] == frame.seq)
        return result

    async def test_worker_is_separate_and_snapshots_match_input(self):
        self.assertNotEqual(self.runtime.state['diagnostics']['worker_pid'], os.getpid())
        self.assertEqual(self.runtime.visual_data['kind'], 'schematic')
        frame = self.frame()
        result = await self.runtime.offer_frame(frame, 'simulation')
        self.assertTrue(result['accepted'], result)
        await self.wait_for(lambda: self.runtime.state['seq'] == frame.seq)
        self.assertEqual(self.runtime.frame['seq'], self.runtime.state['seq'])
        self.assertEqual(self.runtime.frame['seq'], frame.seq)
        self.assertIn('receive_to_consume_ms', self.runtime.state['diagnostics'])

    async def test_pose_burst_uses_latest_without_unbounded_queue(self):
        frames = [self.frame(x=i / 10000) for i in range(250)]
        results = await asyncio.gather(*(self.runtime.offer_frame(f, 'simulation') for f in frames))
        self.assertTrue(results[-1]['accepted'])
        self.assertTrue(all(r['reason'] == 'superseded' for r in results[:-1]))
        await self.wait_for(lambda: self.runtime.state['seq'] == frames[-1].seq)
        self.assertEqual(self.runtime.state['seq'], frames[-1].seq)
        self.assertGreaterEqual(self.runtime.state['diagnostics']['frames_coalesced'], 249)

    async def test_coalesced_release_reanchors_without_jump(self):
        await self.calibrate()
        await self.consume(self.frame(grip=1))
        await self.wait_for(lambda: self.runtime.state['following'])
        before = copy.deepcopy(self.runtime.state['arms'])
        released = self.runtime.offer_frame(self.frame(grip=0, x=.1), 'simulation')
        resumed_frame = self.frame(grip=1, x=.15)
        resumed = self.runtime.offer_frame(resumed_frame, 'simulation')
        self.assertEqual((await released)['reason'], 'superseded')
        self.assertTrue((await resumed)['accepted'])
        await self.wait_for(lambda: self.runtime.state['seq'] == resumed_frame.seq)
        self.assertTrue(self.runtime.state['following'])
        for side in ('left', 'right'):
            np.testing.assert_allclose(self.runtime.state['arms'][side]['joints_deg'],
                                       before[side]['joints_deg'], atol=1e-7)
            np.testing.assert_allclose(self.runtime.state['arms'][side]['target']['position'],
                                       self.runtime.state['arms'][side]['actual']['position'], atol=1e-7)

    async def test_coalesced_loss_cannot_be_hidden_by_recovery(self):
        for event in ('tracking', 'focus'):
            with self.subTest(event=event):
                await self.calibrate()
                lost = self.frame()
                if event == 'tracking':
                    lost.controllers['left'].tracked = False
                else:
                    lost.focused = False
                dropped = self.runtime.offer_frame(lost, 'simulation')
                recovered = self.frame()
                latest = self.runtime.offer_frame(recovered, 'simulation')
                await asyncio.gather(dropped, latest)
                await self.wait_for(lambda: self.runtime.state['seq'] == recovered.seq)
                self.assertFalse(self.runtime.state['enabled'])
                self.assertFalse(self.runtime.state['following'])

    async def test_command_barrier_and_same_sequence_calibration(self):
        first = self.frame()
        accepted = self.runtime.offer_frame(first, 'simulation')
        calibration = asyncio.create_task(self.runtime.command({'action': 'calibrate'}, frame=first))
        # Give the command a queue position before offering the next pose.
        await asyncio.sleep(0)
        next_frame = self.frame(grip=1)
        next_result = self.runtime.offer_frame(next_frame, 'simulation')
        self.assertTrue((await accepted)['accepted'])
        self.assertTrue((await calibration)['ok'])
        next_ack = await next_result
        self.assertTrue(next_ack['accepted'], next_ack)
        await self.wait_for(lambda: self.runtime.state['seq'] == next_frame.seq)
        await self.wait_for(lambda: self.runtime.state['following'])
        stale = await self.runtime.command({'action': 'calibrate'}, frame=first)
        self.assertFalse(stale['ok'])
        self.assertIn('替代', stale['error'])

    async def test_network_event_loop_stall_does_not_stall_watchdog(self):
        await self.calibrate()
        await self.consume(self.frame(grip=1))
        await self.wait_for(lambda: self.runtime.state['following'])
        # Deliberately block only the parent event loop. The child must detect
        # the input timeout while the network cannot run or send it a pause.
        time.sleep(self.runtime.config['input_timeout_s'] + .15)
        await self.wait_for(lambda: not self.runtime.state['enabled'])
        self.assertIn('超时', self.runtime.state['reason'])
        self.assertTrue(self.runtime.healthy)
        await self.runtime.offer_frame(self.frame(grip=1), 'simulation')
        self.assertFalse(self.runtime.state['enabled'])

    async def test_command_overflow_latches_fault_and_rejects_restart(self):
        await self.calibrate()
        results = await asyncio.gather(*(self.runtime.command({'action': 'scale', 'value': .75})
                                         for _ in range(80)))
        self.assertTrue(any(not r['ok'] for r in results))
        self.assertFalse(self.runtime.healthy)
        self.assertFalse(self.runtime.state['enabled'])
        self.assertIn('溢出', self.runtime.fault)
        self.assertFalse((await self.runtime.command({'action': 'reset'}))['ok'])
        with self.assertRaises(RuntimeError):
            await self.runtime.start()

    async def test_60_hz_ack_is_independent_of_25_hz_broadcast(self):
        latencies = []
        started = time.perf_counter()
        for index in range(60):
            await asyncio.sleep(max(0., started + index / 60 - time.perf_counter()))
            sent = time.perf_counter()
            result = await self.runtime.offer_frame(self.frame(), 'simulation')
            latencies.append(time.perf_counter() - sent)
            self.assertTrue(result['accepted'], result)
        # A broadcast-paced ACK would need >= 2.4 s for these 60 frames.
        self.assertLess(time.perf_counter() - started, 1.9)
        self.assertLess(float(np.median(latencies)), .025)
        self.assertEqual(self.runtime.config['broadcast_hz'], 25)

    async def test_platform_timer_resolution_is_reported_and_released(self):
        expected = 1 if sys.platform == 'win32' else 0
        self.assertEqual(self.runtime.timer_resolution.period_ms, expected)
        self.assertEqual(self.runtime.state['diagnostics']['timer_period_ms'], expected)
        await self.runtime.close()
        self.assertEqual(self.runtime.timer_resolution.period_ms, 0)

    async def test_modern_event_expiry_is_not_refreshed_by_context(self):
        first = self.frame()
        await self.consume(first)
        fresh = self.frame()
        command = {'action': 'calibrate', 'id': 'expired', 'session': fresh.session,
                   'captured_ms': (fresh.captured_s - 1.) * 1000}
        result = await self.runtime.command(command, frame=fresh)
        self.assertFalse(result['ok'])
        self.assertIn('过期', result['error'])
        self.assertFalse(self.runtime.state['enabled'])
        command.update(action='pause')
        self.assertTrue((await self.runtime.command(command))['ok'])

    async def test_desktop_quest_command_does_not_compare_browser_clocks(self):
        self.assertTrue((await self.runtime.command({'action': 'source', 'value': 'quest'}))['ok'])
        frame = self.frame()
        self.assertTrue((await self.runtime.offer_frame(frame, 'quest'))['accepted'])
        # A desktop event timestamp is intentionally unrelated to Quest time.
        result = await self.runtime.command({'action': 'calibrate', 'id': 'desktop-a',
                                            'session': 'desktop', 'captured_ms': 1})
        self.assertTrue(result['ok'], result)
        self.assertTrue(self.runtime.state['enabled'])
        result = await self.runtime.command({'action': 'reset', 'id': 'quest-b',
                'session': frame.session, 'captured_ms': frame.captured_s * 1000}, source='quest')
        self.assertFalse(result['ok'])
        self.assertIn('同帧', result['error'])

    async def test_same_sequence_command_cannot_hide_new_loss_latch(self):
        frame = await self.calibrate()
        lost = copy.deepcopy(frame)
        lost.safety = {'tracking_lost': True}
        result = await self.runtime.command({'action': 'calibrate'}, frame=lost)
        self.assertFalse(result['ok'])
        self.assertFalse(self.runtime.state['enabled'])
        # Explicit B retains the existing return policy for tracking loss.
        self.assertTrue((await self.runtime.command({'action': 'reset'}, frame=lost))['ok'])
        lost.safety = {'focus_lost': True}
        result = await self.runtime.command({'action': 'reset'}, frame=lost)
        self.assertFalse(result['ok'])
        self.assertFalse(self.runtime.state['resetting'])

    async def test_idle_frames_preserve_first_calibration_anchor(self):
        await self.calibrate()
        before = self.runtime.state['arms']['left']['joints_deg']
        await asyncio.gather(self.runtime.offer_frame(self.frame(), 'simulation'),
                             self.runtime.offer_frame(self.frame(), 'simulation'))
        await self.consume(self.frame(grip=1, x=.04))
        await self.wait_for(lambda: self.runtime.state['following'])
        self.assertGreater(np.max(np.abs(np.array(self.runtime.state['arms']['left']['joints_deg']) - before)), .01)

    async def test_worker_crash_latches_fault_and_never_resumes(self):
        await self.calibrate()
        self.runtime._process.terminate()
        await self.wait_for(lambda: self.runtime.fault is not None)
        self.assertFalse(self.runtime.state['enabled'])
        self.assertTrue(self.runtime.state['diagnostics']['worker_fault'])
        result = await self.runtime.offer_frame(self.frame(), 'simulation')
        self.assertFalse(result['accepted'])
        self.assertFalse((await self.runtime.command({'action': 'calibrate'}))['ok'])

    async def test_no_reply_timeout_latches_fault(self):
        exchange = self.runtime._exchange

        def delayed(items):
            time.sleep(.2)
            return exchange(items)

        self.runtime._exchange = delayed
        self.runtime.REPLY_TIMEOUT_S = .06
        await self.wait_for(lambda: self.runtime.fault is not None)
        self.assertFalse(self.runtime.state['enabled'])
        self.assertIn('TimeoutError', self.runtime.fault)
        self.assertFalse((await self.runtime.command({'action': 'reset'}))['ok'])


class StopHandshakeTests(unittest.IsolatedAsyncioTestCase):
    def test_worker_ack_follows_hardware_close(self):
        events = []

        class Engine:
            def __init__(self, config, kinematics, clock=None):
                self.kine = kinematics
                self.frame = None

            def attach_hardware(self, hardware, joints):
                pass

            def snapshot(self):
                return {}

            def pause(self, reason):
                events.append('pause')

        class Hardware:
            def __init__(self, config):
                pass

            def connect(self):
                return {}

            def close(self):
                events.append('close:start')
                time.sleep(2.2)  # Real standby confirmation may take 2–3 seconds.
                events.append('close:end')
                return True

        class Connection:
            def __init__(self):
                self.sent = []

            def send(self, value):
                self.sent.append(value)
                events.append('ack' if value.get('stopped') else 'ready')

            def poll(self, timeout):
                return True

            def recv(self):
                return {'stop': True}

            def close(self):
                pass

        class Timer:
            period_ms = 0
            def start(self): pass
            def close(self): pass

        connection = Connection()
        config = {'control_hz': 90, 'broadcast_hz': 20,
                  'hardware': {'enabled': True, 'startup_move_enabled': False},
                  'model': 'test'}
        with patch('quest_teleop.control_process.TeleopEngine', Engine), \
                patch('quest_teleop.control_process.Kinematics',
                      return_value=SimpleNamespace(visual=None)), \
                patch('quest_teleop.control_process.TianjiHardwareOutput', Hardware), \
                patch('quest_teleop.control_process.TimerResolution', Timer):
            _worker(connection, config)
        self.assertEqual(events.index('close:end') + 1, events.index('ack'))
        self.assertTrue(connection.sent[-1]['standby_confirmed'])

    async def test_parent_waits_for_slow_confirmed_standby_without_early_kill(self):
        class Process:
            alive = True
            terminated = False
            killed = False
            def is_alive(self): return self.alive
            def join(self, timeout): pass
            def terminate(self): self.terminated = True
            def kill(self): self.killed = True

        process = Process()
        runtime = ControlProcess({'hardware': {'enabled': True}})
        runtime._process = process
        runtime._connection = SimpleNamespace(close=lambda: None)
        runtime._stop_requested = True

        def delayed_stop():
            time.sleep(2.2)
            process.alive = False
            return {'stopped': True, 'standby_confirmed': True}

        runtime._exchange_stop = delayed_stop
        runtime._pump_task = asyncio.create_task(runtime._pump())
        reply = await runtime.close()
        self.assertTrue(reply['standby_confirmed'])
        self.assertFalse(process.terminated)
        self.assertFalse(process.killed)

    async def test_parent_rejects_unconfirmed_standby(self):
        class Process:
            alive = True
            def is_alive(self): return self.alive
            def join(self, timeout): pass

        process = Process()
        runtime = ControlProcess({'hardware': {'enabled': True}})
        runtime._process = process
        runtime._connection = SimpleNamespace(close=lambda: None)
        runtime._stop_requested = True
        def unconfirmed_stop():
            process.alive = False
            return {'stopped': True, 'standby_confirmed': False}
        runtime._exchange_stop = unconfirmed_stop
        runtime._pump_task = asyncio.create_task(runtime._pump())
        with self.assertRaisesRegex(RuntimeError, '未确认双臂待机'):
            await runtime.close()
        self.assertIn('未确认双臂待机', runtime.fault)
        self.assertFalse(runtime._stop_reply['standby_confirmed'])


if __name__ == '__main__':
    unittest.main()
