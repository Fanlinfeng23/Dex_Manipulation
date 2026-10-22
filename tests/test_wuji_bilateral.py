"""Offline bilateral Wuji identification, gating, and cleanup checks."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import wuji_joint_hand as bilateral
from wuji_sdk import DeviceType


def frame(seq: int):
    joint = SimpleNamespace(pose=SimpleNamespace(position=[0.0, 0.0, 0.0]))
    return SimpleNamespace(
        joints=[joint] * 21,
        header=SimpleNamespace(seq=seq, timestamp_us=seq * 8_333),
    )


class FakeHand:
    def __init__(self, side):
        self.side = side
        self.enable_calls = 0
        self.disable_calls = 0

    def handedness(self):
        return SimpleNamespace(get=lambda: self.side)

    def enable(self):
        self.enable_calls += 1

    def disable(self):
        self.disable_calls += 1


class FakeGlove:
    def __init__(self, side):
        self.side = side

    def hand_side(self):
        return SimpleNamespace(get=lambda: self.side)


class FakeLiveHand(FakeHand):
    def __init__(self, side):
        super().__init__(side)
        self.publisher = FakePublisher()
        self.publisher.close = lambda: None
        self.setup_calls = []
        self.publish_calls = 0

    def effort_limit(self):
        return SimpleNamespace(set=lambda value: self.setup_calls.append(('effort', value)))

    def mit_params(self):
        return SimpleNamespace(set=lambda value: self.setup_calls.append(('mit', value)))

    def joint_command(self):
        def publish():
            self.publish_calls += 1
            return self.publisher

        return SimpleNamespace(publish=publish)


class FakeLiveGlove(FakeGlove):
    def __init__(self, side, *, fail_set=False, get_override=None):
        super().__init__(side)
        self.fail_set = fail_set
        self.get_override = get_override
        self.model_path = ''
        self.model_set_calls = []

    def hand_model_path(self):
        def set_path(value):
            self.model_set_calls.append(value)
            if self.fail_set:
                raise RuntimeError('offline set failure')
            self.model_path = value

        def get_path():
            return self.model_path if self.get_override is None else self.get_override

        return SimpleNamespace(set=set_path, get=get_path)

    def hand_skeleton(self):
        return SimpleNamespace(subscribe=lambda: object())


class FakeManager:
    def __init__(self, spec=None):
        spec = spec or {
            'LH': ('hand', 'left'),
            'RH': ('hand', 'right'),
            'LG': ('glove', 'left'),
            'RG': ('glove', 'right'),
        }
        self.spec = spec
        self.handles = {
            sn: FakeHand(side) if kind == 'hand' else FakeGlove(side)
            for sn, (kind, side) in spec.items()
        }
        self.connect_calls = []
        self.disconnect_calls = 0
        self.current = {'user_id': '', 'display_name': 'Default', 'is_default': True}
        self.restored = False

    def scan(self):
        result = []
        for sn, (kind, _side) in self.spec.items():
            result.append(SimpleNamespace(
                sn=sn,
                device_type=DeviceType.WujiHand2 if kind == 'hand' else DeviceType.WujiGlove,
                address='offline',
            ))
        return result

    def connect(self, sn, device_name):
        self.connect_calls.append(sn)
        return self.handles[sn]

    def disconnect_all(self):
        self.disconnect_calls += 1

    def current_user(self):
        return self.current

    def switch_to_default_user(self):
        self.restored = True
        self.current = {'user_id': '', 'display_name': 'Default', 'is_default': True}
        return self.current


class SequencedScanManager(FakeManager):
    def __init__(self, scans, spec=None):
        super().__init__(spec)
        self.scans = scans
        self.scan_calls = 0

    def scan(self):
        selected = self.scans[min(self.scan_calls, len(self.scans) - 1)]
        self.scan_calls += 1
        return [item for item in super().scan() if item.sn in selected]


def requested(**values):
    result = {'left': {'hand': '', 'glove': ''}, 'right': {'hand': '', 'glove': ''}}
    for key, value in values.items():
        side, kind = key.split('_')
        result[side][kind] = value
    return result


class DiscoveryTests(unittest.TestCase):
    def run_fake_live(self, manager):
        named = {'user_id': 'u_both', 'display_name': 'both', 'is_default': False}
        argv = ['wuji_joint_hand.py', '--gate', '/tmp/offline-gate',
                '--model-user-id', 'u_both']
        stderr = io.StringIO()
        with (patch.object(bilateral.SdkManager, 'instance', return_value=manager),
              patch.object(bilateral.hand_profile, 'find_user', return_value=named),
              patch.object(bilateral.hand_profile, 'select_user', return_value=named),
              patch.object(bilateral.hand_profile, 'checked_model_paths',
                           return_value={'left': Path('/tmp/left.urdf'),
                                         'right': Path('/tmp/right.urdf')}),
              patch.object(bilateral.RetargetSession, 'for_hand', return_value=object()),
              patch.object(bilateral, 'run_joint_loop') as loop,
              patch.object(sys, 'argv', argv),
              contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(stderr)):
            result = bilateral.main()
        return result, loop, stderr.getvalue()

    def test_legacy_right_aliases_and_conflicts(self):
        args = SimpleNamespace(
            left_hand_sn='LH', right_hand_sn='', hand_sn='RH',
            left_glove_sn='LG', right_glove_sn='', glove_sn='RG',
        )
        self.assertEqual(bilateral.requested_serials(args)['right'],
                         {'hand': 'RH', 'glove': 'RG'})
        args.right_hand_sn = 'OTHER'
        with self.assertRaisesRegex(ValueError, '不同的右手序列号'):
            bilateral.requested_serials(args)

    def test_unpinned_devices_are_identified_by_read_only_side_query(self):
        manager = FakeManager()
        pairs = bilateral.discover_devices(manager, requested())
        self.assertEqual(bilateral.device_serials(pairs), {
            'left': {'hand_sn': 'LH', 'glove_sn': 'LG'},
            'right': {'hand_sn': 'RH', 'glove_sn': 'RG'},
        })
        self.assertEqual(set(manager.connect_calls), {'LH', 'RH', 'LG', 'RG'})
        self.assertEqual(manager.handles['LH'].enable_calls, 0)
        self.assertEqual(manager.handles['RH'].enable_calls, 0)

    def test_late_left_hand_is_combined_with_previously_verified_sides(self):
        manager = SequencedScanManager([('RH', 'LG', 'RG'), ('LH',)])
        waits = []
        pairs = bilateral.discover_devices(manager, requested(),
                                           sleep=waits.append)
        self.assertEqual(manager.scan_calls, 2)
        self.assertEqual(waits, [bilateral.DISCOVERY_RETRY_DELAY_S])
        self.assertEqual(bilateral.device_serials(pairs), {
            'left': {'hand_sn': 'LH', 'glove_sn': 'LG'},
            'right': {'hand_sn': 'RH', 'glove_sn': 'RG'},
        })
        self.assertCountEqual(manager.connect_calls, ['LH', 'RH', 'LG', 'RG'])
        self.assertEqual(manager.handles['LH'].enable_calls, 0)
        self.assertEqual(manager.handles['RH'].enable_calls, 0)

    def test_cached_hand_that_goes_offline_cannot_complete_four_device_gate(self):
        class VanishingHand(FakeHand):
            reads = 0

            def handedness(self):
                def get():
                    self.reads += 1
                    if self.reads > 1:
                        raise RuntimeError('read-only side query: offline')
                    return self.side

                return SimpleNamespace(get=get)

        manager = SequencedScanManager([('RH', 'LG', 'RG'), ('LH',)])
        manager.handles['RH'] = VanishingHand('right')
        with self.assertRaises(RuntimeError) as raised:
            bilateral.discover_devices(manager, requested(), sleep=lambda _: None)
        message = str(raised.exception)
        self.assertIn('未找到右侧 Wuji Hand2', message)
        self.assertIn('RH（read-only side query: offline）', message)
        self.assertEqual(manager.scan_calls, bilateral.DISCOVERY_SCAN_ATTEMPTS)
        self.assertEqual(manager.connect_calls.count('RH'), 1)

    def test_cached_hand_side_change_fails_before_devices_are_accepted(self):
        class ChangingHand(FakeHand):
            reads = 0

            def handedness(self):
                def get():
                    self.reads += 1
                    return 'right' if self.reads == 1 else 'left'

                return SimpleNamespace(get=get)

        manager = SequencedScanManager([('RH', 'LG', 'RG'), ('LH',)])
        manager.handles['RH'] = ChangingHand('right')
        with self.assertRaisesRegex(RuntimeError, 'SN=RH 侧别变化'):
            bilateral.discover_devices(manager, requested(), sleep=lambda _: None)
        self.assertEqual(manager.scan_calls, 2)

    def test_persistent_missing_left_reports_all_found_after_finite_scans(self):
        manager = SequencedScanManager([('RH', 'LG', 'RG')])
        waits = []
        with self.assertRaises(RuntimeError) as raised:
            bilateral.discover_devices(manager, requested(), sleep=waits.append)
        message = str(raised.exception)
        self.assertIn('扫描 3 次后设备未齐', message)
        self.assertIn('未找到左侧 Wuji Hand2', message)
        for item in ('右侧 Wuji Hand2=RH', '左侧 Wuji Glove=LG',
                     '右侧 Wuji Glove=RG'):
            self.assertIn(item, message)
        self.assertEqual(manager.scan_calls, bilateral.DISCOVERY_SCAN_ATTEMPTS)
        self.assertEqual(len(waits), bilateral.DISCOVERY_SCAN_ATTEMPTS - 1)
        self.assertCountEqual(manager.connect_calls, ['RH', 'LG', 'RG'])
        self.assertEqual(manager.handles['RH'].enable_calls, 0)

    def test_late_explicit_hand_wrong_side_still_fails(self):
        manager = SequencedScanManager(
            [('RH', 'LG', 'RG'), ('LH',)],
            {'LH': ('hand', 'right'), 'RH': ('hand', 'right'),
             'LG': ('glove', 'left'), 'RG': ('glove', 'right')},
        )
        with self.assertRaisesRegex(RuntimeError, 'SN=LH 实际为 right 侧'):
            bilateral.discover_devices(
                manager, requested(left_hand='LH', right_hand='RH'), sleep=lambda _: None)

    def test_late_same_side_duplicate_does_not_bypass_ambiguity_gate(self):
        manager = SequencedScanManager(
            [('LH', 'RH', 'LG'), ('LH2', 'RG')],
            {'LH': ('hand', 'left'), 'LH2': ('hand', 'left'),
             'RH': ('hand', 'right'), 'LG': ('glove', 'left'),
             'RG': ('glove', 'right')},
        )
        with self.assertRaisesRegex(RuntimeError, '多只左侧 Wuji Hand2'):
            bilateral.discover_devices(manager, requested(), sleep=lambda _: None)

    def test_transient_connect_failure_retries_without_reconnecting_other_sides(self):
        class ConnectOnceManager(SequencedScanManager):
            failed = False

            def connect(self, sn, device_name):
                if sn == 'LH' and not self.failed:
                    self.failed = True
                    self.connect_calls.append(sn)
                    raise RuntimeError('temporary offline')
                return super().connect(sn, device_name)

        manager = ConnectOnceManager([('LH', 'RH', 'LG', 'RG')])
        pairs = bilateral.discover_devices(manager, requested(), sleep=lambda _: None)
        self.assertEqual(pairs['left']['hand'].descriptor.sn, 'LH')
        self.assertEqual(manager.scan_calls, 2)
        self.assertEqual(manager.connect_calls.count('LH'), 2)
        for sn in ('RH', 'LG', 'RG'):
            self.assertEqual(manager.connect_calls.count(sn), 1)

    def test_duplicate_same_side_requires_explicit_serial(self):
        manager = FakeManager({'LH': ('hand', 'left'), 'LH2': ('hand', 'left'),
                               'RH': ('hand', 'right'), 'LG': ('glove', 'left'),
                               'RG': ('glove', 'right')})
        with self.assertRaisesRegex(RuntimeError, '多只左侧 Wuji Hand2'):
            bilateral.discover_devices(manager, requested())
        pairs = bilateral.discover_devices(manager, requested(left_hand='LH'))
        self.assertEqual(pairs['left']['hand'].descriptor.sn, 'LH')

    def test_explicit_serial_must_match_requested_side(self):
        manager = FakeManager()
        with self.assertRaisesRegex(RuntimeError, '实际为 left 侧'):
            bilateral.discover_devices(manager, requested(right_hand='LH'))

    def test_check_only_json_never_enables_or_commands(self):
        manager = FakeManager()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / 'devices.json'
            argv = ['wuji_joint_hand.py', '--check-only', '--device-output', str(output)]
            with (patch.object(bilateral.SdkManager, 'instance', return_value=manager),
                  patch.object(sys, 'argv', argv),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(bilateral.main(), 0)
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')),
                             {'left': {'hand_sn': 'LH', 'glove_sn': 'LG'},
                              'right': {'hand_sn': 'RH', 'glove_sn': 'RG'}})
        self.assertEqual(manager.disconnect_calls, 1)
        for sn in ('LH', 'RH'):
            self.assertEqual(manager.handles[sn].enable_calls, 0)
            self.assertEqual(manager.handles[sn].disable_calls, 0)

    def test_ambiguous_check_only_writes_no_json_and_disconnects(self):
        manager = FakeManager({'LH': ('hand', 'left'), 'LH2': ('hand', 'left'),
                               'RH': ('hand', 'right'), 'LG': ('glove', 'left'),
                               'RG': ('glove', 'right')})
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / 'devices.json'
            argv = ['wuji_joint_hand.py', '--check-only', '--device-output', str(output)]
            with (patch.object(bilateral.SdkManager, 'instance', return_value=manager),
                  patch.object(sys, 'argv', argv),
                  contextlib.redirect_stdout(io.StringIO()),
                  contextlib.redirect_stderr(io.StringIO())):
                self.assertEqual(bilateral.main(), 1)
            self.assertFalse(output.exists())
        self.assertEqual(manager.disconnect_calls, 1)
        self.assertEqual(manager.handles['LH'].disable_calls, 0)
        self.assertEqual(manager.handles['LH2'].disable_calls, 0)

    def test_side_error_disconnects_and_restores_user_before_any_enable(self):
        manager = FakeManager({'LH': ('hand', 'left'), 'RH': ('hand', 'left'),
                               'LG': ('glove', 'left'), 'RG': ('glove', 'right')})
        named = {'user_id': 'u_both', 'display_name': 'both', 'is_default': False}

        def select_user(_manager, _user):
            manager.current = named
            return named

        argv = ['wuji_joint_hand.py', '--gate', '/tmp/offline-gate',
                '--left-hand-sn', 'LH', '--right-hand-sn', 'RH',
                '--left-glove-sn', 'LG', '--right-glove-sn', 'RG',
                '--model-user-id', 'u_both']
        with (patch.object(bilateral.SdkManager, 'instance', return_value=manager),
              patch.object(bilateral.hand_profile, 'find_user', return_value=named),
              patch.object(bilateral.hand_profile, 'select_user', side_effect=select_user),
              patch.object(bilateral.hand_profile, 'checked_model_paths',
                           return_value={'left': Path('/tmp/left.urdf'),
                                         'right': Path('/tmp/right.urdf')}),
              patch.object(sys, 'argv', argv),
              contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(io.StringIO())):
            self.assertEqual(bilateral.main(), 1)
        self.assertEqual(manager.disconnect_calls, 1)
        self.assertTrue(manager.restored)
        self.assertEqual(manager.connect_calls, ['LH', 'RH', 'LG', 'RG'])
        self.assertEqual(manager.handles['LH'].enable_calls, 0)
        self.assertEqual(manager.handles['RH'].enable_calls, 0)
        self.assertEqual(manager.handles['LH'].disable_calls, 0)
        self.assertEqual(manager.handles['RH'].disable_calls, 0)

    def test_live_main_without_device_output_does_not_write_json(self):
        manager = FakeManager()
        for sn, (kind, side) in manager.spec.items():
            manager.handles[sn] = (FakeLiveHand(side) if kind == 'hand'
                                   else FakeLiveGlove(side))
        result, loop, errors = self.run_fake_live(manager)
        self.assertIsNone(result, errors)
        loop.assert_called_once()
        self.assertEqual(manager.handles['LG'].model_set_calls, ['/tmp/left.urdf'])
        self.assertEqual(manager.handles['RG'].model_set_calls, ['/tmp/right.urdf'])
        self.assertEqual(manager.disconnect_calls, 1)

    def test_wrong_model_path_aborts_before_hand_setup_or_motion(self):
        manager = FakeManager()
        for sn, (kind, side) in manager.spec.items():
            manager.handles[sn] = (FakeLiveHand(side) if kind == 'hand' else
                                   FakeLiveGlove(side, get_override='/tmp/stale.urdf')
                                   if side == 'right' else FakeLiveGlove(side))
        result, loop, errors = self.run_fake_live(manager)
        self.assertEqual(result, 1)
        self.assertIn('右侧 Wuji Glove 手型模型不匹配', errors)
        loop.assert_not_called()
        self.assertEqual(manager.handles['LH'].enable_calls, 0)
        self.assertEqual(manager.handles['RH'].enable_calls, 0)
        self.assertEqual(manager.handles['LH'].setup_calls, [])
        self.assertEqual(manager.handles['RH'].setup_calls, [])
        self.assertEqual(manager.handles['LH'].publish_calls, 0)
        self.assertEqual(manager.handles['RH'].publish_calls, 0)
        self.assertEqual(manager.handles['LH'].publisher.sent, [])
        self.assertEqual(manager.handles['RH'].publisher.sent, [])
        self.assertEqual(manager.disconnect_calls, 1)

    def test_model_set_failure_aborts_before_hand_setup_or_motion(self):
        manager = FakeManager()
        for sn, (kind, side) in manager.spec.items():
            manager.handles[sn] = (FakeLiveHand(side) if kind == 'hand' else
                                   FakeLiveGlove(side, fail_set=(side == 'left')))
        result, loop, errors = self.run_fake_live(manager)
        self.assertEqual(result, 1)
        self.assertIn('左侧 Wuji Glove 手型模型设置或回读失败', errors)
        loop.assert_not_called()
        self.assertEqual(manager.handles['LH'].enable_calls, 0)
        self.assertEqual(manager.handles['RH'].enable_calls, 0)
        self.assertEqual(manager.handles['LH'].setup_calls, [])
        self.assertEqual(manager.handles['RH'].setup_calls, [])
        self.assertEqual(manager.handles['LH'].publish_calls, 0)
        self.assertEqual(manager.handles['RH'].publish_calls, 0)
        self.assertEqual(manager.handles['LH'].publisher.sent, [])
        self.assertEqual(manager.handles['RH'].publisher.sent, [])
        self.assertEqual(manager.disconnect_calls, 1)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class TimedSubscription:
    def __init__(self, clock, first_tick=0, stop_tick=None):
        self.clock = clock
        self.first_tick = first_tick
        self.stop_tick = stop_tick
        self.last_read_tick = None

    def recv(self):
        tick = int(self.clock.now / (1 / bilateral.FPS) + 1e-6)
        if self.last_read_tick == tick:
            return None
        if tick < self.first_tick or (self.stop_tick is not None and tick >= self.stop_tick):
            return None
        self.last_read_tick = tick
        return frame(tick + 1)


class FakePublisher:
    def __init__(self):
        self.sent = []

    def send(self, commands):
        self.sent.append(commands)


class StreamTests(unittest.TestCase):
    def make_channels(self, clock, left_first=0, right_first=0, right_stop=None):
        channels = {}
        for side, first, stop in (('left', left_first, None),
                                  ('right', right_first, right_stop)):
            session = SimpleNamespace(step=lambda _kp: np.zeros(20, dtype=np.float32))
            channels[side] = bilateral.SideChannel(
                side=side,
                hand=FakeHand(side),
                publisher=FakePublisher(),
                subscription=TimedSubscription(clock, first_tick=first, stop_tick=stop),
                session=session,
            )
        return channels

    def test_ready_requires_both_new_streams_and_one_failure_stops_both(self):
        clock = FakeClock()
        channels = self.make_channels(clock, right_first=5, right_stop=15)
        ready_times = []
        ready = SimpleNamespace(write_text=lambda *_a, **_k: ready_times.append(clock.now))
        gate = SimpleNamespace(exists=lambda: True)
        enabled_hands = []
        with self.assertRaisesRegex(RuntimeError, '右侧 Wuji Glove 骨架流'):
            bilateral.run_joint_loop(channels, gate, ready, enabled_hands,
                                     clock=clock, sleep=clock.sleep)
        self.assertEqual(len(ready_times), 1)
        self.assertGreaterEqual(ready_times[0], 5 / bilateral.FPS)
        self.assertEqual(len(enabled_hands), 2)
        for side in ('left', 'right'):
            self.assertGreater(channels[side].hand.enable_calls, 0)
            self.assertGreater(len(channels[side].publisher.sent), 0)

    def test_no_right_stream_never_creates_ready_or_enables_hands(self):
        clock = FakeClock()
        channels = self.make_channels(clock, right_first=10_000)
        writes = []
        ready = SimpleNamespace(write_text=lambda *_a, **_k: writes.append(True))
        gate = SimpleNamespace(exists=lambda: True)
        enabled_hands = []
        with self.assertRaisesRegex(RuntimeError, '骨架流未就绪'):
            bilateral.run_joint_loop(channels, gate, ready, enabled_hands,
                                     clock=clock, sleep=clock.sleep)
        self.assertEqual(writes, [])
        self.assertEqual(enabled_hands, [])

    def test_rejoin_ramps_each_hand_from_its_own_last_command(self):
        clock = FakeClock()
        channels = self.make_channels(clock)
        for side, start, target in (('left', 0.0, 1.0), ('right', 1.0, 0.0)):
            channel = channels[side]
            channel.last_sent_qpos = np.full(20, start, dtype=np.float64)
            channel.current_keypoints = np.zeros((21, 3), dtype=np.float32)
            channel.resume_ramp_started = 0.0
            channel.session = SimpleNamespace(
                step=lambda _kp, target=target: np.full(20, target, dtype=np.float32))
            bilateral.send_channel_target(channel, 0.01)
            command = channel.publisher.sent[-1]
            self.assertEqual(len(command), 20)
            self.assertLessEqual(abs(command[0].position - start), 0.025000001)
        self.assertAlmostEqual(channels['left'].last_sent_qpos[0], 0.025)
        self.assertAlmostEqual(channels['right'].last_sent_qpos[0], 0.975)

    def test_invalid_right_target_sends_neither_side(self):
        clock = FakeClock()
        channels = self.make_channels(clock)
        channels['right'].session = SimpleNamespace(
            step=lambda _kp: np.full(20, np.nan, dtype=np.float32))
        gate = SimpleNamespace(exists=lambda: True)
        with self.assertRaisesRegex(RuntimeError, '右侧 Wuji 重定向输出'):
            bilateral.run_joint_loop(channels, gate, None, [],
                                     clock=clock, sleep=clock.sleep)
        self.assertEqual(len(channels['left'].publisher.sent), 0)
        self.assertEqual(len(channels['right'].publisher.sent), 0)

    def test_gate_heartbeat_rejects_stale_or_far_future_mtime(self):
        gate = SimpleNamespace(exists=lambda: True,
                               stat=lambda: SimpleNamespace(st_mtime=100.0))
        self.assertTrue(bilateral.gate_heartbeat_active(gate, 0, 200.0))
        self.assertTrue(bilateral.gate_heartbeat_active(gate, .5, 100.5))
        self.assertFalse(bilateral.gate_heartbeat_active(gate, .5, 100.501))
        self.assertFalse(bilateral.gate_heartbeat_active(gate, .5, 99.0))
        self.assertFalse(bilateral.gate_heartbeat_active(
            SimpleNamespace(exists=lambda: False), .5, 100.0))
        self.assertFalse(bilateral.gate_heartbeat_active(
            SimpleNamespace(exists=lambda: True, stat=Mock(side_effect=FileNotFoundError)),
            .5, 100.0))

    def test_expired_gate_holds_both_hands_until_fresh_touch_then_ramps(self):
        clock = FakeClock()
        channels = self.make_channels(clock)
        for side in ('left', 'right'):
            channels[side].session = SimpleNamespace(
                step=lambda _kp: np.full(20, 1.0 if clock.now >= .6 else 0.0,
                                         dtype=np.float32))
            sent = []
            channels[side].publisher = SimpleNamespace(
                sent=sent,
                send=lambda commands, sent=sent: sent.append((clock.now, commands[0].position)))
        gate = SimpleNamespace(
            exists=lambda: True,
            stat=lambda: SimpleNamespace(st_mtime=100.0 if clock.now < .8 else 100.8),
        )

        class StopLoop(Exception):
            pass

        def sleep(duration):
            clock.sleep(duration)
            if clock.now >= .95:
                raise StopLoop

        with self.assertRaises(StopLoop):
            bilateral.run_joint_loop(
                channels, gate, None, [], gate_timeout=.5,
                clock=clock, gate_clock=lambda: 100.0 + clock.now, sleep=sleep)
        for side in ('left', 'right'):
            sent = channels[side].publisher.sent
            self.assertTrue(any(t < .5 for t, _ in sent))
            self.assertFalse(any(.51 < t < .8 for t, _ in sent))
            resumed = [position for t, position in sent if t >= .8]
            self.assertTrue(resumed)
            self.assertLessEqual(resumed[0], bilateral.RESUME_MAX_JOINT_STEP_RAD + 1e-9)
            self.assertTrue(all(b - a <= bilateral.RESUME_MAX_JOINT_STEP_RAD + 1e-7
                                for a, b in zip(resumed, resumed[1:])), resumed)

    def test_gate_timeout_cli_rejects_negative_or_nonfinite_values(self):
        for value in ('-1', 'nan', 'inf'):
            with self.subTest(value=value), patch.object(sys, 'argv',
                    ['wuji_joint_hand.py', '--gate', '/tmp/offline-gate',
                     '--gate-timeout', value]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    bilateral.main()
                self.assertEqual(raised.exception.code, 2)

    def test_unbounded_queued_frames_faults_instead_of_starving_watchdog(self):
        subscription = SimpleNamespace(recv=lambda: frame(1))
        with self.assertRaisesRegex(RuntimeError, '骨架流积压过多'):
            bilateral.latest_keypoints(subscription)


if __name__ == '__main__':
    unittest.main()
