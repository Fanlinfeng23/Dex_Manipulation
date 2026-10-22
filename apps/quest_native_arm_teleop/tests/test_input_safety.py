"""Safety edges must survive the 50 Hz pose stream and bounded IPC mailbox."""
import copy
import unittest

import numpy as np

from quest_teleop.config import load_config
from quest_teleop.engine import TeleopEngine
from quest_teleop.protocol import ProtocolError, parse_frame
from test_core import LinearKine, packet


class SafetyProtocolTests(unittest.TestCase):
    def test_optional_safety_defaults_and_release_deduplication(self):
        data = packet()
        self.assertEqual(parse_frame(data, 10.).safety,
                         {'released': [], 'focus_lost': False, 'tracking_lost': False})
        data['safety'] = {'released': ['left', 'left'], 'tracking_lost': True, 'focus_lost': True}
        self.assertEqual(parse_frame(data, 10.).safety,
                         {'released': ['left'], 'focus_lost': True, 'tracking_lost': True})

    def test_malformed_safety_flags_are_rejected(self):
        for safety in ([], None, {'released': 'left'}, {'released': ['head']},
                       {'released': ['left', 'right', 'left']}, {'released': [None]},
                       {'focus_lost': 1}, {'tracking_lost': 'false'}):
            with self.subTest(safety=safety), self.assertRaises(ProtocolError):
                data = packet()
                data['safety'] = safety
                parse_frame(data, 10.)


class InputSafetyTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.data = packet()
        self.engine = TeleopEngine(load_config(), LinearKine(), clock=lambda: self.now)
        self.engine.set_source('quest')
        self.feed()
        self.engine.calibrate()

    def new_frame(self, dt=.02):
        self.now += dt
        self.data['seq'] += 1
        self.data['captured_ms'] = self.now * 1000
        return parse_frame(copy.deepcopy(self.data), self.now)

    def feed(self, dt=.02):
        accepted = self.engine.receive(self.new_frame(dt), 'quest')
        self.engine.step()
        return accepted

    def follow_both(self):
        for controller in self.data['controllers'].values():
            controller['grip'] = 1.
        self.feed()
        for controller in self.data['controllers'].values():
            controller['position'][2] -= .04
        for _ in range(20):
            self.feed()
        self.assertTrue(self.engine.enabled, self.engine.reason)
        self.assertTrue(all(self.engine.active.values()))

    def test_omitted_trigger_release_reanchors_before_latest_gripped_pose(self):
        self.follow_both()
        held = self.engine.q['left'].copy()
        right_anchor = self.engine.anchors['right']
        # Between two transmitted poses the left trigger was released, the
        # controller moved beyond the jump threshold, then it was gripped again.
        # The latest analog value alone cannot reveal that clutch release.
        self.data['controllers']['left']['position'] = [.8, 1.2, -.5]
        self.data['safety'] = {'released': ['left']}
        self.assertTrue(self.engine.receive(self.new_frame(), 'quest'))
        self.assertFalse(self.engine.active['left'])
        self.assertNotIn('left', self.engine.anchors)
        self.assertNotIn('left', self.engine.last_controller)
        self.assertIs(self.engine.anchors['right'], right_anchor)
        self.assertTrue(self.engine.active['right'])
        self.engine.step()
        self.assertTrue(self.engine.enabled, self.engine.reason)
        np.testing.assert_allclose(self.engine.q['left'], held, rtol=0, atol=1e-9)
        np.testing.assert_allclose(self.engine.anchors['left'][0][:3, 3], [.8, 1.2, -.5])
        # Subsequent movement is relative to that new anchor, with no replay of
        # the large motion performed while the trigger was released.
        self.data.pop('safety')
        self.data['controllers']['left']['position'][2] -= .01
        self.feed()
        self.assertTrue(self.engine.enabled, self.engine.reason)
        self.assertGreater(self.engine.actual['left'][0, 3], held[0] / 1000)

    def test_latched_tracking_or_focus_loss_pauses_even_if_latest_pose_recovered(self):
        for flag in ('tracking_lost', 'focus_lost'):
            with self.subTest(flag=flag):
                self.setUp()
                self.follow_both()
                held = {side: q.copy() for side, q in self.engine.q.items()}
                self.data['safety'] = {flag: True}
                self.assertTrue(self.feed())
                self.assertFalse(self.engine.enabled)
                self.assertFalse(any(self.engine.active.values()))
                self.assertFalse(self.engine.anchors)
                self.assertTrue(self.engine.frame.focused)
                self.assertTrue(all(c.tracked for c in self.engine.frame.controllers.values()))
                self.data.pop('safety')
                for controller in self.data['controllers'].values():
                    controller['position'][2] -= .04
                for _ in range(10):
                    self.assertTrue(self.feed())
                    self.assertFalse(self.engine.enabled)
                    for side in held:
                        np.testing.assert_array_equal(self.engine.q[side], held[side])
                # Fresh poses and even a release/regrip must not silently rearm.
                for controller in self.data['controllers'].values():
                    controller['grip'] = 0.
                self.feed()
                self.assertFalse(self.engine.enabled)
                for controller in self.data['controllers'].values():
                    controller['grip'] = 1.
                self.feed()
                self.assertFalse(self.engine.enabled)
                for side in held:
                    np.testing.assert_array_equal(self.engine.q[side], held[side])
                for controller in self.data['controllers'].values():
                    controller['grip'] = 0.
                self.feed()
                self.engine.calibrate()
                for controller in self.data['controllers'].values():
                    controller['grip'] = 1.
                self.feed()
                self.assertTrue(self.engine.enabled, self.engine.reason)
                for side in held:
                    np.testing.assert_allclose(self.engine.q[side], held[side], rtol=0, atol=1e-9)

    def test_fresh_network_frame_expiring_before_consumption_is_rejected(self):
        self.follow_both()
        held = {side: q.copy() for side, q in self.engine.q.items()}
        last = self.engine.frame
        arrivals = tuple(self.engine.arrivals)
        late = self.new_frame()
        # The parent received a genuinely new frame; only its IPC delivery is
        # delayed. Using received_s alone would incorrectly declare it fresh.
        self.now += self.engine.config['frame_age_limit_s'] + .001
        self.assertFalse(self.engine.receive(late, 'quest'))
        self.assertIs(self.engine.frame, last)
        self.assertEqual(tuple(self.engine.arrivals), arrivals)
        self.assertGreater(self.engine.consume_delay_s, self.engine.config['frame_age_limit_s'])
        self.assertFalse(self.engine.enabled)
        with self.assertRaises(ValueError):
            self.engine.calibrate()
        self.engine.step()
        for _ in range(10):
            self.assertTrue(self.feed())
            self.assertFalse(self.engine.enabled)
            for side in held:
                np.testing.assert_array_equal(self.engine.q[side], held[side])

    def test_duplicate_and_reordered_frames_cannot_refresh_watchdog(self):
        for seq_offset in (0, -1):
            with self.subTest(seq_offset=seq_offset):
                self.setUp()
                self.follow_both()
                last = self.engine.frame
                arrivals = tuple(self.engine.arrivals)
                held = {side: q.copy() for side, q in self.engine.q.items()}
                duplicate = copy.deepcopy(self.data)
                duplicate['seq'] += seq_offset
                self.now += self.engine.config['input_timeout_s'] + .001
                duplicate['captured_ms'] = self.now * 1000
                self.assertFalse(self.engine.receive(parse_frame(duplicate, self.now), 'quest'))
                self.assertIs(self.engine.frame, last)
                self.assertEqual(tuple(self.engine.arrivals), arrivals)
                self.engine.step()
                self.assertFalse(self.engine.enabled)
                self.assertTrue(self.feed())
                self.assertFalse(self.engine.enabled)
                for side in held:
                    np.testing.assert_array_equal(self.engine.q[side], held[side])


if __name__ == '__main__':
    unittest.main()
