import copy
import unittest

import numpy as np

from quest_teleop.config import load_config
from quest_teleop.engine import TeleopEngine
from quest_teleop.math3d import XR_TO_WORLD
from quest_teleop.one_euro import OneEuroQuaternion, OneEuroVector
from quest_teleop.protocol import parse_frame
from test_core import LinearKine, packet


class OneEuroTests(unittest.TestCase):
    def test_stationary_noise_is_suppressed_and_fast_motion_remains_responsive(self):
        filt = OneEuroVector(1., .7 * 20., 1.)
        rng = np.random.default_rng(7)
        raw, filtered = [], []
        for _ in range(180):
            sample = rng.normal(0., .001, 3)
            raw.append(sample)
            filtered.append(filt(sample, 1 / 90))
        self.assertLess(np.std(filtered[-90:]), np.std(raw[-90:]) * .45)
        response = filt(np.array([.12, 0., 0.]), 1 / 90)
        self.assertGreater(response[0], .015)

    def test_quaternion_filter_handles_double_cover_and_stays_normalized(self):
        filt = OneEuroQuaternion(1., .7 * 4., 1.)
        first = filt([0., 0., 0., 1.], 1 / 90)
        second = filt([0., 0., 0., -1.], 1 / 90)
        np.testing.assert_allclose(second, first)
        self.assertAlmostEqual(np.linalg.norm(second), 1.)


class NeckModeTests(unittest.TestCase):
    def test_fixed_heading_calibrates_without_valid_head_pose(self):
        config = load_config('configs/hardware_dual_arm.json')
        config['hardware']['enabled'] = False
        now = 10.
        engine = TeleopEngine(config, LinearKine(), clock=lambda: now)
        engine.set_source('quest')
        data = packet()
        data['head'] = {'quaternion': [0., 0., 0., 1.], 'tracked': False}
        frame = parse_frame(copy.deepcopy(data), now)
        engine.receive(frame, 'quest')
        engine.calibrate()
        self.assertTrue(engine.enabled)
        self.assertEqual(engine.snapshot()['heading_source'], 'fixed')
        np.testing.assert_allclose(engine.basis, XR_TO_WORLD)

    def test_protocol_exposes_physical_a_b_state_to_desktop(self):
        data = packet()
        data['buttons'] = {'calibrate': True, 'reset': False}
        frame = parse_frame(data, 10.)
        self.assertEqual(frame.buttons, {'calibrate': True, 'reset': False})


if __name__ == '__main__':
    unittest.main()
