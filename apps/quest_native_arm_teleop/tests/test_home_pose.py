"""Default startup geometry and its B return, with the real vendor kinematics."""
import copy
import sys
import unittest

import numpy as np

from quest_teleop.config import SDK_ROOT, load_config
from quest_teleop.engine import TeleopEngine
from quest_teleop.kinematics import Kinematics
from quest_teleop.protocol import parse_frame
from test_core import packet


@unittest.skipUnless((SDK_ROOT / 'SDK_PYTHON' /
                      ('libKine.dll' if sys.platform == 'win32' else 'libKine.so')).exists(),
                     'Platform SDK required')
class HomePoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.kine = Kinematics(cls.config)

    def test_default_uses_98_mm_mount_spacing_and_straight_output_flanges(self):
        engine = TeleopEngine(self.config, self.kine)
        straight = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
        for side, sign in (('left', 1), ('right', -1)):
            with self.subTest(side=side):
                np.testing.assert_allclose(engine.actual[side][:3, 3], [.369, sign * .250, .868],
                                           rtol=0, atol=1e-7)
                np.testing.assert_allclose(engine.actual[side][:3, :3], straight, rtol=0, atol=1e-7)
                self.assertTrue(self.kine.valid_joints(side, engine.q[side]))
                limits = self.kine.arms[side]['limits']
                self.assertGreater(float(np.min(np.minimum(engine.q[side] - limits[:, 0],
                                                           limits[:, 1] - engine.q[side]))), 13.)
        self.assertAlmostEqual(float(self.kine.arms['left']['base'][1, 3] -
                                     self.kine.arms['right']['base'][1, 3]), .098, places=9)
        self.assertAlmostEqual(float(np.linalg.norm(engine.actual['left'][:3, 3] -
                                                    engine.actual['right'][:3, 3])), .5, places=6)

    def test_fifty_hz_input_follows_then_returns_to_default_smoothly(self):
        now = 10.
        engine = TeleopEngine(self.config, self.kine, clock=lambda: now)
        data = packet()
        next_input = now
        control_dt = 1 / self.config['control_hz']

        def tick(forward_m):
            nonlocal now, next_input
            now += control_dt
            if now >= next_input:
                next_input += .02
                data['seq'] += 1
                data['captured_ms'] = now * 1000
                for controller in data['controllers'].values():
                    controller['position'][2] = -.4 - forward_m / engine.scale
                self.assertTrue(engine.receive(parse_frame(copy.deepcopy(data), now), 'simulation'))
            engine.step()

        tick(0.)
        engine.calibrate()
        home = {side: engine.actual[side].copy() for side in ('left', 'right')}
        for controller in data['controllers'].values():
            controller['grip'] = 1.
        for index in range(2 * self.config['control_hz']):
            tick(.02 * min((index + 1) * control_dt, 1.))
            self.assertTrue(engine.enabled, engine.reason)
            for side in home:
                self.assertTrue(self.kine.valid_joints(side, engine.q[side]))
        for side in home:
            np.testing.assert_allclose(engine.actual[side][:3, 3],
                                       home[side][:3, 3] + [.02, 0., 0.], rtol=0, atol=.0005)
        before = {side: engine.q[side].copy() for side in home}
        engine.reset()
        self.assertIsNotNone(engine.reset_plan)
        for side in home:
            np.testing.assert_array_equal(engine.q[side], before[side])
        samples = {side: [before[side]] for side in home}
        for _ in range(8 * self.config['control_hz']):
            tick(.02)
            for side in home:
                self.assertTrue(self.kine.valid_joints(side, engine.q[side]))
                samples[side].append(engine.q[side].copy())
            if engine.reset_plan is None:
                break
        self.assertIsNone(engine.reset_plan)
        self.assertFalse(engine.enabled)
        for side in home:
            np.testing.assert_array_equal(engine.q[side], self.config['arms'][side]['initial_joints_deg'])
            np.testing.assert_allclose(engine.actual[side], home[side], rtol=0, atol=1e-10)
            velocity = np.diff(samples[side], axis=0) / control_dt
            acceleration = np.diff(np.vstack((np.zeros((1, 7)), velocity, np.zeros((1, 7)))), axis=0) / control_dt
            self.assertLessEqual(float(np.max(np.abs(velocity))), self.config['max_reset_speed_deg_s'] + 1e-6)
            self.assertLessEqual(float(np.max(np.abs(acceleration))), self.config['max_reset_accel_deg_s2'] + 1e-6)


if __name__ == '__main__':
    unittest.main()
