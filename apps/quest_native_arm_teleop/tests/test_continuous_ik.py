"""Regression tests use the real model, including its coupled wrist limits."""
import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

from quest_teleop.config import load_config, SDK_ROOT
from quest_teleop.engine import TeleopEngine
from quest_teleop.kinematics import Kinematics
from quest_teleop.math3d import rotation_angle
from quest_teleop.protocol import parse_frame
from test_core import packet


@unittest.skipUnless((SDK_ROOT/'SDK_PYTHON'/('libKine.dll' if sys.platform == 'win32' else 'libKine.so')).exists(),
                     'Platform SDK required')
class ContinuousIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.kine = Kinematics(cls.config)

    def test_world_jacobian_matches_fk_central_difference(self):
        from scipy.spatial.transform import Rotation
        for side in ('left', 'right'):
            home = np.array(self.config['arms'][side]['initial_joints_deg'], dtype=float)
            # FXMath.cpp's polynomial SIN wraps at +/-pi with a tiny residual.
            # COS calls SIN(pi/2-angle), so differencing FK across -90 degrees
            # amplifies that residual. Check home with independent MDH trig;
            # retain the SDK-FK derivative check away from quadrant boundaries.
            offset = np.array([.17, -.23, .31, -.19, .29, .13, -.37])
            references = (
                ('home_mdh', home, lambda s, q: self.kine.chain(s, q)[-1]),
                ('offset_sdk', home + offset, self.kine.fk),
            )
            for source, q, fk in references:
                with self.subTest(side=side, source=source):
                    analytical = self.kine.jacobian(side, q)
                    numerical = np.zeros((6, 7)); eps = 1e-5
                    for joint in range(7):
                        delta = np.zeros(7); delta[joint] = np.rad2deg(eps)
                        plus, minus = fk(side, q + delta), fk(side, q - delta)
                        for sample in (q + delta, q - delta):
                            np.testing.assert_allclose(self.kine.fk(side, sample),
                                                       self.kine.chain(side, sample)[-1],
                                                       atol=1e-8, rtol=0)
                        numerical[:3, joint] = (plus[:3, 3] - minus[:3, 3]) / (2 * eps)
                        numerical[3:, joint] = Rotation.from_matrix(plus[:3, :3] @ minus[:3, :3].T).as_rotvec() / (2 * eps)
                    np.testing.assert_allclose(analytical, numerical, atol=1e-7)

    def test_coupled_wrist_limits_are_enforced_beyond_axis_limits(self):
        for side in ('left', 'right'):
            q = np.array(self.config['arms'][side]['initial_joints_deg'])
            for wrist6 in (-59., 59.):
                q[5], q[6] = wrist6, 30.
                self.assertTrue(self.kine.valid_joints(side, q))
                for wrist7 in (-85., 85.):
                    q[6] = wrist7
                    self.assertFalse(self.kine.valid_joints(side, q))
        invalid = copy.deepcopy(self.config)
        invalid['arms']['left']['initial_joints_deg'][5:7] = [59., 85.]
        with self.assertRaisesRegex(ValueError, '初始姿态'):
            Kinematics(invalid)

    def test_both_arms_close_to_four_cm_without_relaxing_limits(self):
        now = 10.
        # This feasibility regression uses an explicit test pose. It must not
        # redefine the user's configured startup pose or B return destination.
        config = copy.deepcopy(self.config)
        closing = json.loads((Path(__file__).parent/'fixtures/closing_pose.json').read_text(encoding='utf-8'))
        for side in ('left', 'right'):
            config['arms'][side]['initial_joints_deg'] = closing[side]
            config['arms'][side]['base_position_m'] = closing['base_position_m'][side]
        kine = Kinematics(config)
        engine = TeleopEngine(config, kine, clock=lambda: now)
        data = packet()
        def feed():
            nonlocal now
            now += 1/90; data['seq'] += 1; data['captured_ms'] = now*1000
            engine.receive(parse_frame(copy.deepcopy(data), now), 'simulation'); engine.step()
        feed(); engine.calibrate()
        anchors = {s: engine.actual[s].copy() for s in ('left', 'right')}
        previous = {s: engine.q[s].copy() for s in ('left', 'right')}
        for frame in range(451):
            progress = min(frame / 360, 1.)
            for side, sign in (('left', 1), ('right', -1)):
                data['controllers'][side]['grip'] = 1
                data['controllers'][side]['position'][0] = sign*.28/engine.scale*progress
            feed()
            self.assertTrue(engine.enabled, engine.reason)
            for side in ('left', 'right'):
                self.assertTrue(kine.valid_joints(side, engine.q[side]))
                self.assertLessEqual(np.max(np.abs(engine.q[side]-previous[side])), 1.000001)
                previous[side] = engine.q[side].copy()
        for side, sign in (('left', 1), ('right', -1)):
            np.testing.assert_allclose(engine.actual[side][:3, 3], [.35, sign*.02, 1.], atol=.002)
            self.assertLess(rotation_angle(engine.actual[side][:3, :3], anchors[side][:3, :3]), .01)

    def test_bimanual_fixture_closes_with_straight_flanges_and_returns(self):
        now = 10.
        config = copy.deepcopy(self.config)
        home = json.loads((Path(__file__).parent/'fixtures/bimanual_task_pose.json').read_text(encoding='utf-8'))
        for side in ('left', 'right'):
            config['arms'][side]['initial_joints_deg'] = home[side]
            config['arms'][side]['base_position_m'] = home['base_position_m'][side]
        kine = Kinematics(config)
        engine = TeleopEngine(config, kine, clock=lambda: now)
        data = packet()
        straight = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
        for side, sign in (('left', 1), ('right', -1)):
            np.testing.assert_allclose(engine.actual[side][:3, 3], [.35, sign*.18, 1.], atol=1e-6)
            np.testing.assert_allclose(engine.actual[side][:3, :3], straight, atol=1e-6)
        def feed():
            nonlocal now
            now += 1/90; data['seq'] += 1; data['captured_ms'] = now*1000
            engine.receive(parse_frame(copy.deepcopy(data), now), 'simulation'); engine.step()
        feed(); engine.calibrate()
        for frame in range(541):
            for side, sign in (('left', 1), ('right', -1)):
                data['controllers'][side]['grip'] = 1
                data['controllers'][side]['position'][0] = sign*.16/engine.scale*min(frame/360, 1.)
            feed(); self.assertTrue(engine.enabled, engine.reason)
            for side in ('left', 'right'):
                q = engine.q[side]; limits = kine.arms[side]['limits']
                self.assertTrue(kine.valid_joints(side, q))
                self.assertGreater(float(np.min(np.minimum(q-limits[:, 0], limits[:, 1]-q))), 1.)
                self.assertLess(rotation_angle(engine.actual[side][:3, :3], straight), .002)
        for side, sign in (('left', 1), ('right', -1)):
            np.testing.assert_allclose(engine.actual[side][:3, 3], [.35, sign*.02, 1.], atol=.0005)
        engine.reset()
        for _ in range(600):
            feed()
            for side in ('left', 'right'): self.assertTrue(kine.valid_joints(side, engine.q[side]))
            if engine.reset_plan is None: break
        self.assertIsNone(engine.reset_plan); self.assertFalse(engine.enabled)
        for side in ('left', 'right'):
            np.testing.assert_array_equal(engine.q[side], config['arms'][side]['initial_joints_deg'])

    def test_unreachable_input_stays_bounded_and_can_return_without_calibration(self):
        now = 10.
        engine = TeleopEngine(self.config, self.kine, clock=lambda: now)
        data = packet()
        def feed():
            nonlocal now
            now += 1/90; data['seq'] += 1; data['captured_ms'] = now*1000
            engine.receive(parse_frame(copy.deepcopy(data), now), 'simulation'); engine.step()
        feed(); engine.calibrate(); anchor = engine.actual['left'].copy()
        was_limited = False
        for frame in range(811):
            progress = min(frame / 270, 1.) if frame < 360 else max(0., 1.-(frame-360)/270)
            data['controllers']['left']['grip'] = 1
            data['controllers']['left']['position'][2] = -.4-.35/engine.scale*progress
            feed()
            self.assertTrue(engine.enabled, engine.reason)
            self.assertTrue(self.kine.valid_joints('left', engine.q['left']))
            was_limited |= engine.ik_feedback['left']['limited']
        self.assertTrue(was_limited)
        np.testing.assert_allclose(engine.actual['left'][:3, 3], anchor[:3, 3], atol=.002)
        self.assertLess(rotation_angle(engine.actual['left'][:3, :3], anchor[:3, :3]), .01)
        self.assertEqual(engine.calibration_revision, 1)

    def test_quest_heading_drives_real_tcp_forward_right_and_up(self):
        from scipy.spatial.transform import Rotation
        for yaw in (-45, 70):
            facing = Rotation.from_euler('y', yaw, degrees=True)
            for direction, expected in (([0, 0, -1], [1, 0, 0]), ([1, 0, 0], [0, -1, 0]), ([0, 1, 0], [0, 0, 1])):
                with self.subTest(yaw=yaw, direction=direction):
                    now = 10.
                    engine = TeleopEngine(self.config, self.kine, clock=lambda: now)
                    engine.set_source('quest')
                    data = packet()
                    data['head']['quaternion'] = facing.as_quat().tolist()
                    def feed():
                        nonlocal now
                        now += 1 / 90; data['seq'] += 1; data['captured_ms'] = now * 1000
                        engine.receive(parse_frame(copy.deepcopy(data), now), 'quest'); engine.step()
                    feed(); engine.calibrate()
                    home = {s: engine.actual[s].copy() for s in ('left', 'right')}
                    for i in range(180):
                        displacement = .04 * min((i + 1) / 90, 1.) * facing.apply(direction)
                        for c in data['controllers'].values():
                            c['position'] = (np.array([0, 1.2, -.4]) + displacement).tolist()
                            c['grip'] = 1
                        feed()
                        self.assertTrue(engine.enabled, engine.reason)
                        for side in home: self.assertTrue(self.kine.valid_joints(side, engine.q[side]))
                    for side in home:
                        target = home[side][:3, 3] + .04 * engine.scale * np.array(expected)
                        np.testing.assert_allclose(engine.target[side][:3, 3], target, atol=1e-12)
                        np.testing.assert_allclose(engine.actual[side][:3, 3], target, atol=.0005)
                        self.assertLess(rotation_angle(engine.actual[side][:3, :3], home[side][:3, :3]), .002)


if __name__ == '__main__':
    unittest.main()
