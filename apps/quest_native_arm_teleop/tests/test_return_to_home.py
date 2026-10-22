"""Offline safety planning for small limit overruns; never connects to hardware."""
import copy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from quest_teleop.config import SDK_ROOT, load_config
from quest_teleop.engine import TeleopEngine
from quest_teleop.kinematics import Kinematics
from tools import return_to_home
from tools.return_to_home import limit_diagnostics, plan, rescue_waypoint


@unittest.skipUnless((SDK_ROOT / 'SDK_PYTHON' /
                      ('libKine.dll' if sys.platform == 'win32' else 'libKine.so')).exists(),
                     'Platform SDK required')
class ReturnToHomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(Path(__file__).resolve().parents[1] / 'configs/hardware_dual_arm.json', hardware=True)
        cls.kine = Kinematics(cls.config)
        cls.report = {'ready_for_wrist_test': True, 'read_only': True,
                      'motion_commands_sent': False, 'model_verified': cls.config['model'],
                      'robot_ip': '192.168.1.190', 'checked_at': '2026-09-23T15:19:39+08:00',
                      'arms': {
                          'left': {'cur_state': 0, 'err_code': 0, 'telemetry_valid': True,
                                   'servo_error_words': [0] * 7,
                                   'joints_deg': [138.5872, -30.4448, -96.2299, -116.3341,
                                                  -52.598, -23.964, 77.7169]},
                          'right': {'cur_state': 0, 'err_code': 0, 'telemetry_valid': True,
                                    'servo_error_words': [0] * 7,
                                    'joints_deg': [-81.2664, -44.0527, -9.3041, -120.9634,
                                                   -9.1783, 60.1371, -37.2334]}}}

    def test_actual_right_wrist_overrun_gets_right_only_rescue_then_valid_dual_path(self):
        result = plan(self.report, self.config, self.kine)
        self.assertEqual(result['rescue_side'], 'right')
        self.assertEqual(result['phases'][0]['control_sides'], ['right'])
        self.assertEqual(result['phases'][0]['joint'], 6)
        self.assertEqual(result['phases'][1]['control_sides'], ['left', 'right'])
        first_goal = np.asarray(result['phases'][0]['goal']['right'])
        self.assertLess(first_goal[5], 58.)
        self.assertTrue(self.kine.valid_joints('right', first_goal))
        self.assertGreater(result['estimated_motion_s'], 40.)
        self.assertLess(result['estimated_motion_s'], 50.)

    def test_larger_overrun_is_rejected(self):
        report = copy.deepcopy(self.report)
        report['arms']['right']['joints_deg'][5] = 61.
        with self.assertRaisesRegex(ValueError, '轻微越限'):
            plan(report, self.config, self.kine)

    def test_observed_left_wrist_is_outside_coupled_limit_despite_valid_individual_axes(self):
        q = [25.8271, -32.4724, -12.0156, -102.6115, 30.4529, 57.7093, -49.1156]
        details = limit_diagnostics(self.kine, 'left', q)
        self.assertEqual(details['joint_violations'], [])
        self.assertFalse(details['within_model_limits'])
        self.assertAlmostEqual(details['max_wrist_overrun_deg'], 11.1817917)
        lower = next(v for v in details['wrist_constraints'] if v['bound'] == 'lower')
        self.assertAlmostEqual(lower['joint7_bound_deg'], -37.9338083)
        with self.assertRaisesRegex(ValueError, 'J6=57.709.*11.182.*腕部耦合 3°'):
            rescue_waypoint(self.kine, 'left', np.asarray(q))

    def test_cli_rejected_plan_is_structured_and_never_reaches_hardware(self):
        report = copy.deepcopy(self.report)
        report['arms']['left']['joints_deg'] = [25.8271, -32.4724, -12.0156,
                                               -102.6115, 30.4529, 57.7093, -49.1156]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'feedback.json'
            path.write_text(json.dumps(report))
            output = Path(directory) / 'execution.json'
            config = Path(__file__).resolve().parents[1] / 'configs/hardware_dual_arm.json'
            for execute in (False, True):
                with self.subTest(execute=execute):
                    args = ['--report', str(path), '--config', str(config)]
                    if execute:
                        args += ['--execute', '--confirmed', '--output', str(output)]
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                            patch.object(return_to_home, 'execute') as motion, \
                            patch.object(return_to_home, 'TianjiHardwareOutput') as controller:
                        self.assertEqual(return_to_home.main(args), 2)
                    motion.assert_not_called()
                    controller.assert_not_called()
                    failure = json.loads(stdout.getvalue())
                    self.assertFalse(failure['plan_ready'])
                    self.assertFalse(failure['motion_commands_sent'])
                    self.assertIn('left', failure['limit_diagnostics'])
                    self.assertIn('right', failure['limit_diagnostics'])
                    self.assertIn('11.182', stderr.getvalue())
                    self.assertNotIn('Traceback', stderr.getvalue())
                    if execute:
                        self.assertEqual(json.loads(output.read_text()), failure)

    def test_faulted_arm_is_rejected(self):
        report = copy.deepcopy(self.report)
        report['arms']['right']['servo_error_words'][0] = 1
        with self.assertRaisesRegex(ValueError, '无故障待机'):
            plan(report, self.config, self.kine)

    def test_valid_home_needs_no_rescue(self):
        report = copy.deepcopy(self.report)
        for side in ('left', 'right'):
            report['arms'][side]['joints_deg'] = self.config['hardware']['startup_joints_deg'][side]
        result = plan(report, self.config, self.kine)
        self.assertIsNone(result['rescue_side'])
        self.assertEqual(len(result['phases']), 1)
        self.assertIsNone(rescue_waypoint(self.kine, 'right', np.asarray(result['goal_joints_deg']['right'])))

    def test_real_lite_jacobian_is_clipped_before_live_singularity(self):
        engine = TeleopEngine(self.config, self.kine)
        for side in ('left', 'right'):
            start = np.asarray(self.config['hardware']['startup_joints_deg'][side], dtype=float)
            start[3] = -27.
            proposed = start.copy(); proposed[3] = -26.8
            self.assertTrue(self.kine.valid_joints(side, proposed))
            self.assertLess(engine._minimum_singular_value(side, proposed), .035)
            clipped, limited = engine._limit_live_singularity(side, start, proposed)
            self.assertTrue(limited)
            self.assertGreaterEqual(engine._minimum_singular_value(side, clipped), .035 - 1e-9)
            self.assertLess(clipped[3], proposed[3])


if __name__ == '__main__':
    unittest.main()
