"""Wrist commissioning tests: vendor math and mocked output, never a robot."""
import copy
import ctypes
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from quest_teleop.config import load_config
from quest_teleop.engine import TeleopEngine
from quest_teleop.hardware import HardwareError
from quest_teleop.kinematics import Kinematics
from quest_teleop.protocol import parse_frame
from quest_teleop.vendor_checks import read_int_parameter, read_servo_errors, version_info
from quest_teleop.wrist import WristEnvelope
from test_core import packet
from test_hardware import adapter, tick


class WristMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(hardware=True)
        cls.kine = Kinematics(cls.config)

    def setUp(self):
        self.q = {s: np.array(self.config['arms'][s]['initial_joints_deg']) for s in ('left', 'right')}
        self.wrist = WristEnvelope(self.q, self.config['hardware'])

    def test_translation_alone_never_changes_any_joint(self):
        for side in self.q:
            target = self.kine.fk(side, self.q[side]); target[:3, 3] += [100, -100, 20]
            result = self.wrist.track(self.kine, side, target, self.q[side], .02)
            np.testing.assert_allclose(result.joints_deg, self.q[side], atol=1e-10)

    def test_large_rotation_keeps_proximal_joints_and_global_range(self):
        for side in self.q:
            current = self.q[side].copy()
            target = self.kine.fk(side, current)
            axis = self.kine.jacobian(side, current)[3:, 5]
            target[:3, :3] = Rotation.from_rotvec(axis * .8).as_matrix() @ target[:3, :3]
            for _ in range(250):
                result = self.wrist.track(self.kine, side, target, current, .02)
                self.assertLessEqual(np.max(np.abs(result.joints_deg - current)), .02000001)
                np.testing.assert_array_equal(result.joints_deg[:5], self.q[side][:5])
                self.assertTrue(self.kine.valid_joints(side, result.joints_deg))
                current = result.joints_deg
            self.assertGreater(np.max(np.abs(current[5:] - self.q[side][5:])), 1)
            self.assertLessEqual(np.max(np.abs(current[5:] - self.q[side][5:])), 3.00000001)

    def test_output_rejects_proximal_and_out_of_envelope_targets(self):
        for index in range(7):
            q = copy.deepcopy(self.q); q['right'][index] += .01 if index < 5 else 3.01
            with self.assertRaises(ValueError): self.wrist.check_commands(q)

    def test_locked_feedback_drift_is_not_reanchored(self):
        q = copy.deepcopy(self.q); q['left'][0] += .31
        with self.assertRaises(ValueError): self.wrist.check_feedback(q)
        q['left'][0] -= .21
        held = self.wrist.hold_target('left', q['left'])
        np.testing.assert_array_equal(held[:5], self.q['left'][:5])

    def test_repeated_calibration_does_not_reset_startup_envelope(self):
        now = [10.]
        engine = TeleopEngine(self.config, self.kine, clock=lambda: now[0])
        baseline = copy.deepcopy(engine.wrist.baseline)
        data = packet()
        for index in range(5):
            now[0] += .02; data['seq'] = index; data['captured_ms'] = now[0] * 1000
            engine.receive(parse_frame(data, now[0]), 'simulation')
            engine.calibrate()
        for side in baseline: np.testing.assert_array_equal(engine.wrist.baseline[side], baseline[side])

    def test_engine_routes_live_controller_rotation_only_to_wrist(self):
        now = [10.]
        engine = TeleopEngine(self.config, self.kine, clock=lambda: now[0])
        data = packet()
        def feed():
            now[0] += .02; data['seq'] += 1; data['captured_ms'] = now[0] * 1000
            engine.receive(parse_frame(copy.deepcopy(data), now[0]), 'simulation'); engine.step()
        feed(); engine.calibrate()
        data['controllers']['left']['grip'] = 1
        feed()
        axis = self.kine.jacobian('left', engine.q['left'])[3:, 5]
        moved = False
        for i in range(250):
            rotation = Rotation.from_rotvec(axis * min(.5, i * .003)).as_matrix()
            data['controllers']['left']['quaternion'] = Rotation.from_matrix(
                engine.basis.T @ rotation @ engine.basis).as_quat().tolist()
            data['controllers']['left']['position'][0] += .0002
            feed()
            self.assertTrue(engine.enabled, engine.reason)
            for side in self.q:
                np.testing.assert_array_equal(engine.q[side][:5], self.q[side][:5])
                self.assertLessEqual(np.max(np.abs(engine.q[side][5:] - self.q[side][5:])), 3.00000001)
            moved |= bool(np.max(np.abs(engine.q['left'][5:] - self.q['left'][5:])) > .1)
        self.assertTrue(moved)
        np.testing.assert_allclose(engine.q['right'], self.q['right'])

    def test_config_rejects_expanding_wrist_restrictions(self):
        for key, value in [('wrist_max_excursion_deg', 4), ('wrist_max_speed_deg_s', 3),
                           ('locked_joint_tolerance_deg', 1), ('servo_mode', 'impedance'),
                           ('vel_ratio', 10), ('acc_ratio', 10)]:
            config = copy.deepcopy(self.config); config['hardware'][key] = value
            import json
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'config.json'; path.write_text(json.dumps(config))
                with self.assertRaises(ValueError): load_config(path)


class WristOutputTests(unittest.TestCase):
    def wrist_adapter(self):
        output = adapter()
        output.settings = load_config(hardware=True)['hardware']
        output._wrist = WristEnvelope(output.read_joints(), output.settings)
        return output

    def ready(self, output):
        output.enable()
        self.assertEqual(output.phase, 'position_parameters')
        output.robot.set_state.assert_not_called()
        tick(output, .11)
        self.assertEqual(output.phase, 'position_wait')
        for row in output.data['states']: row['cur_state'] = 1
        tick(output, .5); self.assertFalse(output.enabled)
        tick(output, .51); self.assertTrue(output.enabled)

    def test_position_enable_waits_and_does_not_write_impedance(self):
        output = self.wrist_adapter(); self.ready(output)
        output.robot.set_joint_kd_params.assert_not_called()
        output.robot.set_impedance_type.assert_not_called()
        self.assertIn(unittest.mock.call('A', 1), output.robot.set_state.call_args_list)

    def test_speed_is_limited_again_at_output_boundary(self):
        output = self.wrist_adapter(); self.ready(output)
        q = output.read_joints(); q['right'][5] = 3
        output.send_joint_targets(q)
        sent = output.robot.set_joint_cmd_pose.call_args_list[-1].args[1]
        self.assertLessEqual(sent[5], .100000001)
        output.now += .01
        output.send_joint_targets(q)
        self.assertLessEqual(output.robot.set_joint_cmd_pose.call_args_list[-1].args[1][5], .120000001)

    def test_proximal_command_is_blocked_before_either_arm_is_buffered(self):
        output = self.wrist_adapter(); self.ready(output); output.robot.reset_mock()
        q = output.read_joints(); q['right'][4] = .01
        with self.assertRaises(HardwareError): output.send_joint_targets(q)
        output.robot.set_joint_cmd_pose.assert_not_called()

    def test_stop_hold_never_moves_locked_targets_to_measurement_noise(self):
        output = self.wrist_adapter(); self.ready(output)
        output.data['outputs'][0]['fb_joint_pos'][0] = .1
        output.hold()
        sent = output.robot.set_joint_cmd_pose.call_args_list[-2].args[1]
        self.assertEqual(sent[0], 0.)

    def test_emergency_stop_and_version_mismatch_connect_send_no_motion(self):
        for mismatch in (True, False):
            output = adapter(); output.connected = False
            if not mismatch: output.data['states'][0]['err_code'] = 13
            with patch('quest_teleop.hardware.version_info', side_effect=RuntimeError('version') if mismatch else None,
                       return_value={}), patch('quest_teleop.hardware.controller_model_info', return_value={}):
                with self.assertRaises((RuntimeError, HardwareError)): output.connect()
            output.robot.set_state.assert_not_called()
            output.robot.set_joint_cmd_pose.assert_not_called()
            output.robot.send_cmd.assert_not_called()
            output.robot.release_robot.assert_called_once()


class VendorABITests(unittest.TestCase):
    def test_servo_error_read_is_raw_and_read_only(self):
        robot = MagicMock()
        def fill_a(values):
            for i in range(7): values[i] = i
        def fill_b(values):
            for i in range(7): values[i] = 10 + i
        robot.robot.OnGetServoErr_A.side_effect = fill_a
        robot.robot.OnGetServoErr_B.side_effect = fill_b
        self.assertEqual(read_servo_errors(robot), {
            'left': list(range(7)), 'right': list(range(10, 17))})
        robot.robot.OnClearErr_A.assert_not_called()
        robot.robot.OnClearErr_B.assert_not_called()

    def test_integer_query_uses_c_long_buffer(self):
        signature = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_char_p, ctypes.POINTER(ctypes.c_long))
        expected = 2 ** 40 + 100343009 if ctypes.sizeof(ctypes.c_long) == 8 else 100343009
        @signature
        def query(name, pointer):
            self.assertEqual(name, b'VERSION'); pointer[0] = expected; return 0
        robot = MagicMock(); robot.robot.OnGetIntPara = query
        self.assertEqual(read_int_parameter(robot, 'VERSION'), expected)

    def test_mismatched_protocol_is_rejected(self):
        robot = MagicMock(); robot.SDK_version.return_value = 100343014
        with patch('quest_teleop.vendor_checks.read_int_parameter', return_value=100341001):
            with self.assertRaisesRegex(RuntimeError, '不匹配'): version_info(robot)


if __name__ == '__main__':
    unittest.main()
