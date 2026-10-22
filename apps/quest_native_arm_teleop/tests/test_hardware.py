"""Offline fault injection; no controller is contacted by these tests."""
import copy
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from quest_teleop.config import load_config
from quest_teleop.hardware import HardwareError, TianjiHardwareOutput
from quest_teleop.engine import TeleopEngine
from quest_teleop.kinematics import IKError
from quest_teleop.protocol import parse_frame
from quest_teleop.control_process import _command
from test_core import LinearKine, packet


def adapter(config=None):
    module = MagicMock()
    with patch.object(TianjiHardwareOutput, '_load_module', return_value=module), \
            patch('quest_teleop.hardware.bind_robot_abi'):
        output = TianjiHardwareOutput(config or load_config(), clock=lambda: output.now)
    output.now = 10.
    output.connected = True
    output.data = {'states': [{'cur_state': 0, 'err_code': 0} for _ in range(2)],
                   'outputs': [{'frame_serial': 1, 'fb_joint_pos': [0.] * 7,
                                'fb_joint_them': [25.] * 7} for _ in range(2)]}
    output._subscribe = lambda: copy.deepcopy(output.data)
    output.read_feedback(initializing=True)
    for row in output.data['outputs']: row['frame_serial'] += 1
    output.read_feedback()
    return output


def tick(output, seconds=.02):
    output.now += seconds
    for row in output.data['outputs']: row['frame_serial'] += 1
    return output.poll()


def ready(output):
    assert output.enable() is False
    for row in output.data['states']: row['cur_state'] = 3
    tick(output)
    tick(output, .5)
    tick(output, .5)
    assert output.enabled


class HardwareOutputTests(unittest.TestCase):
    def test_configured_wuji_tool_is_sent_before_nonstartup_servo_once(self):
        config = load_config('configs/hardware_dual_arm.json', hardware=True)
        config['hardware'].update(startup_move_enabled=False, servo_mode='position',
                                  vel_ratio=1, acc_ratio=1)
        output = adapter(config)
        with patch('quest_teleop.hardware.time.sleep', side_effect=AssertionError('blocking enable')):
            self.assertFalse(output.enable())
            self.assertEqual(output.phase, 'tool_parameters')
            self.assertEqual([call.args[0] for call in output.robot.set_tool.call_args_list], ['A', 'B'])
            for call in output.robot.set_tool.call_args_list:
                self.assertEqual(call.args[1:], (
                    config['hardware']['tool_kine'], config['hardware']['tool_dyn']))
            output.robot.set_joint_cmd_pose.assert_not_called()
            output.robot.set_state.assert_not_called()
            self.assertEqual(output.robot.send_cmd.call_count, 1)
            tick(output, .29)
            output.robot.set_joint_cmd_pose.assert_not_called()
            # A canceled preparation keeps the tool setting, but a new enable
            # still waits out the vendor's 300 ms settling interval.
            output.cancel_enable()
            self.assertFalse(output.enable())
            self.assertEqual(output.phase, 'tool_parameters')
            self.assertEqual(output.robot.set_tool.call_count, 2)
            tick(output, .02)
            self.assertEqual(output.phase, 'position_parameters')
            self.assertEqual([call.args[0] for call in output.robot.set_joint_cmd_pose.call_args_list],
                             ['A', 'B'])
            tick(output, .11)
            self.assertEqual([call.args for call in output.robot.set_state.call_args_list[-2:]],
                             [('A', 1), ('B', 1)])
            for row in output.data['states']:
                row['cur_state'] = 1
            tick(output, 1.01)
            self.assertTrue(output.enabled)
            self.assertEqual(output.robot.set_tool.call_count, 2)

    def test_failed_tool_configuration_never_enables_or_targets_robot(self):
        config = load_config('configs/hardware_dual_arm.json', hardware=True)
        config['hardware']['startup_move_enabled'] = False
        output = adapter(config)
        output.robot.set_tool.side_effect = [True, False]
        with self.assertRaisesRegex(HardwareError, 'B Wuji Hand 工具参数'):
            output.enable()
        self.assertIsNotNone(output.fault)
        self.assertFalse(output.enabled)
        output.robot.send_cmd.assert_not_called()
        output.robot.set_joint_cmd_pose.assert_not_called()
        output.robot.set_state.assert_not_called()

    def test_dual_arm_profile_matches_wuji_official_startup_sequence(self):
        config = load_config('configs/hardware_dual_arm.json', hardware=True)
        config['hardware']['startup_duration_s'] = .5
        config['hardware']['startup_settle_s'] = 0.
        config['hardware']['startup_joints_deg'] = {
            'left': [0.] * 7, 'right': [0.] * 7}
        output = adapter(config)
        output._startup_joints = {s: np.zeros(7) for s in ('left', 'right')}
        def advance(seconds):
            output.now += seconds
            for row in output.data['outputs']:
                row['frame_serial'] += 1
            if output.robot.set_state.called:
                for row in output.data['states']:
                    row['cur_state'] = 3
        output.robot.reset_mock()
        with patch('quest_teleop.hardware.time.sleep', side_effect=advance):
            result = output.move_to_startup_pose()
        self.assertTrue(output.enabled)
        self.assertEqual(output.servo_state, 3)
        self.assertEqual([call.args[0] for call in output.robot.set_tool.call_args_list], ['A', 'B'])
        for call in output.robot.set_tool.call_args_list:
            self.assertEqual(call.args[1], [0, 0, 0, 0, 0, 0])
            self.assertEqual(call.args[2], [.95, 0, 0, 90, 0, 0, 0, 0, 0, 0])
        self.assertEqual([call.args for call in output.robot.set_state.call_args_list], [('A', 3), ('B', 3)])
        self.assertEqual([call.args for call in output.robot.set_vel_acc.call_args_list],
                         [('A', 60, 60), ('B', 60, 60)])
        self.assertEqual([call.args[0] for call in output.robot.set_joint_kd_params.call_args_list], ['A', 'B'])
        self.assertEqual([call.args for call in output.robot.set_impedance_type.call_args_list], [('A', 1), ('B', 1)])
        self.assertGreaterEqual(output.robot.set_joint_cmd_pose.call_count, 104)
        self.assertEqual([call.args[0] for call in output.robot.set_joint_cmd_pose.call_args_list[-2:]], ['A', 'B'])
        np.testing.assert_allclose(result['left'], np.zeros(7))

    def test_right_arm_profile_never_targets_or_enables_left_arm(self):
        config = load_config('configs/hardware_right_arm.json', hardware=True)
        output = adapter(config)
        output.robot.reset_mock()
        self.assertFalse(output.enable())
        self.assertEqual([call.args[0] for call in output.robot.set_joint_cmd_pose.call_args_list], ['B'])
        self.assertEqual([call.args[0] for call in output.robot.set_vel_acc.call_args_list], ['B'])
        tick(output, .11)
        self.assertEqual([call.args for call in output.robot.set_state.call_args_list], [('B', 1)])
        output.data['states'][1]['cur_state'] = 1
        tick(output, 1.01)
        self.assertTrue(output.enabled)
        output.robot.reset_mock()
        targets = {s: q.copy() for s, q in output.feedback['joints'].items()}
        targets['right'][0] += .02
        output.send_joint_targets(targets)
        self.assertEqual([call.args[0] for call in output.robot.set_joint_cmd_pose.call_args_list], ['B'])

        output.robot.reset_mock()
        output._set_standby()
        self.assertEqual([call.args for call in output.robot.set_state.call_args_list], [('B', 0)])

    def test_right_arm_profile_rejects_excursion_and_inactive_arm_drift(self):
        config = load_config('configs/hardware_right_arm.json', hardware=True)
        output = adapter(config)
        output._startup_joints = {s: np.zeros(7) for s in ('left', 'right')}
        output.enabled = True
        output.data['states'][1]['cur_state'] = 1
        bad = {s: np.zeros(7) for s in ('left', 'right')}; bad['right'][2] = 3.01
        with self.assertRaisesRegex(HardwareError, '±3'):
            output._write_targets(bad)
        output.data['outputs'][0]['fb_joint_pos'][0] = .31
        with self.assertRaisesRegex(HardwareError, '非控制臂偏移'):
            tick(output)

    def test_official_mode_gain_type_sequence_without_blocking(self):
        output = adapter()
        with patch('quest_teleop.hardware.time.sleep', side_effect=AssertionError('blocking enable')):
            ready(output)
        names = [call[0] for call in output.robot.method_calls]
        self.assertLess(names.index('set_joint_cmd_pose'), names.index('set_state'))
        self.assertLess(names.index('set_state'), names.index('set_joint_kd_params'))
        self.assertLess(names.index('set_joint_kd_params'), names.index('set_impedance_type'))
        self.assertFalse(output.enabling)

    def test_no_silent_send_while_disabled(self):
        output = adapter()
        with self.assertRaisesRegex(HardwareError, '尚未就绪'):
            output.send_joint_targets({'left': np.zeros(7), 'right': np.zeros(7)})
        output.robot.send_cmd.assert_not_called()

    def test_clear_set_retries_transient_sdk_sender_busy(self):
        output = adapter()
        output.robot.clear_set.side_effect = [False, False, True]
        output._send_clear_timeout_s = .01
        with patch('quest_teleop.hardware.time.sleep', side_effect=lambda _: setattr(output, 'now', output.now + .001)):
            output._clear_set()
        self.assertEqual(output.robot.clear_set.call_count, 3)
        self.assertEqual(output.send_clear_retries, 2)

    def test_clear_set_busy_timeout_is_explicit(self):
        output = adapter()
        output.robot.clear_set.return_value = False
        output._send_clear_timeout_s = .001
        with patch('quest_teleop.hardware.time.sleep', side_effect=lambda _: setattr(output, 'now', output.now + .001)):
            with self.assertRaisesRegex(HardwareError, '清空发送缓冲超时'):
                output._clear_set()

    def test_both_feedback_counters_must_advance(self):
        output = adapter()
        for _ in range(12):
            output.now += .02
            output.data['outputs'][0]['frame_serial'] += 1
        with self.assertRaisesRegex(HardwareError, 'right.*超时'):
            output.poll()
        self.assertIsNotNone(output.fault)

    def test_zero_or_nonfinite_or_error_feedback_rejected(self):
        for field, value in [('fb_joint_pos', [float('nan')] * 7), ('fb_joint_pos', [0.] * 6)]:
            output = adapter(); output.data['outputs'][0][field] = value
            with self.assertRaises(HardwareError): output.poll()
        output = adapter(); output.data['states'][1]['err_code'] = 27
        with self.assertRaisesRegex(HardwareError, '错误码 27'): output.poll()
        output = adapter(); output._changed[1] = False
        with self.assertRaises(HardwareError): output.poll()
        output = adapter(); output.data['outputs'][1]['frame_serial'] = 0
        with self.assertRaises(HardwareError): output.poll()

    def test_placeholder_axis_telemetry_is_rejected_even_when_frames_advance(self):
        output = adapter()
        output.data['outputs'][0]['fb_joint_them'] = [0.] * 7
        with self.assertRaisesRegex(HardwareError, 'DC48V'):
            tick(output)

    def test_feedback_stale_while_paused_still_faults(self):
        output = adapter(); output.now += .21
        with self.assertRaises(HardwareError): output.poll()
        self.assertFalse(output.enabled)
        with self.assertRaises(HardwareError): output.enable()

    def test_servo_loss_and_mode_timeout_latch(self):
        output = adapter(); ready(output)
        output.data['states'][0]['cur_state'] = 0
        with self.assertRaisesRegex(HardwareError, '退出控制'): tick(output)
        output = adapter(); output.enable()
        with self.assertRaisesRegex(HardwareError, '3 秒'): tick(output, 3.01)

    def test_failed_joint_send_requests_standby_even_after_disable(self):
        output = adapter(); ready(output)
        output.robot.send_cmd.side_effect = [False, True]
        before = output.robot.set_state.call_count
        with self.assertRaises(HardwareError):
            output.send_joint_targets({'left': np.zeros(7), 'right': np.zeros(7)})
        self.assertFalse(output.enabled)
        self.assertEqual(output.robot.set_state.call_count, before + 2)
        with self.assertRaises(HardwareError): output.enable()

    def test_partial_invalid_target_is_never_sent(self):
        output = adapter(); ready(output); output.robot.reset_mock()
        with self.assertRaises(HardwareError):
            output.send_joint_targets({'left': np.zeros(7), 'right': [float('nan')] * 7})
        output.robot.set_joint_cmd_pose.assert_not_called()

    def test_hold_uses_measured_not_unreached_command(self):
        output = adapter(); ready(output)
        output.data['outputs'][0]['fb_joint_pos'] = [2.] * 7
        output.hold({'left': np.ones(7) * 10, 'right': np.zeros(7)})
        self.assertIn(unittest.mock.call('A', [2.] * 7), output.robot.set_joint_cmd_pose.call_args_list)
        output.data['outputs'][0]['fb_joint_pos'] = [1.5] * 7
        output.robot.reset_mock()
        output.hold()
        self.assertEqual(output.robot.set_joint_cmd_pose.call_args_list[0].args, ('A', [2.] * 7))

    def test_startup_goal_is_held_until_a_live_command_then_pause_captures_once(self):
        config = load_config('configs/hardware_dual_arm.json', hardware=True)
        config['hardware']['startup_duration_s'] = .02
        config['hardware']['startup_settle_s'] = 0.
        config['hardware']['startup_joints_deg'] = {
            'left': [1.] * 7, 'right': [1.] * 7}
        output = adapter(config)
        output._startup_joints = {s: np.zeros(7) for s in ('left', 'right')}
        def advance(seconds):
            output.now += seconds
            for row in output.data['outputs']:
                row['frame_serial'] += 1
            if output.robot.set_state.called:
                for row in output.data['states']:
                    row['cur_state'] = 3
        with patch('quest_teleop.hardware.time.sleep', side_effect=advance):
            feedback = output.move_to_startup_pose()
        np.testing.assert_allclose(feedback['left'], np.zeros(7))
        np.testing.assert_allclose(output.hold_targets['left'], np.ones(7))
        output.robot.reset_mock()
        output.hold()
        self.assertEqual(output.robot.set_joint_cmd_pose.call_args_list[0].args, ('A', [1.] * 7))
        output.data['outputs'][0]['fb_joint_pos'] = [-.5] * 7
        output.robot.reset_mock()
        output.hold()
        self.assertEqual(output.robot.set_joint_cmd_pose.call_args_list[0].args, ('A', [1.] * 7))
        output.send_joint_targets({'left': np.ones(7), 'right': np.ones(7)})
        self.assertIsNone(output.hold_targets)
        output.data['outputs'][0]['fb_joint_pos'] = [-1.] * 7
        output.robot.reset_mock()
        output.hold()
        self.assertEqual(output.robot.set_joint_cmd_pose.call_args_list[0].args, ('A', [-1.] * 7))
        output.data['outputs'][0]['fb_joint_pos'] = [-1.5] * 7
        output.robot.reset_mock()
        output.hold()
        self.assertEqual(output.robot.set_joint_cmd_pose.call_args_list[0].args, ('A', [-1.] * 7))

    def test_cancel_pending_enable_and_close(self):
        output = adapter(); output.enable(); output.hold()
        self.assertFalse(output.enabling); self.assertFalse(output.enabled)
        output._set_standby = MagicMock()
        self.assertTrue(output.close())
        output._set_standby.assert_called_once_with(wait=True)
        output.robot.release_robot.assert_called_once_with()
        self.assertFalse(output.connected)

    def test_close_reports_standby_confirmation_failure(self):
        output = adapter(); output._motion_started = True
        output._set_standby = MagicMock(side_effect=HardwareError('state=0 timeout'))
        self.assertFalse(output.close())
        self.assertFalse(output.close())  # A second close cannot invent confirmation.
        output.robot.release_robot.assert_called_once_with()

    def test_close_without_motion_rechecks_standby_read_only(self):
        output = adapter()
        self.assertTrue(output.close())
        output.robot.set_state.assert_not_called()
        output = adapter(); output.data['states'][0]['cur_state'] = 3
        self.assertFalse(output.close())
        output.robot.set_state.assert_not_called()

    def test_connect_none_does_not_count_as_robot_connection(self):
        output = adapter(); output.connected = False; output.robot.connect.return_value = None
        with self.assertRaises(HardwareError): output.connect()
        self.assertFalse(output.connected)


class FakeHardware:
    def __init__(self, joints):
        self.joints = {s: q.copy() for s, q in joints.items()}
        self.connected, self.enabled, self.enabling = True, False, False
        self.commands = []; self.starts = 0; self.fault = None
    def poll(self):
        if self.fault: raise HardwareError(self.fault)
        return {'joints': self.read_joints(), 'states': [3, 3] if self.enabled else [0, 0],
                'serials': [1, 1], 'age_ms': [0., 0.]}
    def read_joints(self): return {s: q.copy() for s, q in self.joints.items()}
    def enable(self):
        if not self.enabling: self.starts += 1
        self.enabling = True
        return False
    def send_joint_targets(self, joints):
        if not self.enabled: raise HardwareError('disabled send')
        self.commands.append({s: q.copy() for s, q in joints.items()})
    def hold(self, _joints=None): self.enabling = False
    def cancel_enable(self): self.enabling = False
    def stop(self, reason): self.fault = reason; self.enabled = self.enabling = False


class LatchingFakeHardware(FakeHardware):
    def __init__(self, measured, startup_goal):
        super().__init__(measured)
        self.enabled = True
        self._hold_targets = {s: q.copy() for s, q in startup_goal.items()}

    @property
    def hold_targets(self):
        return (None if self._hold_targets is None else
                {s: q.copy() for s, q in self._hold_targets.items()})

    def hold(self, _joints=None):
        if self._hold_targets is None:
            self._hold_targets = self.read_joints()
        self.commands.append(self.hold_targets)
        return self.hold_targets

    def send_joint_targets(self, joints):
        super().send_joint_targets(joints)
        self._hold_targets = None


class StartupHoldTests(unittest.TestCase):
    def test_preparation_retains_goal_and_space_then_pause_captures_once(self):
        config = load_config('configs/hardware_dual_arm.json', hardware=True)
        kine = LinearKine()
        kine.valid_joints = lambda side, q: np.isfinite(q).all()
        kine.jacobian = lambda side, q: np.eye(6, 7)
        now = 10.
        engine = TeleopEngine(config, kine, clock=lambda: now)
        measured = {s: q.copy() for s, q in engine.q.items()}
        goal = {s: q.copy() for s, q in measured.items()}
        for side in ('left', 'right'):
            goal[side][0] += 1.
        output = LatchingFakeHardware(measured, goal)
        engine.attach_hardware(output, measured)
        engine.attach_keyboard()
        data = packet(seq=1, captured_ms=10000)
        engine.receive(parse_frame(data, now), 'quest')
        engine.calibrate()
        for side in ('left', 'right'):
            np.testing.assert_allclose(engine.q[side], goal[side])
            np.testing.assert_allclose(engine.measured_q[side], measured[side])
        output.joints['left'][0] -= .5
        engine.pause('准备阶段再次暂停')
        engine.receive(parse_frame({**data, 'seq': 2, 'captured_ms': 10020}, now + .02), 'quest')
        engine.calibrate()
        np.testing.assert_allclose(engine.q['left'], goal['left'])
        np.testing.assert_allclose(output.hold_targets['left'], goal['left'])
        engine.set_follow(True)
        engine.step()
        self.assertTrue(engine.active['left'])
        np.testing.assert_allclose(output.commands[-1]['left'], output.joints['left'])
        output.joints['left'][0] -= .2
        count = len(output.commands)
        engine.set_follow(False)
        self.assertEqual(len(output.commands), count + 1)
        held = output.hold_targets['left'].copy()
        np.testing.assert_allclose(held, output.joints['left'])
        output.joints['left'][0] -= .2
        engine.set_follow(False)
        np.testing.assert_allclose(output.hold_targets['left'], held)
        np.testing.assert_allclose(engine.q['left'], held)
        np.testing.assert_allclose(engine.measured_q['left'], output.joints['left'])


class EngineHardwareTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        kine = LinearKine(); kine.valid_joints = lambda side, q: np.isfinite(q).all()
        self.engine = TeleopEngine(load_config(), kine, clock=lambda: self.now)
        self.output = FakeHardware(self.engine.q)
        self.engine.attach_hardware(self.output, self.output.joints)
        self.data = packet()
        self.feed(); self.engine.calibrate()
    def feed(self, dt=.02):
        self.now += dt; self.data['seq'] += 1; self.data['captured_ms'] = self.now * 1000
        if self.engine.keyboard_mode:
            self.engine.keyboard_heartbeat()
        self.engine.receive(parse_frame(copy.deepcopy(self.data), self.now), 'quest')
        self.engine.step()
    def engage(self):
        self.data['controllers']['left']['grip'] = 1.
        self.feed()
        self.assertEqual(self.output.starts, 1)
        self.assertFalse(self.output.commands)
        self.output.enabled, self.output.enabling = True, False
        self.feed()
    def test_physical_trigger_enables_and_reanchors_without_jump(self):
        self.data['controllers']['left']['position'][2] -= .2
        self.engage()
        np.testing.assert_allclose(self.output.commands[-1]['left'], self.output.joints['left'])
        self.data['controllers']['left']['position'][2] -= .01
        self.feed()
        self.assertFalse(np.allclose(self.output.commands[-1]['left'], self.output.joints['left']))
    def test_input_watchdog_cancels_pending_servo_enable(self):
        self.data['controllers']['left']['grip'] = 1.; self.feed()
        self.now += .3; self.engine.step()
        self.assertFalse(self.engine.enabled); self.assertFalse(self.output.enabling)
        self.assertFalse(self.output.commands)

    def test_brief_tracking_loss_holds_and_reanchors_without_a(self):
        self.engine.config['tracking_loss_grace_s'] = .25
        self.engine.calibrate()
        self.engage()
        self.assertTrue(self.engine.active['left'])
        held = self.engine.q['left'].copy()
        self.data['controllers']['left']['tracked'] = False
        self.feed()
        self.assertTrue(self.engine.tracking_loss_active)
        np.testing.assert_allclose(self.engine.q['left'], held, rtol=0., atol=1e-10)
        self.data['controllers']['left']['tracked'] = True
        self.data['controllers']['left']['position'][2] -= .2
        self.feed()
        self.assertFalse(self.engine.tracking_loss_active)
        np.testing.assert_allclose(self.engine.q['left'], held, rtol=0., atol=1e-10)
        self.assertTrue(self.engine.enabled)

    def test_tracking_loss_longer_than_grace_still_requires_recalibration(self):
        self.engine.config['tracking_loss_grace_s'] = .05
        self.engine.calibrate()
        self.data['controllers']['left']['grip'] = 1.; self.feed()
        self.data['controllers']['left']['tracked'] = False
        self.feed(); self.now += .06; self.feed()
        self.assertFalse(self.engine.enabled)
        self.assertFalse(self.engine.tracking_loss_active)
        self.assertIn('持续失去位置追踪', self.engine.reason)

    def test_keyboard_tracking_loss_reanchors_and_resumes_follow(self):
        self.engine.config['tracking_loss_grace_s'] = .05
        self.engine.config['keyboard_auto_recover_tracking'] = True
        self.engine.attach_keyboard()
        self.engine.calibrate()
        self.output.enabled, self.output.enabling = True, False
        self.engine.set_follow(True)
        self.feed()
        held = self.engine.q['left'].copy()
        self.data['controllers']['left']['tracked'] = False
        self.feed(); self.now += .06; self.feed()
        self.assertTrue(self.engine.tracking_recovery_pending)
        self.assertFalse(self.engine.enabled)
        self.data['controllers']['left']['tracked'] = True
        self.data['controllers']['left']['position'][2] -= .2
        for _ in range(12):
            self.feed()
        self.assertTrue(self.engine.enabled)
        self.assertFalse(self.engine.keyboard_follow)
        self.assertFalse(self.engine.tracking_recovery_pending)
        np.testing.assert_allclose(self.engine.q['left'], held, rtol=0., atol=1e-10)
        self.engine.set_follow(True)
        self.assertTrue(self.engine.keyboard_follow)
        self.data['controllers']['left']['position'][2] -= .01
        self.feed()
        self.data['controllers']['left']['position'][2] -= .01
        self.feed()
        self.assertFalse(np.allclose(self.engine.q['left'], held))

    def _start_auto_resume_keyboard(self):
        self.engine.config['tracking_loss_grace_s'] = .05
        self.engine.config['keyboard_auto_recover_tracking'] = True
        self.engine.config['keyboard_auto_resume_tracking'] = True
        self.engine.attach_keyboard()
        self.engine.calibrate()
        self.output.enabled, self.output.enabling = True, False
        self.engine.set_follow(True)
        self.feed()
        self.assertTrue(all(self.engine.active.values()))

    def _expire_optical_tracking(self):
        self.data['controllers']['left']['tracked'] = False
        for _ in range(4):
            self.feed()
        self.assertTrue(self.engine.tracking_recovery_pending)
        self.assertFalse(self.engine.enabled)
        self.assertFalse(self.engine.keyboard_follow)

    def test_joint_keyboard_auto_resumes_after_stable_fresh_tracking_without_jump(self):
        self._start_auto_resume_keyboard()
        held = {side: q.copy() for side, q in self.engine.q.items()}
        revision = self.engine.calibration_revision
        self._expire_optical_tracking()
        self.data['controllers']['left']['tracked'] = True
        for controller in self.data['controllers'].values():
            controller['position'][2] -= .2
        for _ in range(30):
            self.feed()
        self.assertTrue(self.engine.tracking_recovery_pending)
        self.assertFalse(self.engine.keyboard_follow)
        for _ in range(4):
            self.feed()
        self.assertFalse(self.engine.tracking_recovery_pending)
        self.assertTrue(self.engine.keyboard_follow)
        self.assertTrue(self.engine.enabled)
        self.assertEqual(self.engine.calibration_revision, revision + 1)
        for side in held:
            np.testing.assert_allclose(self.engine.q[side], held[side], atol=1e-9)
        for controller in self.data['controllers'].values():
            controller['position'][2] -= .01
        self.feed()
        self.assertTrue(any(not np.allclose(self.engine.q[side], held[side]) for side in held))

    def test_joint_keyboard_pose_jump_holds_and_auto_resumes_without_space(self):
        self._start_auto_resume_keyboard()
        held = {side: q.copy() for side, q in self.engine.q.items()}
        revision = self.engine.calibration_revision
        commands = len(self.output.commands)
        self.data['controllers']['left']['position'][0] += .4
        self.feed()
        self.assertTrue(self.engine.tracking_recovery_pending)
        self.assertTrue(self.engine.tracking_recovery_resume)
        self.assertFalse(self.engine.keyboard_follow)
        self.assertEqual(len(self.output.commands), commands)
        for side in held:
            np.testing.assert_allclose(self.engine.q[side], held[side], atol=1e-9)
        for _ in range(34):
            self.feed()
        self.assertTrue(self.engine.keyboard_follow)
        self.assertTrue(self.engine.enabled)
        self.assertEqual(self.engine.calibration_revision, revision + 1)
        for side in held:
            np.testing.assert_allclose(self.engine.q[side], held[side], atol=1e-9)
        self.data['controllers']['left']['position'][2] -= .01
        self.feed()
        self.assertFalse(np.allclose(self.engine.q['left'], held['left']))

    def test_joint_keyboard_recovery_restarts_window_on_another_pose_jump(self):
        self._start_auto_resume_keyboard()
        self.data['controllers']['left']['position'][0] += .4
        self.feed()
        self.assertTrue(self.engine.tracking_recovery_pending)
        for _ in range(25):
            self.feed()
        self.assertEqual(self.engine.tracking_recovery_valid_frames, 25)
        self.data['controllers']['left']['position'][0] -= .4
        self.feed()
        self.assertEqual(self.engine.tracking_recovery_valid_frames, 1)
        self.assertFalse(self.engine.keyboard_follow)
        for _ in range(25):
            self.feed()
        self.assertFalse(self.engine.keyboard_follow)
        for _ in range(8):
            self.feed()
        self.assertTrue(self.engine.keyboard_follow)

    def test_joint_keyboard_workspace_boundary_limits_without_pausing(self):
        self.engine.config['workspace_radius_m'] = .17
        self._start_auto_resume_keyboard()
        self.data['controllers']['left']['position'][0] += .08
        self.feed()
        self.assertTrue(self.engine.keyboard_follow)
        self.assertFalse(self.engine.tracking_recovery_pending)
        base = self.engine.kine.arms['left']['base'][:3, 3]
        boundary_distance = np.linalg.norm(self.engine.target['left'][:3, 3] - base)
        self.assertLessEqual(boundary_distance, .165 + 1e-9)
        self.assertTrue(self.engine.ik_feedback['left']['limited'])
        self.data['controllers']['left']['position'][0] -= .01
        self.feed()
        self.assertTrue(self.engine.keyboard_follow)
        self.assertLess(np.linalg.norm(self.engine.target['left'][:3, 3] - base), boundary_distance)

    def test_joint_keyboard_recovery_requires_continuous_valid_frames(self):
        self._start_auto_resume_keyboard()
        self._expire_optical_tracking()
        self.data['controllers']['left']['tracked'] = True
        for _ in range(20):
            self.feed()
        self.assertEqual(self.engine.tracking_recovery_valid_frames, 20)
        # A gap in fresh optical samples restarts the stable window.
        self.feed(dt=.18)
        self.assertEqual(self.engine.tracking_recovery_valid_frames, 1)
        for _ in range(20):
            self.feed()
        self.assertFalse(self.engine.keyboard_follow)
        for _ in range(14):
            self.feed()
        self.assertTrue(self.engine.keyboard_follow)

    def test_joint_keyboard_auto_resume_does_not_override_hard_pause(self):
        for interruption in ('focus', 'reference', 'session', 'usb', 'hardware'):
            with self.subTest(interruption=interruption):
                self.setUp()
                self._start_auto_resume_keyboard()
                self._expire_optical_tracking()
                self.data['controllers']['left']['tracked'] = True
                if interruption == 'focus':
                    self.data['focused'] = False
                elif interruption == 'reference':
                    self.data['reference'] += 1
                elif interruption == 'session':
                    self.data['session'] = 'new-session'
                elif interruption == 'usb':
                    self.now += self.engine.config['input_timeout_s'] + .01
                    self.engine.step()
                else:
                    self.output.fault = 'injected feedback failure'
                    self.engine.step()
                if interruption in ('focus', 'reference', 'session'):
                    self.feed()
                self.assertFalse(self.engine.tracking_recovery_pending)
                self.assertFalse(self.engine.keyboard_follow)
                self.data['focused'] = True
                if interruption != 'hardware':
                    for _ in range(40):
                        self.feed()
                self.assertFalse(self.engine.keyboard_follow)
                self.assertFalse(self.engine.enabled)

    def test_simulated_input_and_homing_cannot_drive_hardware(self):
        self.assertEqual(self.engine.source, 'quest')
        with self.assertRaises(ValueError): self.engine.set_source('simulation')
        with self.assertRaises(ValueError): self.engine.reset()
        self.assertFalse(self.output.commands)

    def test_hardware_default_profile_limits_and_feedback_configuration(self):
        config = load_config(hardware=True)
        self.assertEqual(config['max_joint_speed_deg_s'], 2)
        self.assertEqual(config['hardware']['vel_ratio'], 1)
        self.assertEqual(config['hardware']['motion_scope'], 'wrist')
        self.assertLess(config['hardware']['feedback_timeout_s'], config['input_timeout_s'])

    def test_right_arm_profile_rejects_singular_start_and_ignores_left_trigger(self):
        config = load_config('configs/hardware_right_arm.json', hardware=True)
        kine = LinearKine(); kine.valid_joints = lambda side, q: np.isfinite(q).all()
        kine.jacobian = lambda side, q: np.diag([1., 1., 1., 1., 1., .001])[:, :7]
        engine = TeleopEngine(config, kine, clock=lambda: self.now)
        output = FakeHardware(engine.q)
        with self.assertRaisesRegex(ValueError, '奇异位'):
            engine.attach_hardware(output, output.joints)
        kine.jacobian = lambda side, q: np.eye(6, 7)
        engine.attach_hardware(output, output.joints)
        self.assertEqual(engine.control_sides, ('right',))
        self.assertEqual(engine.trigger('left'), 0.)

    def test_dual_arm_profile_checks_both_starting_jacobians(self):
        config = load_config('configs/hardware_dual_arm.json', hardware=True)
        kine = LinearKine(); kine.valid_joints = lambda side, q: np.isfinite(q).all()
        kine.jacobian = lambda side, q: (np.diag([1., 1., 1., 1., 1., .001])[:, :7]
                                        if side == 'left' else np.eye(6, 7))
        engine = TeleopEngine(config, kine, clock=lambda: self.now)
        output = FakeHardware(engine.q)
        with self.assertRaisesRegex(ValueError, 'left.*奇异位'):
            engine.attach_hardware(output, output.joints)
        kine.jacobian = lambda side, q: (np.diag([1., 1., 1., 1., 1., .001])[:, :7]
                                        if side == 'right' else np.eye(6, 7))
        with self.assertRaisesRegex(ValueError, 'right.*奇异位'):
            engine.attach_hardware(output, output.joints)
        kine.jacobian = lambda side, q: np.eye(6, 7)
        engine.attach_hardware(output, output.joints)
        self.assertEqual(engine.control_sides, ('left', 'right'))
        kine.jacobian = lambda side, q: np.diag([1., 1., 1., 1., 1., .001])[:, :7]
        with self.assertRaisesRegex(IKError, '已暂停整臂跟随'):
            engine._check_full_arm_singularity('left', output.joints['left'])

    def test_live_singularity_step_is_limited_without_recalibration_loop(self):
        config = load_config('configs/hardware_dual_arm.json', hardware=True)
        kine = LinearKine()
        kine.jacobian = lambda side, q: np.diag([1., 1., 1., 1., 1., float(q[0]) / 1000.])[:, :7]
        engine = TeleopEngine(config, kine, clock=lambda: self.now)
        start = np.array([35.1, 0., 0., 0., 0., 0., 0.])
        toward_singular = start.copy(); toward_singular[0] = 34.8
        clipped, limited = engine._limit_live_singularity('right', start, toward_singular)
        self.assertTrue(limited)
        self.assertGreaterEqual(engine._minimum_singular_value('right', clipped), .035 - 1e-9)
        self.assertGreater(clipped[0], toward_singular[0])
        away = clipped.copy(); away[0] += .2
        accepted, limited = engine._limit_live_singularity('right', clipped, away)
        self.assertFalse(limited)
        np.testing.assert_allclose(accepted, away)
        deep = start.copy(); deep[0] = 10.
        with self.assertRaisesRegex(IKError, '深度进入奇异区'):
            engine._limit_live_singularity('right', deep, away)

    def test_quest_a_returns_web_clutch_to_physical_triggers(self):
        self.engine.calibrate(web_control=True)
        self.assertIs(self.engine.ui_follow, False)
        with patch('quest_teleop.control_process.time.perf_counter', return_value=self.now):
            reply = _command(self.engine, {'data': {'action': 'calibrate'},
                                          'source': 'quest', 'queued_s': self.now})
        self.assertTrue(reply['ok'], reply)
        self.assertIsNone(self.engine.ui_follow)
        self.engage()
    def test_stale_feedback_fault_cannot_be_cleared_by_calibration(self):
        self.output.fault = 'feedback timeout'; self.feed()
        self.assertFalse(self.engine.enabled)
        with self.assertRaises(ValueError): self.engine.calibrate()
        self.assertFalse(self.output.commands)
    def test_measurement_is_distinct_from_command_and_error_latches(self):
        self.engage()
        self.output.joints['left'][0] -= 10
        state = self.engine.snapshot()
        self.feed(); state = self.engine.snapshot()
        self.assertEqual(state['arms']['left']['measured_joints_deg'][0], self.output.joints['left'][0])
        self.assertGreater(state['arms']['left']['tracking_error_deg'], 8)
        for _ in range(12): self.feed()
        self.assertIn('跟随误差', self.engine.hardware_fault)
    def test_release_during_enable_cancels_and_requires_new_trigger(self):
        self.data['controllers']['left']['grip'] = 1.; self.feed()
        self.data['controllers']['left']['grip'] = 0.; self.feed()
        self.assertFalse(self.output.enabling)
        self.assertFalse(self.output.commands)


if __name__ == '__main__': unittest.main()
