"""Standby utility tests use fakes only and never contact a controller."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


TOOL = Path(__file__).resolve().parents[1] / 'tools' / 'standby_robot.py'
SPEC = importlib.util.spec_from_file_location('_test_standby_robot', TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(state=1, error=0, temperatures=None, serial=1):
    temperatures = [25.] * 7 if temperatures is None else temperatures
    return {
        'states': [{'cur_state': state, 'cmd_state': -1, 'err_code': error} for _ in range(2)],
        'outputs': [{'frame_serial': serial, 'fb_joint_pos': [1.] * 7,
                     'fb_joint_them': list(temperatures)} for _ in range(2)],
    }


class StandbyToolTests(unittest.TestCase):
    def robot(self):
        robot = MagicMock()
        robot.connect.return_value = True
        robot.clear_set.return_value = True
        robot.set_state.return_value = True
        robot.send_cmd.return_value = True
        return robot

    def simulate(self, before, after=None, servo=None):
        robot = self.robot()
        values = [(before, [2, 2])]
        if after is not None:
            values.append((after, [2, 2]))
        servo = servo or {'left': [0] * 7, 'right': [0] * 7}
        with patch.object(MODULE, '_load_robot', return_value=(MagicMock(), robot)), \
             patch.object(MODULE, '_sample', side_effect=values), \
             patch.object(MODULE, 'read_servo_errors', return_value=servo), \
             patch.object(MODULE, 'version_info', return_value={}):
            report = MODULE.standby('192.168.1.190', .5)
        return robot, report

    def test_position_mode_can_only_transition_to_standby(self):
        robot, report = self.simulate(row(), row(state=0, serial=2))
        self.assertTrue(report['standby_confirmed'], report)
        self.assertTrue(report['standby_command_sent'])
        self.assertEqual([call.args for call in robot.set_state.call_args_list],
                         [('A', 0), ('B', 0)])
        robot.clear_error.assert_not_called()
        robot.set_joint_cmd_pose.assert_not_called()

    def test_impedance_mode_can_transition_to_standby(self):
        robot, report = self.simulate(row(state=3), row(state=0, serial=2))
        self.assertTrue(report['standby_confirmed'], report)
        self.assertEqual([call.args for call in robot.set_state.call_args_list],
                         [('A', 0), ('B', 0)])
        robot.clear_error.assert_not_called()
        robot.set_joint_cmd_pose.assert_not_called()

    def test_existing_standby_sends_nothing(self):
        robot, report = self.simulate(row(state=0))
        self.assertTrue(report['standby_confirmed'])
        self.assertFalse(report['standby_command_sent'])
        robot.send_cmd.assert_not_called()

    def test_fault_placeholder_or_other_mode_is_rejected(self):
        for before in (row(error=13), row(temperatures=[0.] * 7), row(state=2)):
            robot, report = self.simulate(before)
            self.assertFalse(report['standby_confirmed'])
            robot.send_cmd.assert_not_called()

    def test_unconfirmed_standby_is_failure(self):
        robot, report = self.simulate(row(), row(state=1, serial=2))
        self.assertFalse(report['standby_confirmed'])
        self.assertIn('未在时限内', report['error'])


if __name__ == '__main__':
    unittest.main()
