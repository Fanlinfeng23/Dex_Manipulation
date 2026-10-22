"""Recovery utility tests use fakes only and never contact a controller."""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


TOOL = Path(__file__).resolve().parents[1] / 'tools' / 'recover_robot.py'
SPEC = importlib.util.spec_from_file_location('_test_recover_robot', TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(state=100, command=0, error=13, joints=None, serial=1):
    joints = [1.] * 7 if joints is None else joints
    return {'states': [{'cur_state': state, 'cmd_state': command, 'err_code': error} for _ in range(2)],
            'outputs': [{'frame_serial': serial, 'fb_joint_pos': list(joints)} for _ in range(2)]}


class RecoveryToolTests(unittest.TestCase):
    def robot(self):
        robot = MagicMock()
        robot.connect.return_value = True
        robot.clear_set.return_value = True
        robot.set_state.return_value = True
        robot.send_cmd.return_value = True
        return robot

    def simulate(self, before, after):
        robot = self.robot()
        with patch.object(MODULE, '_load_robot', return_value=(MagicMock(), robot)), \
             patch.object(MODULE, '_sample', side_effect=[(before, [2, 2]), (after, [2, 2])]), \
             patch.object(MODULE, '_snapshot', side_effect=lambda _r, value, unique: {
                 'unique_feedback_frames': unique,
                 'arms': {side: {**value['states'][i], 'joints_deg': value['outputs'][i]['fb_joint_pos'],
                                  'frame_serial': value['outputs'][i]['frame_serial'],
                                  'servo_error_words': [0] * 7}
                          for i, side in enumerate(('left', 'right'))}}), \
             patch.object(MODULE, 'version_info', return_value={}), \
             patch.object(MODULE, 'controller_model_info', return_value={}):
            report = MODULE.recover('192.168.1.190', .5)
        return robot, report

    def test_only_estop_standby_shape_may_be_cleared(self):
        robot, report = self.simulate(row(), row(state=0, error=0, serial=2))
        self.assertTrue(report['recovered'], report)
        self.assertEqual(robot.clear_error.call_count, 2)
        self.assertEqual(robot.set_state.call_args_list[0].args, ('A', 0))
        self.assertEqual(robot.set_state.call_args_list[1].args, ('B', 0))
        robot.set_joint_cmd_pose.assert_not_called()

    def test_unknown_error_is_never_cleared(self):
        robot, report = self.simulate(row(error=27), row(error=27))
        self.assertFalse(report['recovered'])
        robot.clear_error.assert_not_called()
        robot.send_cmd.assert_not_called()

    def test_all_zero_feedback_never_passes_recovery(self):
        robot, report = self.simulate(row(), row(state=0, error=0, joints=[0.] * 7, serial=2))
        self.assertFalse(report['recovered'])
        self.assertIn('非全零', report['message'])

    def test_persistent_dual_estop_is_classified_as_hardware_safety_chain(self):
        robot, report = self.simulate(row(), row(serial=2))
        self.assertFalse(report['recovered'])
        self.assertEqual(report['failure_class'], 'hardware_safety_chain_still_asserted')
        self.assertIn('不要重复', report['message'])


if __name__ == '__main__':
    unittest.main()
