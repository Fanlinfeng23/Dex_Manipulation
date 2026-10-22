"""Native Quest adapter safety gates, using synthetic SDK payloads only."""

import json
import unittest

from quest_teleop.native_adapter import NativeAdapter
from quest_teleop.protocol import parse_frame


def payload(time_ms, *, left=0.0, right=0.2, input_value=1, focus=True,
            head_status=3, left_q='0,0,0,1', right_q='0,0,0,1',
            left_valid=None, right_valid=None, function='Tracking', trigger=1.0):
    controllers = {
        'left': {'pose': f'{left},0,0,{left_q}', 'trigger': trigger},
        'right': {'pose': f'{right},0,0,{right_q}', 'trigger': trigger},
    }
    for side, valid in [('left', left_valid), ('right', right_valid)]:
        if valid is not None:
            controllers[side]['isPoseValid'] = valid
    value = {'predictTime': time_ms * 1000, 'Input': input_value,
             'appState': {'focus': focus}, 'Head': {'status': head_status,
                                                  'pose': '0,1,0,0,0,0,1'},
             'Controller': controllers}
    return json.dumps({'functionName': function, 'value': json.dumps(value)})


def warm(adapter, *, start_s=1.0, start_ms=1000.0, device='Quest'):
    frame = None
    for index in range(7):
        frame = adapter.consume(payload(start_ms + index * 100,
                                        left=index * .001,
                                        right=.2 + index * .001),
                                device, start_s + index * .1)
    return frame


class NativeAdapterTests(unittest.TestCase):
    def test_ready_requires_two_moving_controllers_and_stable_time(self):
        adapter = NativeAdapter(bridge_uuid='test')
        first = adapter.consume(payload(1000), 'Quest', 1.0)
        self.assertFalse(first['controllers']['left']['tracked'])
        self.assertEqual(first['captured_ms'], 1000.0)
        self.assertEqual(first['controllers']['left']['position'], [0.0, 0.0, 0.0])
        self.assertEqual(first['controllers']['left']['grip'], 0.0)
        for index in range(1, 7):
            frame = adapter.consume(payload(1000 + index * 100,
                                            left=index * .001,
                                            right=.2 + index * .001),
                                    'Quest', 1.0 + index * .1)
        self.assertTrue(frame['controllers']['left']['tracked'])
        self.assertTrue(frame['controllers']['right']['tracked'])
        self.assertEqual(adapter.status['tracking_validity'], 'unreported')
        self.assertFalse(adapter.status['optical_6dof_proven'])
        parse_frame(frame, 1.61)

    def test_no_input_focus_head_and_explicit_false_fail_closed(self):
        for change, expected in [({'input_value': 0}, 'native_input_not_controllers'),
                                 ({'focus': False}, 'native_focus_lost'),
                                 ({'head_status': 0}, 'native_head_invalid'),
                                 ({'left_valid': False, 'right_valid': True},
                                  'native_explicit_tracking_invalid:left')]:
            with self.subTest(change=change):
                adapter = NativeAdapter(bridge_uuid='test')
                warm(adapter)
                frame = adapter.consume(payload(1700, left=.007, right=.207, **change),
                                        'Quest', 1.7)
                self.assertFalse(frame['controllers']['left']['tracked'])
                self.assertFalse(frame['controllers']['right']['tracked'])
                self.assertEqual(adapter.status['reason'], expected)
                if change.get('input_value') == 0:
                    self.assertTrue(frame['focused'])
        adapter = NativeAdapter()
        adapter.consume(payload(1000, left_valid=True), 'Quest', 1.0)
        self.assertEqual(adapter.status['tracking_validity'], 'unreported')
        adapter.consume(payload(1100, left=.001, right=.201,
                                left_valid=True, right_valid=True), 'Quest', 1.1)
        self.assertEqual(adapter.status['tracking_validity'], 'reported')

    def test_input_two_is_not_exclusive_but_controller_gates_remain(self):
        adapter = NativeAdapter()
        for index in range(8):
            frame = adapter.consume(payload(1000+index*100, input_value=2,
                                            left=index*.001, right=.2+index*.001),
                                    'Quest', 1+index*.1)
        self.assertTrue(frame['controllers']['left']['tracked'])
        self.assertEqual(adapter.status['tracking_validity'], 'unreported')
        self.assertFalse(adapter.status['optical_6dof_proven'])
        for index in range(8, 14):
            frame = adapter.consume(payload(1000+index*100, input_value=2,
                                            left=.007, right=.207), 'Quest', 1+index*.1)
        self.assertFalse(frame['controllers']['left']['tracked'])
        self.assertIn('native_controller_frozen', adapter.reason)
        missing = json.loads(payload(2500, input_value=2))
        value = json.loads(missing['value'])
        del value['Controller']['right']
        frame = adapter.consume(value, 'Quest', 2.5)
        self.assertFalse(frame['controllers']['right']['tracked'])
        self.assertIn('native_controller_pose_invalid', adapter.reason)

    def test_freeze_treats_quaternion_sign_as_same_pose(self):
        adapter = NativeAdapter()
        warm(adapter)
        frame = None
        for index in range(7, 12):
            q = '0,0,0,-1' if index % 2 else '0,0,0,1'
            frame = adapter.consume(payload(1000 + index * 100,
                                            left=.006, right=.206,
                                            left_q=q, right_q=q),
                                    'Quest', 1.0 + index * .1)
        self.assertFalse(frame['controllers']['left']['tracked'])
        self.assertIn('native_controller_frozen', adapter.status['reason'])

    def test_jump_duplicate_out_of_order_reset_and_device_change(self):
        adapter = NativeAdapter(bridge_uuid='test')
        warm(adapter)
        last_rx = adapter.status['last_rx']
        session = adapter.status['session']
        self.assertIsNone(adapter.consume(payload(1600), 'Quest', 1.7))
        self.assertEqual(adapter.status['last_rx'], last_rx)
        self.assertIsNone(adapter.consume(payload(1580), 'Quest', 1.72))
        self.assertEqual(adapter.status['last_rx'], last_rx)
        jump = adapter.consume(payload(1700, left=.3, right=.5), 'Quest', 1.8)
        self.assertFalse(jump['controllers']['left']['tracked'])
        self.assertIn('native_controller_jump', adapter.status['reason'])
        reset = adapter.consume(payload(500, left=.31, right=.51), 'Quest', 1.9)
        self.assertNotEqual(reset['session'], session)
        self.assertFalse(reset['controllers']['left']['tracked'])
        self.assertEqual(adapter.status['clock_resets'], 1)
        other = adapter.consume(payload(600, left=.32, right=.52), 'Other', 2.0)
        self.assertNotEqual(other['session'], reset['session'])
        self.assertFalse(other['controllers']['left']['tracked'])

    def test_nontracking_wrapper_is_ignored_and_repeats_timeout(self):
        adapter = NativeAdapter()
        warm(adapter)
        previous = adapter.status['last_rx']
        self.assertIsNone(adapter.consume(payload(1700, function='DeviceStatus'),
                                          'Quest', 1.7))
        self.assertEqual(adapter.status['last_rx'], previous)
        self.assertIsNone(adapter.consume(payload(1600), 'Quest', 1.75))
        self.assertTrue(adapter.check_stale(1.81))
        self.assertFalse(adapter.status['eligible'])
        self.assertEqual(adapter.status['reason'], 'native_stream_timeout')


if __name__ == '__main__':
    unittest.main()
