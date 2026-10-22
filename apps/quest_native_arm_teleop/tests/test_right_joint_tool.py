"""右臂逐轴工具的纯计算测试；不连接控制柜。"""
import unittest

import numpy as np

from tools.test_right_joints import quintic, target_at


class RightJointToolTests(unittest.TestCase):
    def test_quintic_has_zero_endpoint_velocity_and_is_monotonic(self):
        values = np.array([quintic(x) for x in np.linspace(0, 1, 1001)])
        self.assertEqual(values[0], 0.)
        self.assertEqual(values[-1], 1.)
        self.assertTrue(np.all(np.diff(values) >= 0.))
        self.assertLess(values[1], 1e-6)
        self.assertLess(1 - values[-2], 1e-6)

    def test_target_changes_only_requested_axis(self):
        start = np.arange(7, dtype=float)
        for axis in range(7):
            result = target_at(start, axis, .5, 1.)
            expected = start.copy(); expected[axis] += .5
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(start, np.arange(7, dtype=float))


if __name__ == '__main__':
    unittest.main()
