import unittest
from unittest.mock import MagicMock, patch

from quest_teleop.timing import TimerResolution


class TimerResolutionTests(unittest.TestCase):
    def test_native_request_is_balanced_and_repeated_close_is_safe(self):
        api = MagicMock()
        api.timeBeginPeriod.return_value = 0
        with patch('quest_teleop.timing.sys.platform', 'win32'), \
                patch('quest_teleop.timing.ctypes.WinDLL', return_value=api, create=True):
            timer = TimerResolution()
            timer.start()
            timer.start()
            self.assertEqual(timer.period_ms, 1)
            api.timeBeginPeriod.assert_called_once_with(1)
            timer.close()
            timer.close()
            self.assertEqual(timer.period_ms, 0)
            api.timeEndPeriod.assert_called_once_with(1)

    def test_failed_request_keeps_default_and_does_not_release_an_unacquired_period(self):
        api = MagicMock()
        api.timeBeginPeriod.return_value = 97
        with patch('quest_teleop.timing.sys.platform', 'win32'), \
                patch('quest_teleop.timing.ctypes.WinDLL', return_value=api, create=True), \
                self.assertLogs('quest_teleop.timing', level='WARNING'):
            timer = TimerResolution()
            timer.start()
            self.assertEqual(timer.period_ms, 0)
            timer.close()
            api.timeEndPeriod.assert_not_called()

    def test_non_windows_does_not_load_native_library(self):
        with patch('quest_teleop.timing.sys.platform', 'linux'), \
                patch('quest_teleop.timing.ctypes.WinDLL', create=True) as native:
            timer = TimerResolution()
            timer.start()
            timer.close()
            native.assert_not_called()
            self.assertEqual(timer.period_ms, 0)
