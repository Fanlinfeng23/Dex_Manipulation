import time
import unittest

from quest_teleop.config import load_config
from quest_teleop.control_process import _command
from quest_teleop.engine import TeleopEngine
from quest_teleop.protocol import parse_frame
from test_core import LinearKine, packet


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.engine = TeleopEngine(load_config(), LinearKine())

    def pause(self, source=None, cause='ack_timeout'):
        return _command(self.engine, {'data': {'action': 'input_pause', 'value': cause},
                                     'source': source, 'queued_s': time.perf_counter() - 10})

    def test_input_pause_is_source_scoped_and_stop_does_not_expire_in_queue(self):
        self.engine.set_source('quest')
        now = time.perf_counter()
        self.engine.receive(parse_frame(packet(captured_ms=now * 1000), now), 'quest')
        self.engine.calibrate()
        self.assertFalse(self.pause()['input_paused'])
        self.assertTrue(self.engine.enabled)
        self.assertTrue(self.pause('quest')['input_paused'])
        self.assertFalse(self.engine.enabled)
        self.assertIn('150 ms', self.engine.reason)

    def test_unknown_pause_cause_and_keyboard_sender_are_rejected(self):
        for cause in (None, [], 'arbitrary reason'):
            self.assertFalse(self.pause(cause=cause)['ok'])
        self.assertFalse(self.pause('keyboard')['ok'])


if __name__ == '__main__':
    unittest.main()
