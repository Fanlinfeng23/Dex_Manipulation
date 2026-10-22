import copy
import unittest

import numpy as np

from quest_teleop.config import load_config
from quest_teleop.engine import TeleopEngine
from quest_teleop.protocol import parse_frame
from test_core import LinearKine, packet


class DesktopFollowTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.data = packet()
        self.engine = TeleopEngine(load_config(), LinearKine(), clock=lambda: self.now)
        self.engine.set_source('quest')
        self.feed()

    def feed(self):
        self.now += .02
        self.data['seq'] += 1
        self.data['captured_ms'] = self.now * 1000
        self.engine.receive(parse_frame(copy.deepcopy(self.data), self.now), self.engine.source)
        self.engine.step()

    def move(self):
        self.engine.set_follow(True)
        self.feed()
        for controller in self.data['controllers'].values():
            controller['position'][2] -= .04
        for _ in range(8):
            self.feed()

    def test_start_requires_a_and_stop_resume_preserves_pose_and_calibration(self):
        with self.assertRaises(ValueError):
            self.engine.set_follow(True)
        self.engine.calibrate()
        revision = self.engine.calibration_revision
        before = self.engine.q['left'].copy()
        self.move()
        self.assertTrue(all(self.engine.active.values()))
        self.assertFalse(np.allclose(before, self.engine.q['left']))
        self.engine.set_follow(False)
        held = {s: q.copy() for s, q in self.engine.q.items()}
        self.assertTrue(self.engine.enabled)
        for c in self.data['controllers'].values():
            c['position'][0] += .4
            c['grip'] = 1.
        self.feed()
        self.assertFalse(any(self.engine.active.values()))
        self.engine.set_follow(True)
        self.feed()
        for side in held:
            np.testing.assert_allclose(self.engine.q[side], held[side], atol=1e-9)
        self.assertEqual(self.engine.calibration_revision, revision)

    def test_desktop_and_space_share_latch_without_renewing_keyboard_heartbeat(self):
        self.engine.attach_keyboard()
        self.feed()
        self.engine.calibrate()
        deadline = self.engine.keyboard_deadline
        self.engine.set_follow(True)
        self.feed()
        self.assertTrue(self.engine.keyboard_follow)
        self.assertEqual(self.engine.keyboard_deadline, deadline)
        self.engine.toggle_keyboard_follow()
        self.assertFalse(self.engine.keyboard_follow)
        self.engine.toggle_keyboard_follow()
        self.engine.set_follow(False)
        self.assertFalse(self.engine.keyboard_follow)
        self.assertTrue(self.engine.enabled)

    def test_tracking_loss_and_b_clear_latch_and_stop_aborts_return(self):
        self.engine.calibrate()
        self.move()
        self.data['controllers']['left']['tracked'] = False
        self.feed()
        self.assertIsNone(self.engine.ui_follow)
        self.assertFalse(self.engine.enabled)
        self.data['controllers']['left']['tracked'] = True
        self.feed()
        with self.assertRaises(ValueError):
            self.engine.set_follow(True)
        self.engine.calibrate()
        self.engine.reset()
        self.assertIsNotNone(self.engine.reset_plan)
        held = self.engine.q['left'].copy()
        self.engine.set_follow(False)
        self.assertIsNone(self.engine.reset_plan)
        self.feed()
        np.testing.assert_allclose(self.engine.q['left'], held, atol=1e-9)

    def test_physical_release_does_not_cancel_ui_latch_and_simulation_can_return_to_manual(self):
        self.engine.set_source('simulation')
        self.feed()
        self.engine.calibrate()
        self.engine.set_follow(True)
        self.feed()
        anchor = self.engine.anchors['left']
        self.data['safety'] = {'released': ['left', 'right']}
        self.feed()
        self.assertIs(self.engine.anchors['left'], anchor)
        self.engine.use_controller_triggers()
        self.feed()
        self.assertFalse(any(self.engine.active.values()))
        self.data['controllers']['left']['grip'] = 1.
        self.feed()
        self.assertTrue(self.engine.active['left'])
        self.assertFalse(self.engine.active['right'])

    def test_invalid_start_values_rejected_and_stale_input_cannot_start(self):
        self.engine.calibrate()
        for value in (None, 1, 'true'):
            with self.assertRaises(ValueError):
                self.engine.set_follow(value)
        self.now += .3
        with self.assertRaises(ValueError):
            self.engine.set_follow(True)
        self.engine.set_follow(False)  # Stop remains available with no fresh input.
        self.assertFalse(any(self.engine.active.values()))

    def test_high_trigger_does_not_block_a_or_move_until_a_later_release(self):
        for c in self.data['controllers'].values(): c['grip'] = 1.
        # A release edge from BEFORE the calibration must not unlock the arm.
        self.data['safety'] = {'released': ['left', 'right']}
        self.feed()
        self.engine.calibrate()
        held = {s: q.copy() for s, q in self.engine.q.items()}
        self.engine.step()
        self.assertTrue(all(self.engine.trigger_release_required.values()))
        self.assertFalse(any(self.engine.active.values()))
        self.data.pop('safety')
        for _ in range(10):
            for c in self.data['controllers'].values(): c['position'][0] += .04
            self.feed()
        self.assertTrue(self.engine.enabled)
        self.assertFalse(any(self.engine.active.values()))
        for side in held: np.testing.assert_allclose(self.engine.q[side], held[side])
        self.assertEqual(self.engine.snapshot()['arms']['right']['raw_trigger'], 1.)
        for c in self.data['controllers'].values(): c['grip'] = 0.
        self.feed()
        self.assertFalse(any(self.engine.trigger_release_required.values()))
        for c in self.data['controllers'].values(): c['grip'] = 1.
        self.feed()
        self.assertTrue(all(self.engine.active.values()))
        for side in held: np.testing.assert_allclose(self.engine.q[side], held[side])

    def test_web_start_can_explicitly_follow_after_high_trigger_calibration(self):
        for c in self.data['controllers'].values(): c['grip'] = 1.
        self.feed(); self.engine.calibrate()
        held = {s: q.copy() for s, q in self.engine.q.items()}
        for c in self.data['controllers'].values(): c['position'][0] += .4
        self.feed()
        self.engine.set_follow(True); self.feed()
        self.assertTrue(all(self.engine.active.values()))
        for side in held: np.testing.assert_allclose(self.engine.q[side], held[side])
        self.assertEqual(self.engine.snapshot()['trigger_source'], 'web')

    def test_trigger_diagnostics_distinguish_software_override_and_missing_tracking(self):
        self.engine.calibrate(); self.engine.set_follow(True); self.feed()
        state = self.engine.snapshot()
        self.assertEqual(state['trigger_source'], 'web')
        self.assertEqual(state['arms']['right']['grip'], 1.)
        self.assertEqual(state['arms']['right']['raw_trigger'], 0.)
        self.data['controllers']['right']['tracked'] = False; self.feed()
        self.assertIsNone(self.engine.snapshot()['arms']['right']['raw_trigger'])

    def test_web_calibration_does_not_let_later_physical_triggers_start_motion(self):
        self.engine.calibrate(web_control=True)
        held = {s: q.copy() for s, q in self.engine.q.items()}
        for c in self.data['controllers'].values(): c['grip'] = 1.
        self.feed()
        self.assertTrue(self.engine.enabled)
        self.assertFalse(any(self.engine.active.values()))
        self.assertEqual(self.engine.snapshot()['trigger_source'], 'web')
        for side in held: np.testing.assert_allclose(self.engine.q[side], held[side])
