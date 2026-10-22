"""Offline integration checks for the two-tracker adapter and reused arm engine."""
from __future__ import annotations

import unittest
import asyncio
import math
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from quest_teleop.engine import TeleopEngine
from quest_teleop.kinematics import Kinematics
from quest_teleop.math3d import XR_TO_WORLD
from quest_teleop.protocol import Controller, Frame

from vive_tianji.calibration import basis_from_forward_motion
from vive_tianji.teleop import (_config, _frame, _read_calibration, _gather,
                                _centers_from_stationary_samples, TrackingGateError,
                                _both_usable, _station_geometry,
                                _repeated_forward_basis, _check_live_chaperone,
                                _run, calibrate, _baseline_limits,
                                _uncalibrated_preview_calibration, _collect_pivot,
                                _joint_gate_ready, _set_joint_gate,
                                SERIALS, SAMPLE_HZ)
from vive_tianji.tracker_input import BASE_STATION_SERIALS, PairSample


def _calibration():
    return {'operator_yaw_deg': 0.0,
            'offset_tracker_m': {'left': [0.0, 0.0, 0.0],
                                 'right': [0.0, 0.0, 0.0]}}


def _pose(x, y=1.2, z=-0.5):
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix


class AdapterTests(unittest.TestCase):
    def test_vive_live_rejects_existing_quest_before_openvr_or_robot(self):
        args = SimpleNamespace(mode='live', side='both', scale=.5, seconds=1.)
        with patch('vive_tianji.joint_launcher._quest_live', return_value=True), \
                patch('vive_tianji.teleop.TrackerPair') as tracker, \
                patch('vive_tianji.teleop.ControlProcess') as controller:
            with self.assertRaisesRegex(RuntimeError, 'Quest 实机会话仍在运行'):
                asyncio.run(_run(args))
        tracker.assert_not_called()
        controller.assert_not_called()

    def test_joint_gate_needs_both_active_arms_and_fresh_valid_tracking(self):
        state = {
            'hardware': True, 'hardware_connected': True,
            'hardware_enabled': True, 'source': 'quest', 'enabled': True,
            'keyboard_mode': True, 'keyboard_follow': True, 'following': True,
            'fresh': True, 'frame_age_ms': 25., 'hardware_fault': None,
            'tracking_loss_active': False, 'tracking_recovery_pending': False,
            'arms': {'left': {'active': True, 'tracked': True},
                     'right': {'active': True, 'tracked': True}},
        }
        with tempfile.TemporaryDirectory() as folder:
            gate = Path(folder) / 'hand.follow'
            self.assertTrue(_joint_gate_ready(state, valid=True, healthy=True))
            _set_joint_gate(gate, True)
            self.assertTrue(gate.is_file())
            for change in ({'valid': False}, {'healthy': False},
                           {'state': {'keyboard_follow': False}},
                           {'state': {'hardware_fault': 'joint error'}},
                           {'state': {'frame_age_ms': 125.}},
                           {'arm': ('left', 'active', False)},
                           {'arm': ('right', 'tracked', False)}):
                variant = json.loads(json.dumps(state))
                variant.update(change.get('state', {}))
                if 'arm' in change:
                    side, field, value = change['arm']
                    variant['arms'][side][field] = value
                self.assertFalse(_joint_gate_ready(
                    variant, valid=change.get('valid', True),
                    healthy=change.get('healthy', True)))
            _set_joint_gate(gate, False)
            self.assertFalse(gate.exists())

    def test_failed_wrist_pivot_is_recorded_and_retried_in_place(self):
        sample = PairSample(1., {'left': _pose(-.3), 'right': _pose(.3)},
                            {}, False, 2, .240)
        expected = SimpleNamespace(offset_tracker_m=np.array([.01, .02, .03]))
        with tempfile.TemporaryDirectory() as folder:
            with (patch('vive_tianji.teleop.ROOT', Path(folder)),
                  patch('vive_tianji.teleop._gather',
                        return_value=([sample] * 8, 1.)) as gather,
                  patch('vive_tianji.teleop._station_geometry'),
                  patch('vive_tianji.teleop.solve_pivot',
                        side_effect=[ValueError('pivot fit is too noisy: RMS 20.3 mm, max 40.5 mm'),
                                     expected]) as solver,
                  patch('builtins.input', side_effect=['', '', ''])):
                result = _collect_pivot(None, 'left', '左', (.22, .26), {})
            self.assertIs(result, expected)
            self.assertEqual(gather.call_count, 2)
            self.assertEqual(solver.call_count, 2)
            failures = list((Path(folder) / 'logs').glob('vive-pivot-left-*.json'))
            self.assertEqual(len(failures), 1)
            data = json.loads(failures[0].read_text())
            self.assertEqual(data['serial'], SERIALS['left'])
            self.assertEqual(data['samples'], 8)
            self.assertIn('RMS 20.3 mm', data['error'])

    def test_user_can_exit_failed_wrist_retake_without_saving_calibration(self):
        sample = PairSample(1., {'left': _pose(-.3), 'right': _pose(.3)},
                            {}, False, 2, .240)
        with tempfile.TemporaryDirectory() as folder:
            with (patch('vive_tianji.teleop.ROOT', Path(folder)),
                  patch('vive_tianji.teleop._gather', return_value=([sample] * 8, 1.)),
                  patch('vive_tianji.teleop._station_geometry'),
                  patch('vive_tianji.teleop.solve_pivot',
                        side_effect=ValueError('pivot fit is too noisy')),
                  patch('builtins.input', side_effect=['', 'q'])):
                with self.assertRaisesRegex(RuntimeError, '本次未保存双腕标定'):
                    _collect_pivot(None, 'right', '右', (.22, .26), {})
            self.assertEqual(list(Path(folder).glob('vive_tianji_calibration.json')), [])

    def test_live_chaperone_gate_allows_headless_201_but_rejects_conflict(self):
        _check_live_chaperone({'state': 1, 'hmd_tracking_system': 'lighthouse'})
        _check_live_chaperone({'state': 201, 'hmd_tracking_system': 'null'})
        for status in ({'state': 201, 'hmd_tracking_system': 'lighthouse'},
                       {'state': 202, 'hmd_tracking_system': 'null'},
                       {'state': None, 'hmd_tracking_system': None, 'error': 'API failed'}):
            with self.subTest(status=status):
                with self.assertRaisesRegex(RuntimeError, '实机拒绝启动'):
                    _check_live_chaperone(status)

    def test_conflicting_chaperone_stops_before_arm_connection_and_is_logged(self):
        class Pair:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def chaperone_status(self):
                return {'state': 202, 'hmd_tracking_system': 'null'}

        sample = PairSample(1., {'left': _pose(-.3), 'right': _pose(.3)},
                            {}, False, 2, 1.7)
        args = SimpleNamespace(mode='live', side='right', scale=.2, seconds=1.)
        with tempfile.TemporaryDirectory() as folder:
            session_dir = Path(folder)
            with (patch('vive_tianji.teleop.TrackerPair', return_value=Pair()),
                  patch('vive_tianji.teleop._session_dir', return_value=session_dir),
                  patch('vive_tianji.teleop._gather', return_value=([sample], 1.)),
                  patch('vive_tianji.teleop._read_calibration') as read_calibration,
                  patch('vive_tianji.teleop._robot_preflight') as robot_preflight,
                  patch('vive_tianji.teleop.ControlProcess') as control_process):
                with self.assertRaisesRegex(RuntimeError, '实机拒绝启动'):
                    asyncio.run(_run(args))
            read_calibration.assert_not_called()
            robot_preflight.assert_not_called()
            control_process.assert_not_called()
            report = json.loads((session_dir / 'summary.json').read_text())
            self.assertEqual(report['chaperone'],
                             {'state': 202, 'hmd_tracking_system': 'null'})
            self.assertIn('Chaperone=202', report['error'])

    def test_close_layout_live_requires_real_calibration_before_robot_connection(self):
        class Pair:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def chaperone_status(self):
                return {'state': 201, 'hmd_tracking_system': 'null'}

        sample = PairSample(1., {'left': _pose(-.3), 'right': _pose(.3)},
                            {}, False, 2, .240)
        args = SimpleNamespace(mode='live', side='both', scale=.2, seconds=1.,
                               expected_baseline_m=.240)
        transient = TrackingGateError('末尾短时失位', {
            'samples': int(3 * SAMPLE_HZ), 'valid_samples': int(3 * SAMPLE_HZ) - 3,
            'max_consecutive_invalid_frames': 3,
        })
        with tempfile.TemporaryDirectory() as folder:
            with (patch('vive_tianji.teleop.TrackerPair', return_value=Pair()),
                  patch('vive_tianji.teleop._session_dir', return_value=Path(folder)),
                  patch('vive_tianji.teleop._gather',
                        side_effect=[transient, ([sample], 1.)]) as gather,
                  patch('vive_tianji.teleop._read_calibration',
                        side_effect=RuntimeError('真实标定缺失')) as read_calibration,
                  patch('vive_tianji.teleop._robot_preflight') as robot_preflight,
                  patch('vive_tianji.teleop.ControlProcess') as control_process):
                with self.assertRaisesRegex(RuntimeError, '真实标定缺失'):
                    asyncio.run(_run(args))
            self.assertEqual(gather.call_args.kwargs['baseline_limits_m'], (.22, .26))
            self.assertEqual(gather.call_args.kwargs['max_invalid_streak_frames'], 3)
            self.assertEqual(gather.call_count, 2)
            self.assertTrue(read_calibration.call_args.kwargs['hardware'])
            self.assertEqual(read_calibration.call_args.kwargs['expected_baseline_m'], .240)
            robot_preflight.assert_not_called()
            control_process.assert_not_called()

    def test_close_layout_calibration_uses_measured_window_from_first_sample(self):
        class Pair:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

        with (patch('vive_tianji.teleop.TrackerPair', return_value=Pair()),
              patch('vive_tianji.teleop._gather',
                    side_effect=RuntimeError('采样测试停止')) as gather):
            with self.assertRaisesRegex(RuntimeError, '采样测试停止'):
                calibrate(SimpleNamespace(expected_baseline_m=.240))
        self.assertEqual(gather.call_args.kwargs['baseline_limits_m'], (.22, .26))
        self.assertEqual(gather.call_args.kwargs['max_invalid_streak_frames'], 3)

    def test_calibration_retries_brief_end_dropout_without_loosening_live_gate(self):
        class Pair:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

        transient = TrackingGateError('末尾短时失位', {
            'samples': int(3 * SAMPLE_HZ), 'valid_samples': int(3 * SAMPLE_HZ) - 3,
            'max_consecutive_invalid_frames': 3,
        })
        with (patch('vive_tianji.teleop.TrackerPair', return_value=Pair()),
              patch('vive_tianji.teleop._gather',
                    side_effect=[transient, RuntimeError('第二次采样停止')]) as gather):
            with self.assertRaisesRegex(RuntimeError, '第二次采样停止'):
                calibrate(SimpleNamespace(expected_baseline_m=.240))
        self.assertEqual(gather.call_count, 2)
        self.assertEqual(gather.call_args.kwargs['max_invalid_streak_frames'], 3)

    def test_calibration_accepts_three_missing_frames_but_not_four_or_unsettled_tail(self):
        class Pair:
            def __init__(self, missing):
                self.missing = missing
                self.index = 0

            def sample(self):
                index = self.index
                self.index += 1
                right_valid = index not in self.missing
                return PairSample(index / SAMPLE_HZ,
                                  {'left': _pose(-.3),
                                   'right': _pose(.3) if right_valid else None},
                                  {'left': {'connected': True, 'tracking_result': 200},
                                   'right': {'connected': True,
                                             'tracking_result': 200 if right_valid else 101}},
                                  False, 2, .240)

        with patch('vive_tianji.teleop.time.sleep'):
            with self.assertRaises(TrackingGateError):
                _gather(Pair(set(range(30, 33))), 3., baseline_limits_m=(.22, .26))
            rows, ratio = _gather(Pair(set(range(30, 33))), 3.,
                                  baseline_limits_m=(.22, .26),
                                  max_invalid_streak_frames=3)
            self.assertEqual(len(rows), int(3 * SAMPLE_HZ) - 3)
            self.assertAlmostEqual(ratio, (3 * SAMPLE_HZ - 3) / (3 * SAMPLE_HZ))
            with self.assertRaises(TrackingGateError):
                _gather(Pair(set(range(30, 34))), 3.,
                        baseline_limits_m=(.22, .26),
                        max_invalid_streak_frames=3)
            with self.assertRaises(TrackingGateError):
                _gather(Pair(set(range(int(3 * SAMPLE_HZ) - 10,
                                       int(3 * SAMPLE_HZ) - 7))), 3.,
                        baseline_limits_m=(.22, .26),
                        max_invalid_streak_frames=3)

    def test_tracking_gate_identifies_disconnected_and_unlocated_devices(self):
        class Pair:
            def sample(self):
                return PairSample(1., {'left': None, 'right': None},
                                  {'left': {'connected': True, 'tracking_result': 101},
                                   'right': {'connected': False, 'tracking_result': 1}},
                                  False, 0)

        with patch('vive_tianji.teleop.time.sleep'):
            with self.assertRaises(TrackingGateError) as caught:
                _gather(Pair(), .05)
        expected_frames = round(.05 * SAMPLE_HZ)
        self.assertEqual(caught.exception.diagnostics['base_station_valid_frames']['0'],
                         expected_frames)
        self.assertEqual(caught.exception.diagnostics['trackers']['left']['connected_frames'],
                         expected_frames)
        self.assertEqual(caught.exception.diagnostics['trackers']['right']['connected_frames'], 0)
        self.assertIn('right', str(caught.exception))

    def test_stale_station_baseline_blocks_startup_but_wrist_pose_is_independent(self):
        class Pair:
            def __init__(self, baseline):
                self.baseline = baseline

            def sample(self):
                return PairSample(1., {'left': _pose(-.3), 'right': _pose(.3)},
                                  {'left': {'connected': True, 'tracking_result': 200},
                                   'right': {'connected': True, 'tracking_result': 200}},
                                  False, 2, self.baseline)

        stale = Pair(.129).sample()
        self.assertFalse(_both_usable(stale))
        frame = _frame(stale, 1, 'test', {})
        self.assertFalse(frame.safety['tracking_lost'])
        self.assertTrue(frame.controllers['left'].tracked)
        with patch('vive_tianji.teleop.time.sleep'):
            with self.assertRaises(TrackingGateError) as caught:
                _gather(Pair(.129), .05)
            rows, ratio = _gather(Pair(1.7), .05)
        self.assertEqual(caught.exception.diagnostics['base_station_baseline_m']['latest'], .129)
        self.assertIn('0.129 m', str(caught.exception))
        self.assertEqual(len(rows), round(.05 * SAMPLE_HZ))
        self.assertEqual(ratio, 1.)

    def test_explicit_close_layout_is_required_at_startup_and_pair_loss_stops_both(self):
        limits = _baseline_limits(.240)
        self.assertAlmostEqual(limits[0], .220)
        self.assertAlmostEqual(limits[1], .260)
        pair = PairSample(1., {'left': _pose(-.3), 'right': _pose(.3)},
                          {}, False, 2, .240)
        self.assertFalse(_both_usable(pair))  # The live/default gate stays unchanged.
        self.assertTrue(_both_usable(pair, baseline_limits_m=limits))
        self.assertTrue(all(c.tracked for c in
                            _frame(pair, 1, 'preview', {}, baseline_limits_m=limits).controllers.values()))
        changed_baseline = PairSample(2., pair.poses, {}, False, 2, .280)
        self.assertFalse(_both_usable(changed_baseline, baseline_limits_m=limits))
        self.assertFalse(_frame(changed_baseline, 2, 'preview', {}).safety['tracking_lost'])
        for invalid in (
                PairSample(2., {'left': None, 'right': _pose(.3)}, {}, False, 2, .240),
                PairSample(2., pair.poses, {}, True, 2, .240)):
            frame = _frame(invalid, 2, 'preview', {}, baseline_limits_m=limits)
            self.assertTrue(frame.safety['tracking_lost'])
            self.assertFalse(any(controller.tracked for controller in frame.controllers.values()))
        self.assertEqual(_baseline_limits(None), (.8, 5.))
        for invalid_expected in (.12, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, '0.15–5.0'):
                _baseline_limits(invalid_expected)

    def test_close_layout_calibration_must_match_live_profile(self):
        class Pair:
            reference_matrix = np.eye(4)

        first, second = BASE_STATION_SERIALS
        bases = {first: _pose(0), second: _pose(.24)}
        sample = PairSample(1., {'left': _pose(-.3), 'right': _pose(.3)},
                            {}, False, 2, .24, bases)
        quality = {'rms_m': .001, 'max_m': .002,
                   'rotation_span_deg': 30., 'samples': 100}
        data = {
            'version': 3, 'serials': SERIALS, 'expected_baseline_m': .240,
            'reference_matrix': np.eye(4).tolist(),
            'base_station_poses_standing': {serial: pose.tolist()
                                            for serial, pose in bases.items()},
            'operator_yaw_deg': 0., 'standing_to_robot_rotation': XR_TO_WORLD.tolist(),
            'offset_tracker_m': {'left': [0., 0., 0.], 'right': [0., 0., 0.]},
            'pivot_quality': {'left': quality, 'right': quality},
            'forward_motion_m': {'left': [0., 0., -.24], 'right': [0., 0., -.24]},
            'forward_repeatability': {'direction_difference_deg': 1., 'return_max_m': .003},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'calibration.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            with patch('vive_tianji.teleop.CALIBRATION', path):
                self.assertEqual(_read_calibration(Pair(), sample=sample, hardware=True,
                                                   expected_baseline_m=.240), data)
                for mismatch in (None, 1.7, .260):
                    with self.subTest(mismatch=mismatch):
                        with self.assertRaisesRegex(RuntimeError, '间距配置不一致'):
                            _read_calibration(Pair(), sample=sample, hardware=True,
                                              expected_baseline_m=mismatch)

    def test_uncalibrated_preview_stays_in_memory_and_cannot_enable_hardware(self):
        class Pair:
            reference_matrix = np.eye(4)

        with tempfile.TemporaryDirectory() as folder:
            saved = Path(folder) / 'real-calibration.json'
            with patch('vive_tianji.teleop.CALIBRATION', saved):
                calibration = _uncalibrated_preview_calibration(Pair())
                self.assertTrue(calibration['test_only'])
                self.assertFalse(saved.exists())
                self.assertFalse(_config(calibration, hardware=False,
                                         side='both', scale=.2)['hardware']['enabled'])
                saved.write_text(json.dumps(calibration), encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError, '虚拟标定不能用于实机'):
                    _read_calibration(Pair(), hardware=True)

    def test_heading_dwell_rejects_wrist_motion_while_sampling(self):
        offsets = {'left': np.zeros(3), 'right': np.zeros(3)}
        stationary = [PairSample(float(index), {'left': _pose(-.3), 'right': _pose(.3)},
                                 {}, False, 2) for index in range(90)]
        centers = _centers_from_stationary_samples(stationary, offsets)
        np.testing.assert_allclose(centers['left'], [-.3, 1.2, -.5])
        drifting = [PairSample(float(index),
                               {'left': _pose(-.3, z=-.5 - index * .00012),
                                'right': _pose(.3, z=-.5 - index * .00012)},
                               {}, False, 2) for index in range(90)]
        with self.assertRaisesRegex(RuntimeError, '首尾漂移'):
            _centers_from_stationary_samples(drifting, offsets)

    def test_repeated_long_forward_motion_checks_heading_and_return(self):
        neutral = {'left': np.array([-.3, 1.2, -.5]),
                   'right': np.array([.3, 1.2, -.5])}
        first = {side: point + np.array([0., 0., -.25])
                 for side, point in neutral.items()}
        returned = {side: point + np.array([0., 0., -.003])
                    for side, point in neutral.items()}
        second = {side: point + np.array([0., 0., -.25])
                  for side, point in returned.items()}
        basis, moves, quality = _repeated_forward_basis(
            neutral, first, returned, second)
        np.testing.assert_allclose(basis @ [0., 0., -1.], [1., 0., 0.])
        self.assertAlmostEqual(np.linalg.norm(moves['left']), .25)
        self.assertAlmostEqual(quality['return_max_m'], .003)
        drifted = {side: point + np.array([.03, 0., -.25])
                   for side, point in returned.items()}
        with self.assertRaisesRegex(RuntimeError, '方向相差'):
            _repeated_forward_basis(neutral, first, returned, drifted)
        with self.assertRaisesRegex(RuntimeError, '回到起点'):
            _repeated_forward_basis(neutral, first,
                                    {side: point + np.array([0., 0., -.02])
                                     for side, point in neutral.items()}, second)

    def test_virtual_preview_calibration_cannot_authorize_hardware(self):
        class Pair:
            reference_matrix = np.eye(4)

        data = {
            'version': 3, 'serials': SERIALS, 'test_only': True,
            'reference_matrix': np.eye(4).tolist(),
            'operator_yaw_deg': 0.,
            'standing_to_robot_rotation': XR_TO_WORLD.tolist(),
            'offset_tracker_m': {'left': [0., 0., 0.], 'right': [0., 0., 0.]},
            'pivot_quality': {'left': {'rms_m': .001}, 'right': {'rms_m': .001}},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test-calibration.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            with patch('vive_tianji.teleop.CALIBRATION', path):
                self.assertEqual(_read_calibration(Pair())['version'], 3)
                with self.assertRaisesRegex(RuntimeError, '虚拟标定不能用于实机'):
                    _read_calibration(Pair(), hardware=True)

    def test_old_layout_calibration_version_is_rejected(self):
        class Pair:
            reference_matrix = np.eye(4)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old-calibration.json'
            path.write_text(json.dumps({'version': 2, 'serials': SERIALS}), encoding='utf-8')
            with patch('vive_tianji.teleop.CALIBRATION', path):
                with self.assertRaisesRegex(RuntimeError, '重新标定'):
                    _read_calibration(Pair())

    def test_relocated_or_unresolved_station_rejects_saved_wrist_calibration(self):
        first, second = BASE_STATION_SERIALS
        saved = {first: _pose(0).tolist(), second: _pose(1.7).tolist()}
        okay = PairSample(1., {'left': _pose(-.3), 'right': _pose(.3)},
                          {}, False, 2, 1.7,
                          {first: _pose(0), second: _pose(1.7)})
        _station_geometry(okay, saved)
        relocated = PairSample(2., okay.poses, {}, False, 2, 1.7,
                               {first: _pose(.05), second: _pose(1.75)})
        with self.assertRaisesRegex(RuntimeError, '重新运行 calibrate'):
            _station_geometry(relocated, saved)
        with self.assertRaisesRegex(RuntimeError, '位姿不完整'):
            _station_geometry(PairSample(3., okay.poses, {}, False, 1), saved)
        one_visible = PairSample(4., okay.poses, {}, False, 1, None,
                                 {first: _pose(0)})
        _station_geometry(one_visible, saved, require_both=False)
        moved_visible = PairSample(5., okay.poses, {}, False, 1, None,
                                   {first: _pose(.05)})
        with self.assertRaisesRegex(RuntimeError, '重新运行 calibrate'):
            _station_geometry(moved_visible, saved, require_both=False)

    def test_single_station_reference_does_not_mask_valid_wrist_tracking(self):
        sample = PairSample(1.0, {'left': _pose(-.3), 'right': _pose(.3)},
                            {}, False, 1)
        frame = _frame(sample, 1, 'test', {})
        self.assertFalse(frame.safety['tracking_lost'])
        self.assertTrue(frame.controllers['left'].tracked)
        self.assertTrue(frame.controllers['right'].tracked)
        sample = PairSample(2.0, {'left': None, 'right': _pose(.3)}, {}, False, 2)
        frame = _frame(sample, 2, 'test', {})
        self.assertTrue(frame.safety['tracking_lost'])
        self.assertFalse(frame.controllers['right'].tracked)

    def test_live_profile_never_auto_moves_and_requires_position_mode(self):
        config = _config(_calibration(), hardware=True, side='right', scale=.5)
        self.assertTrue(config['hardware']['enabled'])
        self.assertFalse(config['hardware']['startup_move_enabled'])
        self.assertEqual(config['hardware']['servo_mode'], 'position')
        self.assertEqual(config['hardware']['control_sides'], ['right'])
        self.assertEqual(config['hardware']['joint_excursion_deg'], 3.0)
        self.assertEqual(config['hardware']['vel_ratio'], 1)
        self.assertEqual(config['max_joint_speed_deg_s'], 2.)
        self.assertEqual(config['input_filter']['type'], 'one_euro')
        self.assertTrue(config['keyboard_auto_recover_tracking'])
        self.assertTrue(config['keyboard_auto_resume_tracking'])

    def test_dual_arm_profile_uses_vendor_joint_impedance_without_startup_motion(self):
        config = _config(_calibration(), hardware=True, side='both', scale=1.)
        self.assertFalse(config['hardware']['startup_move_enabled'])
        self.assertEqual(config['hardware']['servo_mode'], 'impedance')
        self.assertEqual(config['hardware']['vel_ratio'], 60)
        self.assertEqual(config['hardware']['acc_ratio'], 60)
        self.assertEqual(config['hardware']['joint_k'], [2, 2, 2, 1.6, 1, 1, 1])
        self.assertEqual(config['hardware']['joint_d'], [.3, .3, .3, .2, .2, .2, .2])
        self.assertEqual(config['hardware']['joint_excursion_deg'], 3.)
        self.assertEqual(config['max_target_speed_m_s'], .12)
        self.assertEqual(config['max_target_rotation_rad_s'], .5)
        self.assertEqual(config['max_joint_speed_deg_s'], 20.)
        self.assertEqual(config['input_filter']['type'], 'none')
        self.assertEqual(config['hardware']['max_tracking_error_deg'], 8)
        self.assertEqual(config['hardware']['tracking_error_timeout_s'], .2)
        self.assertEqual(config['control_hz'], 120)
        self.assertEqual(config['ipc_poll_interval_s'], .001)

    def test_full_range_dual_arm_keeps_speed_and_tracking_safety(self):
        config = _config(_calibration(), hardware=True, side='both', scale=.5,
                         full_range=True)
        self.assertNotIn('joint_excursion_deg', config['hardware'])
        self.assertEqual(config['hardware']['control_sides'], ['left', 'right'])
        self.assertFalse(config['hardware']['startup_move_enabled'])
        self.assertEqual(config['max_joint_speed_deg_s'], 20.)
        self.assertEqual(config['hardware']['vel_ratio'], 60)
        self.assertEqual(config['hardware']['servo_mode'], 'impedance')
        self.assertTrue(config['keyboard_auto_recover_tracking'])
        self.assertTrue(config['keyboard_auto_resume_tracking'])
        with self.assertRaisesRegex(ValueError, '仅用于双臂实机'):
            _config(_calibration(), hardware=True, side='right', scale=.5,
                    full_range=True)

    def test_relative_mapping_follows_robot_forward_in_preview(self):
        mapping = basis_from_forward_motion(
            [-0.3, 1.2, -0.4], [0.3, 1.2, -0.5],
            [-0.3, 1.2, -0.5], [0.3, 1.2, -0.6])
        np.testing.assert_allclose(mapping @ [0, 0, -1], [1, 0, 0])
        yaw_matrix = mapping @ XR_TO_WORLD.T
        self.assertTrue(np.allclose(yaw_matrix, Rotation.from_euler('z', 0.).as_matrix()))
        yaw = math.degrees(math.atan2(yaw_matrix[1, 0], yaw_matrix[0, 0]))
        calibration = _calibration()
        calibration['operator_yaw_deg'] = yaw
        config = _config(calibration, hardware=False, side='both', scale=.5)
        now = [0.0]
        engine = TeleopEngine(config, Kinematics(config), clock=lambda: now[0])
        engine.attach_keyboard()

        def frame(seq, z):
            return Frame('test', 0, seq, now[0], True, {
                'left': Controller([-.3, 1.2, z], [0, 0, 0, 1], True, 0),
                'right': Controller([.3, 1.2, z], [0, 0, 0, 1], True, 0),
            }, now[0], safety={'released': [], 'tracking_lost': False,
                               'focus_lost': False})

        engine.receive(frame(1, -.5), 'quest')
        engine.calibrate()
        engine.toggle_keyboard_follow()
        now[0] += .012
        engine.step()  # establish the clutch anchor
        start_x = engine.target['left'][0, 3]
        now[0] += .012
        engine.receive(frame(2, -.51), 'quest')
        engine.step()
        self.assertAlmostEqual(engine.target['left'][0, 3] - start_x, .005, places=4)
        self.assertAlmostEqual(engine.target['right'][0, 3] - start_x, .005, places=4)
        engine.pause()
        self.assertFalse(engine.keyboard_follow)
        self.assertFalse(any(engine.active.values()))


class MeasuredCalibrationTests(unittest.TestCase):
    def test_manual_left_and_pivot_right_are_recorded_with_correct_serials(self):
        class Pair:
            reference_matrix = np.eye(4)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        first, second = BASE_STATION_SERIALS
        stations = {first: _pose(0.), second: _pose(.24)}
        sample = PairSample(1., {'left': _pose(-.3), 'right': _pose(.3)},
                            {}, False, 2, .24, stations)
        neutral = {'left': np.array([-.3, 1.2, -.5]),
                   'right': np.array([.3, 1.2, -.5])}
        first_forward = {side: point + [0., 0., -.24]
                         for side, point in neutral.items()}
        returned = {side: point + [0., 0., -.003]
                    for side, point in neutral.items()}
        second_forward = {side: point + [0., 0., -.24]
                          for side, point in returned.items()}
        pivot = SimpleNamespace(
            offset_tracker_m=np.array([.04, 0., .06]), rms_m=.001,
            max_m=.003, rotation_span_deg=35., condition_number=12., samples=100)
        args = SimpleNamespace(expected_baseline_m=.240,
                               left_offset_mm=[-12., -35., 54.],
                               right_offset_mm=None)
        with tempfile.TemporaryDirectory() as folder:
            saved = Path(folder) / 'calibration.json'
            with (patch('vive_tianji.teleop.CALIBRATION', saved),
                  patch('vive_tianji.teleop.TrackerPair', return_value=Pair()),
                  patch('vive_tianji.teleop._gather_startup', return_value=([sample], 1.)),
                  patch('vive_tianji.teleop._gather', return_value=([sample], 1.)),
                  patch('vive_tianji.teleop._station_geometry'),
                  patch('vive_tianji.teleop._centers_from_stationary_samples',
                        side_effect=[neutral, first_forward, returned, second_forward]),
                  patch('vive_tianji.teleop._collect_pivot', return_value=pivot) as fit,
                  patch('builtins.input', return_value='')):
                calibrate(args)
            fit.assert_called_once()
            self.assertEqual(fit.call_args.args[1], 'right')
            data = json.loads(saved.read_text(encoding='utf-8'))
            self.assertEqual(data['offset_calibration']['left']['method'], 'measured')
            self.assertEqual(data['offset_calibration']['left']['serial'], SERIALS['left'])
            self.assertEqual(data['offset_calibration']['left']['offset_mm'],
                             [-12., -35., 54.])
            self.assertEqual(data['offset_calibration']['right']['method'], 'pivot')
            self.assertEqual(data['offset_calibration']['right']['serial'], SERIALS['right'])
            self.assertEqual(set(data['pivot_quality']), {'right'})
            with patch('vive_tianji.teleop.CALIBRATION', saved):
                self.assertEqual(_read_calibration(Pair(), sample=sample,
                                                   hardware=True,
                                                   expected_baseline_m=.240), data)

    def test_reader_rejects_serial_or_vector_record_mismatch(self):
        class Pair:
            reference_matrix = np.eye(4)

        first, second = BASE_STATION_SERIALS
        stations = {first: _pose(0.), second: _pose(.24)}
        sample = PairSample(1., {'left': _pose(-.3), 'right': _pose(.3)},
                            {}, False, 2, .24, stations)
        source = lambda side: {
            'method': 'measured', 'serial': SERIALS[side],
            'frame': 'openvr_tracker_local_xyz',
            'origin': 'tracker_3_mounted_screw_datum',
            'landmark': 'anatomical_wrist_center',
            'offset_mm': [10., -35., 55.],
        }
        data = {
            'version': 3, 'serials': SERIALS,
            'expected_baseline_m': .240,
            'reference_matrix': np.eye(4).tolist(),
            'base_station_poses_standing': {serial: pose.tolist()
                                            for serial, pose in stations.items()},
            'operator_yaw_deg': 0.,
            'standing_to_robot_rotation': XR_TO_WORLD.tolist(),
            'offset_tracker_m': {'left': [.01, -.035, .055],
                                 'right': [.01, -.035, .055]},
            'offset_calibration': {side: source(side) for side in ('left', 'right')},
            'pivot_quality': {},
            'forward_motion_m': {'left': [0., 0., -.24],
                                 'right': [0., 0., -.24]},
            'forward_repeatability': {'direction_difference_deg': 1.,
                                      'return_max_m': .003},
        }
        with tempfile.TemporaryDirectory() as folder:
            saved = Path(folder) / 'calibration.json'
            with patch('vive_tianji.teleop.CALIBRATION', saved):
                saved.write_text(json.dumps(data), encoding='utf-8')
                _read_calibration(Pair(), sample=sample, hardware=True,
                                  expected_baseline_m=.240)
                data['offset_calibration']['left']['serial'] = SERIALS['right']
                saved.write_text(json.dumps(data), encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError, '序列|来源'):
                    _read_calibration(Pair(), sample=sample, hardware=True,
                                      expected_baseline_m=.240)
                data['offset_calibration']['left']['serial'] = SERIALS['left']
                data['offset_tracker_m']['left'][0] = .05
                saved.write_text(json.dumps(data), encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError, '不一致'):
                    _read_calibration(Pair(), sample=sample, hardware=True,
                                      expected_baseline_m=.240)


if __name__ == '__main__':
    unittest.main()
