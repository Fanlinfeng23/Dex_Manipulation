"""Mocked OpenVR tests; no SteamVR service or robot hardware is accessed."""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vive_tianji import tracker_input


RUNNING_OK = 200
TRACKER = 3
BASE_STATION = 4


def matrix34(x=0., y=0., z=0., yaw=0.):
    c, s = math.cos(yaw), math.sin(yaw)
    return SimpleNamespace(m=[[c, -s, 0., x], [s, c, 0., y], [0., 0., 1., z]])


def pose(matrix=None, *, connected=True, valid=True, result=RUNNING_OK):
    return SimpleNamespace(bDeviceIsConnected=connected, bPoseIsValid=valid,
                           eTrackingResult=result,
                           mDeviceToAbsoluteTracking=matrix or matrix34())


def frame(index_to_serial, *, reference=None, left_pose=None, right_pose=None,
          base_pose=None, base_poses=None):
    rows = [pose(connected=False, valid=False) for _ in range(8)]
    classes = {}
    serials = {}
    for index, serial in index_to_serial.items():
        classes[index] = TRACKER
        serials[index] = serial
        rows[index] = left_pose if serial == 'LEFT' else right_pose
        if rows[index] is None:
            rows[index] = pose()
    if base_pose is not None:
        classes[7] = BASE_STATION
        serials[7] = 'LHB-BASE'
        rows[7] = base_pose
    for index, (serial, base) in (base_poses or {}).items():
        classes[index] = BASE_STATION
        serials[index] = serial
        rows[index] = base
    return {'poses': rows, 'classes': classes, 'serials': serials,
            'reference': reference or matrix34()}


class FakeSystem:
    def __init__(self, frames):
        self.frames = list(frames)
        self.current = None
        self.pose_calls = 0

    def getDeviceToAbsoluteTrackingPose(self, universe, prediction_s, count):
        assert (universe, prediction_s, count) == (1, 0, 8)
        self.pose_calls += 1
        self.current = self.frames.pop(0)
        return self.current['poses']

    def getRawZeroPoseToStandingAbsoluteTrackingPose(self):
        return self.current['reference']

    def getTrackedDeviceClass(self, index):
        return self.current['classes'].get(index, 0)

    def getStringTrackedDeviceProperty(self, index, prop):
        if index == 0 and prop == 1000:
            return 'null'
        assert prop == 1002
        return self.current['serials'][index]


class TrackerPairTests(unittest.TestCase):
    def make_pair(self, frames):
        self.system = FakeSystem(frames)
        self.init_calls = []
        self.shutdown_calls = []
        fake = SimpleNamespace(
            VRApplication_Other=0, TrackingUniverseStanding=1,
            k_unMaxTrackedDeviceCount=8,
            TrackingResult_Running_OK=RUNNING_OK,
            TrackedDeviceClass_GenericTracker=TRACKER,
            TrackedDeviceClass_TrackingReference=BASE_STATION,
            Prop_SerialNumber_String=1002,
            Prop_TrackingSystemName_String=1000,
            init=lambda application: self.init_calls.append(application),
            VRSystem=lambda: self.system,
            VRChaperone=lambda: SimpleNamespace(getCalibrationState=lambda: 201),
            shutdown=lambda: self.shutdown_calls.append(True),
        )
        self.addCleanup(patch.stopall)
        patch.object(tracker_input, 'openvr', fake).start()
        pair = tracker_input.TrackerPair('LEFT', 'RIGHT')
        self.addCleanup(pair.close)
        return pair

    def test_chaperone_status_reads_headless_driver_without_using_tracker_frames(self):
        pair = self.make_pair([])
        self.assertEqual(pair.chaperone_status(),
                         {'state': 201, 'hmd_tracking_system': 'null'})
        self.assertEqual(self.system.pose_calls, 0)
        pair.close()
        with self.assertRaisesRegex(RuntimeError, '已关闭'):
            pair.chaperone_status()

    def test_both_poses_share_one_snapshot_and_serials_can_change_index(self):
        pair = self.make_pair([
            frame({1: 'LEFT', 3: 'RIGHT'}, left_pose=pose(matrix34(x=.1)),
                  right_pose=pose(matrix34(x=-.2)), base_pose=pose()),
            frame({3: 'LEFT', 1: 'RIGHT'}, left_pose=pose(matrix34(x=.3)),
                  right_pose=pose(matrix34(x=-.4)), base_pose=pose()),
        ])
        self.assertIsNone(pair.reference_matrix)
        first, second = pair.sample(), pair.sample()
        self.assertEqual(self.system.pose_calls, 2)
        self.assertEqual(self.init_calls, [0])
        self.assertFalse(first.reference_changed)
        self.assertFalse(second.reference_changed)
        self.assertEqual((first.base_station_count, second.base_station_count), (1, 1))
        self.assertAlmostEqual(first.poses['left'][0, 3], .1)
        self.assertAlmostEqual(first.poses['right'][0, 3], -.2)
        self.assertAlmostEqual(second.poses['left'][0, 3], .3)
        self.assertAlmostEqual(second.poses['right'][0, 3], -.4)
        self.assertGreater(second.time_s, first.time_s)
        copied = pair.reference_matrix
        copied[0, 3] = 9
        self.assertEqual(pair.reference_matrix[0, 3], 0)

    def test_bad_tracking_or_matrix_is_never_exposed_as_valid_pose(self):
        bad_orthogonal = matrix34()
        bad_orthogonal.m[0][0] = 2
        pair = self.make_pair([
            frame({1: 'LEFT', 2: 'RIGHT'},
                  left_pose=pose(result=1), right_pose=pose(bad_orthogonal),
                  base_pose=pose(result=1)),
            frame({1: 'LEFT', 2: 'RIGHT'},
                  left_pose=pose(matrix34(x=float('nan'))),
                  right_pose=pose(connected=False)),
        ])
        first, second = pair.sample(), pair.sample()
        self.assertIsNone(first.poses['left'])
        self.assertIsNone(first.poses['right'])
        self.assertTrue(first.states['left']['connected'])
        self.assertFalse(first.states['left']['valid'])
        self.assertEqual(first.states['left']['tracking_result'], 1)
        self.assertEqual(first.base_station_count, 0)
        self.assertIsNone(second.poses['left'])
        self.assertFalse(second.states['right']['connected'])

    def test_reference_change_latches_until_explicit_reanchor_ack(self):
        pair = self.make_pair([
            frame({1: 'LEFT', 2: 'RIGHT'}),
            frame({1: 'LEFT', 2: 'RIGHT'}, reference=matrix34(x=.0001)),
            frame({1: 'LEFT', 2: 'RIGHT'}, reference=matrix34(x=.02)),
            frame({1: 'LEFT', 2: 'RIGHT'}, reference=matrix34(x=.02)),
            frame({1: 'LEFT', 2: 'RIGHT'}, reference=matrix34(x=.02)),
        ])
        self.assertFalse(pair.sample().reference_changed)
        self.assertFalse(pair.sample().reference_changed)
        reset = pair.sample()
        self.assertTrue(reset.reference_changed)
        self.assertIsNone(reset.poses['left'])
        self.assertIsNone(reset.poses['right'])
        self.assertAlmostEqual(pair.reference_matrix[0, 3], .02)
        self.assertTrue(pair.sample().reference_changed)
        pair.acknowledge_reference_change()
        resumed = pair.sample()
        self.assertFalse(resumed.reference_changed)
        self.assertTrue(resumed.states['left']['valid'])
        self.assertIsNotNone(resumed.poses['right'])

    def test_invalid_reference_at_start_does_not_claim_a_reset(self):
        invalid = matrix34()
        invalid.m[0][0] = float('nan')
        pair = self.make_pair([frame({1: 'LEFT', 2: 'RIGHT'}, reference=invalid)])
        sample = pair.sample()
        self.assertFalse(sample.reference_changed)
        self.assertIsNone(pair.reference_matrix)
        self.assertIsNone(sample.poses['left'])
        with self.assertRaises(RuntimeError):
            pair.acknowledge_reference_change()

    def test_duplicate_serial_fails_closed_and_close_is_idempotent(self):
        pair = self.make_pair([frame({1: 'LEFT', 2: 'LEFT', 3: 'RIGHT'})])
        sample = pair.sample()
        self.assertIsNone(sample.poses['left'])
        self.assertFalse(sample.states['left']['valid'])
        self.assertTrue(sample.states['right']['valid'])
        pair.close()
        pair.close()
        self.assertEqual(self.shutdown_calls, [True])
        with self.assertRaises(RuntimeError):
            pair.sample()

    def test_named_base_baseline_comes_from_the_same_standing_snapshot(self):
        first, second = tracker_input.BASE_STATION_SERIALS
        pair = self.make_pair([
            frame({1: 'LEFT', 2: 'RIGHT'}, base_poses={
                6: (first, pose(matrix34(x=0.))),
                7: (second, pose(matrix34(x=1.7))),
            }),
            frame({1: 'LEFT', 2: 'RIGHT'}, base_poses={
                6: (second, pose(matrix34(x=1.7))),
                7: (first, pose(matrix34(x=0.))),
            }),
            frame({1: 'LEFT', 2: 'RIGHT'}, base_poses={
                6: (first, pose(matrix34(x=0.))),
                7: (second, pose(matrix34(x=.129))),
            }),
            frame({1: 'LEFT', 2: 'RIGHT'}, base_poses={
                6: (first, pose(matrix34(x=0.))),
                7: (second, pose(matrix34(x=1.7), valid=False)),
            }),
        ])
        a, b, stale, missing = (pair.sample() for _ in range(4))
        self.assertAlmostEqual(a.base_station_baseline_m, 1.7)
        self.assertEqual(set(a.base_station_poses), set(tracker_input.BASE_STATION_SERIALS))
        self.assertAlmostEqual(a.base_station_poses[second][0, 3], 1.7)
        self.assertAlmostEqual(b.base_station_baseline_m, 1.7)
        self.assertAlmostEqual(stale.base_station_baseline_m, .129)
        self.assertIsNone(missing.base_station_baseline_m)
        self.assertEqual(missing.base_station_count, 1)


if __name__ == '__main__':
    unittest.main()
