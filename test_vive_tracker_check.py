"""Hardware-free checks for the read-only four-device tracking gate."""

import contextlib
import io
import sys
import types
import unittest
from unittest.mock import patch

import vive_tracker_check as checker


class FakeVR:
    def __init__(self, bad_by_frame=None, standing_bad_by_frame=None,
                 standing_distance_by_frame=None):
        self.bad_by_frame = bad_by_frame or {}
        self.standing_bad_by_frame = standing_bad_by_frame or {}
        self.standing_distance_by_frame = standing_distance_by_frame or {}
        self.raw_calls = 0
        self.serials = checker.TRACKERS + checker.BASE_STATIONS

    def getDeviceToAbsoluteTrackingPose(self, universe, _prediction, _maximum):
        if universe == checker.openvr.TrackingUniverseRawAndUncalibrated:
            frame = self.raw_calls
            self.raw_calls += 1
        else:
            frame = self.raw_calls - 1
        bad = (self.bad_by_frame if universe == checker.openvr.TrackingUniverseRawAndUncalibrated
               else self.standing_bad_by_frame).get(frame)
        distance = self.standing_distance_by_frame.get(frame, 1.5)
        return [
            types.SimpleNamespace(
                bDeviceIsConnected=True,
                bPoseIsValid=serial != bad,
                eTrackingResult=checker.openvr.TrackingResult_Running_OK,
                mDeviceToAbsoluteTracking=types.SimpleNamespace(m=[
                    [1.0, 0.0, 0.0, distance if serial == checker.BASE_STATIONS[1] else 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]),
            )
            for serial in self.serials
        ]

    def getTrackedDeviceClass(self, index):
        return (checker.openvr.TrackedDeviceClass_GenericTracker if index < 2
                else checker.openvr.TrackedDeviceClass_TrackingReference)

    def getStringTrackedDeviceProperty(self, index, _property):
        return self.serials[index]


class TrackerCheckTest(unittest.TestCase):
    def run_check(self, bad_by_frame, standing_bad_by_frame=None,
                  standing_distance_by_frame=None, expected_baseline_m=None):
        output = io.StringIO()
        argv = ["vive_tracker_check.py", "--seconds", "1"]
        if expected_baseline_m is not None:
            argv.extend(["--expected-baseline-m", str(expected_baseline_m)])
        with (patch.object(checker, "_user_vrserver_running", return_value=True),
              patch.object(checker.openvr, "init"),
              patch.object(checker.openvr, "shutdown"),
              patch.object(checker.openvr, "VRSystem", return_value=FakeVR(
                  bad_by_frame, standing_bad_by_frame,
                  standing_distance_by_frame)),
              patch.object(checker.openvr, "VRChaperone") as chaperone,
              patch.object(checker.time, "sleep"),
              patch.object(sys, "argv", argv),
              contextlib.redirect_stdout(output)):
            chaperone.return_value.getCalibrationState.return_value = 1
            result = checker.main()
        return result, output.getvalue()

    def test_absent_server_exits_before_openvr_init(self):
        output = io.StringIO()
        with (patch.object(checker, "_user_vrserver_running", return_value=False),
              patch.object(checker.openvr, "init") as init,
              patch.object(sys, "argv", ["vive_tracker_check.py", "--seconds", "1"]),
              contextlib.redirect_stdout(output)):
            result = checker.main()
        self.assertEqual(result, 2)
        init.assert_not_called()
        self.assertIn("通过 Steam 启动 SteamVR 后再检查", output.getvalue())

    def test_fails_when_second_base_station_never_has_a_pose(self):
        result, output = self.run_check({frame: checker.BASE_STATIONS[1]
                                         for frame in range(20)})
        self.assertEqual(result, 2)
        self.assertIn(f"Base station {checker.BASE_STATIONS[1]}", output)
        self.assertIn("Both trackers and both base stations usable in the same frame: 0/20", output)

    def test_requires_simultaneous_poses_even_if_each_base_has_95_percent(self):
        result, output = self.run_check({0: checker.BASE_STATIONS[0],
                                         1: checker.BASE_STATIONS[1]})
        self.assertEqual(result, 2)
        self.assertIn("same frame: 18/20", output)

    def test_rejects_base_with_raw_pose_but_no_standing_pose(self):
        result, output = self.run_check(
            {}, {frame: checker.BASE_STATIONS[1] for frame in range(20)})
        self.assertEqual(result, 2)
        self.assertIn("raw 6DoF 20/20, standing 6DoF 0/20", output)
        self.assertIn("same frame: 0/20", output)

    def test_passes_when_all_four_are_simultaneously_valid(self):
        result, output = self.run_check({})
        self.assertEqual(result, 0)
        self.assertIn("same frame: 20/20", output)

    def test_passes_at_exactly_95_percent_simultaneous_poses(self):
        result, output = self.run_check({0: checker.BASE_STATIONS[1]})
        self.assertEqual(result, 0)
        self.assertIn("same frame: 19/20", output)

    def test_reports_in_range_standing_baseline(self):
        result, output = self.run_check(
            {}, standing_distance_by_frame={0: 1.0, 1: 2.0})
        self.assertEqual(result, 0)
        self.assertIn(
            "Base station Standing baseline (20/20 valid frames): "
            "median 1.5000 m, min 1.0000 m, max 2.0000 m", output)

    def test_rejects_stale_13_cm_baseline_despite_four_valid_poses(self):
        result, output = self.run_check(
            {}, standing_distance_by_frame={frame: 0.1289
                                             for frame in range(20)})
        self.assertEqual(result, 2)
        self.assertIn("same frame: 20/20", output)
        self.assertIn("FAIL: Standing base-station baseline", output)
        self.assertIn("median 0.1289 m", output)

    def test_rejects_one_outlier_even_when_median_is_in_range(self):
        result, output = self.run_check(
            {}, standing_distance_by_frame={0: 0.5})
        self.assertEqual(result, 2)
        self.assertIn("median 1.5000 m, min 0.5000 m", output)

    def test_explicit_close_layout_passes_only_within_its_tolerance(self):
        distances = {frame: 0.240 for frame in range(20)}
        result, output = self.run_check(
            {}, standing_distance_by_frame=distances,
            expected_baseline_m=0.240)
        self.assertEqual(result, 0)
        self.assertIn("0.220–0.260 m", output)
        self.assertIn("read-only tracking test only", output)

        distances[0] = 0.270
        result, output = self.run_check(
            {}, standing_distance_by_frame=distances,
            expected_baseline_m=0.240)
        self.assertEqual(result, 2)
        self.assertIn("min 0.2400 m, max 0.2700 m", output)

    def test_close_layout_still_rejects_missing_station_and_old_baseline(self):
        distances = {frame: 0.240 for frame in range(20)}
        result, _ = self.run_check(
            {frame: checker.BASE_STATIONS[1] for frame in range(20)},
            standing_distance_by_frame=distances, expected_baseline_m=0.240)
        self.assertEqual(result, 2)
        result, output = self.run_check(
            {}, standing_distance_by_frame={frame: 0.1289 for frame in range(20)},
            expected_baseline_m=0.240)
        self.assertEqual(result, 2)
        self.assertIn("median 0.1289 m", output)


if __name__ == "__main__":
    unittest.main()
