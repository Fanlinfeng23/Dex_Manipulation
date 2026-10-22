#!/usr/bin/env python3
"""Read-only SteamVR pose check for both wrist trackers and base stations.

Run with: ./.vive-venv/bin/python vive_tracker_check.py --seconds 10
This program never connects to or commands the robot.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import openvr


TRACKERS = ("LHR-8180F66F", "LHR-00037B1A")
BASE_STATIONS = ("LHB-BDC4BE18", "LHB-8DCC9AA0")
SAMPLE_HZ = 20
MIN_BASELINE_M = 0.8
MAX_BASELINE_M = 5.0
EXPECTED_BASELINE_TOLERANCE_M = 0.02


def _user_vrserver_running():
    """Inspect only this user's live processes; never start SteamVR here."""
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            if (process / 'comm').read_text().strip() != 'vrserver':
                continue
            state = (process / 'stat').read_text().rpartition(') ')[2][:1]
            if state and state not in ('Z', 'X'):
                return True
        except OSError:
            # Processes can exit between inspection steps.
            continue
    return False


def _standing_position(pose):
    """Return a valid Standing-space translation, if OpenVR supplies one."""
    try:
        matrix = pose.mDeviceToAbsoluteTracking.m
        position = tuple(float(matrix[row][3]) for row in range(3))
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    return position if all(math.isfinite(value) for value in position) else None


def sample(vr, universe, seconds):
    totals = {
        serial: {
            "connected": 0,
            "valid": 0,
            "standing_valid": 0,
            "running_ok": 0,
            "usable": 0,
            "results": Counter(),
        }
        for serial in TRACKERS
    }
    stations = {
        serial: {"connected": 0, "valid": 0, "standing_valid": 0,
                 "tracking_ok": 0, "usable": 0, "results": Counter()}
        for serial in BASE_STATIONS
    }
    all_four_usable = 0
    baseline_m = []
    count = max(1, round(seconds * SAMPLE_HZ))
    for _ in range(count):
        poses = vr.getDeviceToAbsoluteTrackingPose(
            universe, 0, openvr.k_unMaxTrackedDeviceCount
        )
        standing_poses = vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )
        # Gather by serial first so a missing or duplicated device cannot count as
        # usable, and every device contributes at most once to each sampled frame.
        observations = {}
        duplicates = set()
        for index, pose in enumerate(poses):
            device_class = vr.getTrackedDeviceClass(index)
            if device_class not in (
                openvr.TrackedDeviceClass_GenericTracker,
                openvr.TrackedDeviceClass_TrackingReference,
            ):
                continue
            serial = vr.getStringTrackedDeviceProperty(
                index, openvr.Prop_SerialNumber_String
            )
            if serial not in totals and serial not in stations:
                continue
            if ((serial in totals and device_class != openvr.TrackedDeviceClass_GenericTracker)
                    or (serial in stations and
                        device_class != openvr.TrackedDeviceClass_TrackingReference)):
                continue
            if serial in observations:
                duplicates.add(serial)
            else:
                observations[serial] = (pose, standing_poses[index])
        frame_usable = {}
        standing_positions = {}
        for serial, observation in observations.items():
            if serial in duplicates:
                frame_usable[serial] = False
                continue
            pose, standing_pose = observation
            if serial in totals:
                state = totals[serial]
                state["connected"] += bool(pose.bDeviceIsConnected)
                state["valid"] += bool(pose.bPoseIsValid)
                state["standing_valid"] += bool(standing_pose.bPoseIsValid)
                state["running_ok"] += (
                    int(pose.eTrackingResult) == openvr.TrackingResult_Running_OK
                    and int(standing_pose.eTrackingResult)
                    == openvr.TrackingResult_Running_OK
                )
                usable = (
                    bool(pose.bDeviceIsConnected)
                    and bool(pose.bPoseIsValid)
                    and bool(standing_pose.bPoseIsValid)
                    and int(pose.eTrackingResult) == openvr.TrackingResult_Running_OK
                    and int(standing_pose.eTrackingResult)
                    == openvr.TrackingResult_Running_OK
                )
                state["usable"] += usable
                frame_usable[serial] = usable
                state["results"][int(pose.eTrackingResult)] += 1
            else:
                state = stations[serial]
                connected = bool(pose.bDeviceIsConnected)
                valid = bool(pose.bPoseIsValid)
                standing_valid = bool(standing_pose.bPoseIsValid)
                tracking_ok = (
                    int(pose.eTrackingResult) == openvr.TrackingResult_Running_OK
                    and int(standing_pose.eTrackingResult)
                    == openvr.TrackingResult_Running_OK
                )
                usable = connected and valid and standing_valid and tracking_ok
                state["connected"] += connected
                state["valid"] += valid
                state["standing_valid"] += standing_valid
                state["tracking_ok"] += tracking_ok
                state["usable"] += usable
                state["results"][int(pose.eTrackingResult)] += 1
                frame_usable[serial] = usable
                if usable:
                    position = _standing_position(standing_pose)
                    if position is not None:
                        standing_positions[serial] = position
        if all(serial in standing_positions for serial in BASE_STATIONS):
            baseline_m.append(math.dist(*(standing_positions[serial]
                                          for serial in BASE_STATIONS)))
        all_four_usable += all(
            frame_usable.get(serial, False) for serial in (*TRACKERS, *BASE_STATIONS)
        )
        time.sleep(1 / SAMPLE_HZ)
    return count, totals, stations, all_four_usable, baseline_m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument(
        "--expected-baseline-m", type=float,
        help="read-only close-layout test: require every measured baseline within ±0.02 m",
    )
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60:
        parser.error("--seconds must be between 1 and 60")
    if args.expected_baseline_m is None:
        min_baseline_m, max_baseline_m = MIN_BASELINE_M, MAX_BASELINE_M
    else:
        expected = args.expected_baseline_m
        if not math.isfinite(expected) or expected <= EXPECTED_BASELINE_TOLERANCE_M:
            parser.error("--expected-baseline-m must be finite and greater than 0.02")
        min_baseline_m = expected - EXPECTED_BASELINE_TOLERANCE_M
        max_baseline_m = expected + EXPECTED_BASELINE_TOLERANCE_M

    if not _user_vrserver_running():
        print("SteamVR 服务器未运行：请通过 Steam 启动 SteamVR 后再检查。")
        return 2

    openvr.init(openvr.VRApplication_Other)
    try:
        vr = openvr.VRSystem()
        chaperone = openvr.VRChaperone()
        count, totals, stations, all_four_usable, baseline_m = sample(
            vr, openvr.TrackingUniverseRawAndUncalibrated, args.seconds
        )
        print(f"OpenVR poses: {count} samples in {args.seconds:g} s")
        for serial, state in totals.items():
            print(
                f"{serial}: radio {state['connected']}/{count}, "
                f"raw 6DoF {state['valid']}/{count}, "
                f"standing 6DoF {state['standing_valid']}/{count}, "
                f"tracking OK {state['running_ok']}/{count}, "
                f"same-frame usable {state['usable']}/{count}, "
                f"tracking results {dict(state['results'])}"
            )
        for serial, state in stations.items():
            print(
                f"Base station {serial}: connected {state['connected']}/{count}, "
                f"raw 6DoF {state['valid']}/{count}, "
                f"standing 6DoF {state['standing_valid']}/{count}, "
                f"tracking OK {state['tracking_ok']}/{count}, "
                f"same-frame usable {state['usable']}/{count}, "
                f"tracking results {dict(state['results'])}"
            )
        if baseline_m:
            print(
                f"Base station Standing baseline ({len(baseline_m)}/{count} valid frames): "
                f"median {statistics.median(baseline_m):.4f} m, "
                f"min {min(baseline_m):.4f} m, max {max(baseline_m):.4f} m"
            )
        else:
            print("Base station Standing baseline: unavailable (no two valid positions in one frame)")
        print(f"Both trackers and both base stations usable in the same frame: {all_four_usable}/{count}")
        print(f"Chaperone calibration state: {int(chaperone.getCalibrationState())}")
        print("Base station counts are OpenVR pose validity, not proof of active optical contribution.")
        print("Raw coordinates are diagnostic only; do not drive the robot from them.")
        if all_four_usable < 0.95 * count:
            print("FAIL: both trackers and both named base stations need at least 95% simultaneous valid poses before any arm control.")
            return 2
        if len(baseline_m) < 0.95 * count:
            print("FAIL: insufficient same-frame Standing base-station positions to verify the physical layout.")
            return 2
        median_baseline = statistics.median(baseline_m)
        if (not min_baseline_m <= median_baseline <= max_baseline_m
                or any(not min_baseline_m <= distance <= max_baseline_m
                       for distance in baseline_m)):
            print(
                f"FAIL: Standing base-station baseline has a frame outside the expected "
                f"{min_baseline_m:.3f}–{max_baseline_m:.3f} m layout "
                f"(median {median_baseline:.4f} m, min {min(baseline_m):.4f} m, "
                f"max {max(baseline_m):.4f} m); SteamVR may still hold stale "
                "base-station geometry."
            )
            return 2
        if args.expected_baseline_m is None:
            print("PASS: four simultaneous poses and the Standing base-station baseline match the configured layout.")
        else:
            print("PASS: four simultaneous poses match the specified close-layout baseline "
                  f"({min_baseline_m:.3f}–{max_baseline_m:.3f} m); read-only tracking test only.")
        return 0
    finally:
        openvr.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
