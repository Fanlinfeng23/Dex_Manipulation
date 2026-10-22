"""Read-only, same-frame OpenVR sampling for two serial-numbered Vive Trackers.

The matrices are in OpenVR's *standing* tracking space. This module does not
infer a robot frame, calibrate a wrist mount, or import a robot control SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import openvr


_SIDES = ("left", "right")
BASE_STATION_SERIALS = ("LHB-BDC4BE18", "LHB-8DCC9AA0")
_REFERENCE_TRANSLATION_CHANGE_M = 0.002
_REFERENCE_ROTATION_CHANGE_RAD = 0.002


@dataclass(frozen=True)
class PairSample:
    time_s: float
    poses: dict[str, np.ndarray | None]
    states: dict[str, dict[str, bool | int | None]]
    reference_changed: bool
    base_station_count: int
    base_station_baseline_m: float | None = None
    base_station_poses: dict[str, np.ndarray] = field(default_factory=dict)


def _rigid_matrix(matrix34) -> np.ndarray | None:
    """Convert and validate an OpenVR 3x4 rigid transform, in double precision."""
    try:
        rows = np.asarray(matrix34.m, dtype=np.float64)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if rows.shape != (3, 4) or not np.isfinite(rows).all():
        return None
    rotation = rows[:, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=0)
            or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-3):
        return None
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :] = rows
    return matrix


def _reference_moved(previous: np.ndarray, current: np.ndarray) -> bool:
    if np.linalg.norm(current[:3, 3] - previous[:3, 3]) > _REFERENCE_TRANSLATION_CHANGE_M:
        return True
    delta = previous[:3, :3].T @ current[:3, :3]
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return bool(np.arccos(cosine) > _REFERENCE_ROTATION_CHANGE_RAD)


class TrackerPair:
    """Own one OpenVR client and sample both trackers from one pose snapshot.

    ``reference_changed`` latches after a standing-space reset until the caller
    has stopped motion, re-anchored both hands, and calls
    :meth:`acknowledge_reference_change`. While latched, no poses are usable.
    """

    def __init__(self, left_serial: str, right_serial: str):
        if (not isinstance(left_serial, str) or not left_serial.strip()
                or not isinstance(right_serial, str) or not right_serial.strip()
                or left_serial == right_serial):
            raise ValueError("需要两个不同且非空的 Tracker 序列号")
        self.serials = {"left": left_serial, "right": right_serial}
        self._reference: np.ndarray | None = None
        self._reference_change_latched = False
        self._closed = True
        openvr.init(openvr.VRApplication_Other)
        try:
            self._vr = openvr.VRSystem()
        except BaseException:
            openvr.shutdown()
            raise
        self._closed = False

    @property
    def reference_matrix(self) -> np.ndarray | None:
        """Current raw-to-standing transform, copied for calibration records."""
        return None if self._reference is None else self._reference.copy()

    def chaperone_status(self) -> dict[str, int | str | None]:
        """Read SteamVR's room state and the active HMD tracking driver."""
        if self._closed:
            raise RuntimeError("OpenVR Tracker 采样器已关闭")
        try:
            state = int(openvr.VRChaperone().getCalibrationState())
            hmd_driver = self._vr.getStringTrackedDeviceProperty(
                0, openvr.Prop_TrackingSystemName_String)
        except Exception as exc:
            raise RuntimeError(f"无法读取 SteamVR Chaperone 状态: {exc}") from exc
        return {"state": state, "hmd_tracking_system": hmd_driver}

    def sample(self) -> PairSample:
        if self._closed:
            raise RuntimeError("OpenVR Tracker 采样器已关闭")
        try:
            # One snapshot for both hands; separate calls would mix tracking frames.
            raw_poses = self._vr.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)
            stamp = time.perf_counter()
            raw_reference = self._vr.getRawZeroPoseToStandingAbsoluteTrackingPose()
        except Exception as exc:
            raise RuntimeError(f"OpenVR 采样失败: {exc}") from exc

        reference = _rigid_matrix(raw_reference)
        if reference is None:
            if self._reference is not None:
                self._reference_change_latched = True
        elif self._reference is None:
            self._reference = reference
        else:
            if _reference_moved(self._reference, reference):
                self._reference_change_latched = True
            self._reference = reference

        poses: dict[str, np.ndarray | None] = {side: None for side in _SIDES}
        states: dict[str, dict[str, bool | int | None]] = {
            side: {"connected": False, "valid": False, "tracking_result": None}
            for side in _SIDES
        }
        serial_to_side = {serial: side for side, serial in self.serials.items()}
        observed: set[str] = set()
        base_station_count = 0
        base_poses: dict[str, np.ndarray] = {}
        duplicate_bases: set[str] = set()
        for index, raw_pose in enumerate(raw_poses):
            if not bool(raw_pose.bDeviceIsConnected):
                continue
            try:
                device_class = self._vr.getTrackedDeviceClass(index)
            except Exception:
                continue
            if device_class == openvr.TrackedDeviceClass_TrackingReference:
                matrix = (_rigid_matrix(raw_pose.mDeviceToAbsoluteTracking)
                          if bool(raw_pose.bPoseIsValid) else None)
                if (int(raw_pose.eTrackingResult) == openvr.TrackingResult_Running_OK
                        and matrix is not None):
                    base_station_count += 1
                    try:
                        serial = self._vr.getStringTrackedDeviceProperty(
                            index, openvr.Prop_SerialNumber_String)
                    except Exception:
                        serial = None
                    if serial in BASE_STATION_SERIALS:
                        if serial in base_poses:
                            duplicate_bases.add(serial)
                        else:
                            base_poses[serial] = matrix
                continue
            if device_class != openvr.TrackedDeviceClass_GenericTracker:
                continue
            try:
                serial = self._vr.getStringTrackedDeviceProperty(
                    index, openvr.Prop_SerialNumber_String)
            except Exception:
                continue
            side = serial_to_side.get(serial)
            if side is None:
                continue
            # A duplicate serial must never silently select an arbitrary pose.
            if side in observed:
                poses[side] = None
                states[side]["valid"] = False
                continue
            observed.add(side)
            result = int(raw_pose.eTrackingResult)
            raw_valid = bool(raw_pose.bPoseIsValid)
            matrix = _rigid_matrix(raw_pose.mDeviceToAbsoluteTracking) if raw_valid else None
            usable = (raw_valid and result == openvr.TrackingResult_Running_OK
                      and matrix is not None and reference is not None
                      and not self._reference_change_latched)
            states[side] = {"connected": True, "valid": bool(usable),
                            "tracking_result": result}
            if usable:
                poses[side] = matrix

        baseline = None
        if not duplicate_bases and len(base_poses) == len(BASE_STATION_SERIALS):
            first, second = (base_poses[serial][:3, 3] for serial in BASE_STATION_SERIALS)
            baseline = float(np.linalg.norm(first - second))
        return PairSample(stamp, poses, states, self._reference_change_latched,
                          base_station_count, baseline, base_poses)

    def acknowledge_reference_change(self) -> None:
        """Release the latch only after the downstream controller re-anchors."""
        if self._reference is None:
            raise RuntimeError("尚无有效的 OpenVR 站立坐标参考")
        self._reference_change_latched = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            openvr.shutdown()

    def __enter__(self) -> "TrackerPair":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
