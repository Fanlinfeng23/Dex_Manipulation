#!/usr/bin/env python3
"""Estimate a controller-to-wrist pivot offset from a Quest trajectory JSONL.

Record with tools/record_session.py while keeping each anatomical wrist nearly
still and rotating the attached controller about at least two different axes.
This estimates translation only; it cannot determine wrist orientation or fix
Quest optical tracking loss. No robot connection or command is made here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = APP_ROOT / "configs" / "hardware_dual_arm.json"
SIDES = ("left", "right")
MIN_SAMPLES = 60
MIN_DURATION_S = 3.0
MIN_MAX_ROTATION_RAD = math.radians(25)
MIN_SECOND_AXIS_RMS_RAD = math.radians(6)
MIN_DESIGN_SINGULAR = 0.10  # singular value of centered R / sqrt(N)
MAX_CONDITION = 20.0
MAX_RMSE_M = 0.012
MAX_P95_M = 0.025
MAX_RESIDUAL_M = 0.050
MAX_OFFSET_M = 0.35


class CalibrationError(ValueError):
    """The capture cannot support a trustworthy wrist-pivot calibration."""


def _finite_vector(value: object, count: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != count:
        raise CalibrationError(f"{label}: expected {count} numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise CalibrationError(f"{label}: expected finite numbers")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise CalibrationError(f"{label}: expected finite numbers")
    return result


def _read_pose(controller: dict, label: str) -> tuple[np.ndarray, np.ndarray]:
    position = _finite_vector(controller.get("position"), 3, f"{label}.position")
    quaternion = _finite_vector(controller.get("quaternion"), 4, f"{label}.quaternion")
    norm = float(np.linalg.norm(quaternion))
    if np.max(np.abs(position)) > 30 or not 0.95 <= norm <= 1.05:
        raise CalibrationError(f"{label}: implausible position or non-unit quaternion")
    return position, Rotation.from_quat(quaternion / norm).as_matrix()


def load_samples(path: Path) -> tuple[dict[str, list[tuple[np.ndarray, np.ndarray]]], dict]:
    """Use unique, fresh, focused Quest input frames with both controllers tracked."""
    if not path.is_file():
        raise CalibrationError(f"recording does not exist: {path}")
    poses = {side: [] for side in SIDES}
    sources: set[tuple[str, int]] = set()
    seqs: set[tuple[str, int, int]] = set()
    times: list[float] = []
    counts = {"sample_rows": 0, "eligible_rows": 0}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibrationError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict) or row.get("kind") != "sample":
                continue
            counts["sample_rows"] += 1
            if row.get("new_input") is not True or row.get("input_fresh") is not True:
                continue
            state = row.get("state")
            if not isinstance(state, dict) or state.get("source") != "quest":
                continue
            frame = row.get("input")
            if not isinstance(frame, dict):
                raise CalibrationError(f"line {line_number}: fresh Quest row has no input frame")
            if frame.get("focused") is not True:
                continue
            safety = frame.get("safety") or {}
            if not isinstance(safety, dict):
                raise CalibrationError(f"line {line_number}: invalid input safety flags")
            if safety.get("tracking_lost") or safety.get("focus_lost"):
                continue
            controllers = frame.get("controllers")
            if not isinstance(controllers, dict):
                raise CalibrationError(f"line {line_number}: controllers missing")
            if any(not isinstance(controllers.get(side), dict) or
                   type(controllers[side].get("tracked")) is not bool for side in SIDES):
                raise CalibrationError(f"line {line_number}: invalid controller tracking flags")
            if not all(controllers[side]["tracked"] for side in SIDES):
                continue
            session, reference, seq, captured = (frame.get(key) for key in
                                                 ("session", "reference", "seq", "captured_s"))
            if (not isinstance(session, str) or not session or type(reference) is not int or
                    reference < 0 or type(seq) is not int or seq < 0 or
                    isinstance(captured, bool) or not isinstance(captured, (int, float)) or
                    not math.isfinite(captured) or captured < 0):
                raise CalibrationError(f"line {line_number}: invalid Quest session/sequence/time")
            source = (session, reference)
            key = (session, reference, seq)
            if key in seqs:
                raise CalibrationError(f"line {line_number}: duplicate Quest input sequence")
            seqs.add(key)
            sources.add(source)
            if len(sources) > 1:
                raise CalibrationError("capture crosses a Quest session or reference-space reset; record again")
            for side in SIDES:
                poses[side].append(_read_pose(controllers[side], f"line {line_number} {side}"))
            times.append(float(captured))
            counts["eligible_rows"] += 1
    if len(times) < MIN_SAMPLES:
        raise CalibrationError(f"only {len(times)} usable dual-controller frames; need at least {MIN_SAMPLES}")
    duration = max(times) - min(times)
    if duration < MIN_DURATION_S:
        raise CalibrationError(f"capture spans {duration:.2f} s; need at least {MIN_DURATION_S:.1f} s")
    counts["duration_s"] = round(duration, 3)
    counts["session"] = next(iter(sources))[0]
    counts["reference"] = next(iter(sources))[1]
    return poses, counts


def fit_pivot(samples: list[tuple[np.ndarray, np.ndarray]], side: str) -> dict:
    """Solve p_i + R_i @ t = c, with t in the controller's local grip frame."""
    if len(samples) < MIN_SAMPLES:
        raise CalibrationError(f"{side}: need at least {MIN_SAMPLES} valid samples")
    positions = np.stack([sample[0] for sample in samples])
    rotations = np.stack([sample[1] for sample in samples])
    if positions.shape != (len(samples), 3) or rotations.shape != (len(samples), 3, 3):
        raise CalibrationError(f"{side}: malformed samples")
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise CalibrationError(f"{side}: nonfinite pose")

    # One-axis spinning leaves one component of the pivot unobservable. Require
    # substantive rotations about two axes before considering residual quality.
    relative_rotations = Rotation.from_matrix(rotations[0]).inv() * Rotation.from_matrix(rotations)
    relative = relative_rotations.as_rotvec()
    max_rotation = float(np.max(relative_rotations.magnitude()))
    relative -= relative.mean(axis=0)
    angle_singular = np.linalg.svd(relative, compute_uv=False) / math.sqrt(len(samples))
    if (max_rotation < MIN_MAX_ROTATION_RAD or
            angle_singular[1] < MIN_SECOND_AXIS_RMS_RAD):
        raise CalibrationError(
            f"{side}: insufficient multi-axis wrist rotation; keep the wrist fixed and rotate "
            "through at least 25 degrees around two distinct axes")

    mean_position = positions.mean(axis=0)
    mean_rotation = rotations.mean(axis=0)
    design = (rotations - mean_rotation).reshape(-1, 3)
    displacement = -(positions - mean_position).reshape(-1)
    singular = np.linalg.svd(design, compute_uv=False)
    min_singular = float(singular[-1] / math.sqrt(len(samples)))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
    if min_singular < MIN_DESIGN_SINGULAR or condition > MAX_CONDITION:
        raise CalibrationError(
            f"{side}: ill-conditioned rotation set (min singular {min_singular:.3f}, "
            f"condition {condition:.1f}); rotate about more distinct axes")
    offset, _, rank, _ = np.linalg.lstsq(design, displacement, rcond=None)
    if rank != 3 or not np.isfinite(offset).all():
        raise CalibrationError(f"{side}: pivot fit is rank deficient")
    length = float(np.linalg.norm(offset))
    if length > MAX_OFFSET_M:
        raise CalibrationError(
            f"{side}: estimated grip-to-wrist offset {length * 1000:.1f} mm exceeds "
            f"{MAX_OFFSET_M * 1000:.0f} mm")

    pivots = positions + np.einsum("nij,j->ni", rotations, offset)
    world_pivot = pivots.mean(axis=0)
    residuals = np.linalg.norm(pivots - world_pivot, axis=1)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    p95 = float(np.percentile(residuals, 95))
    max_residual = float(np.max(residuals))
    if rmse > MAX_RMSE_M or p95 > MAX_P95_M or max_residual > MAX_RESIDUAL_M:
        raise CalibrationError(
            f"{side}: wrist pivot moved or tracking jumped "
            f"(RMS {rmse * 1000:.1f}, p95 {p95 * 1000:.1f}, max {max_residual * 1000:.1f} mm); "
            "capture again while keeping the wrist in place")
    return {
        "controller_to_wrist_m": [round(float(value), 6) for value in offset],
        "offset_length_mm": round(length * 1000, 2),
        "pivot_rmse_mm": round(rmse * 1000, 2),
        "pivot_p95_mm": round(p95 * 1000, 2),
        "pivot_max_residual_mm": round(max_residual * 1000, 2),
        "max_rotation_deg": round(math.degrees(max_rotation), 1),
        "second_axis_rms_deg": round(math.degrees(float(angle_singular[1])), 1),
        "design_min_singular": round(min_singular, 4),
        "design_condition": round(condition, 2),
        "samples": len(samples),
    }


def write_config(path: Path, results: dict[str, dict]) -> Path:
    """Back up the exact original and atomically replace only requested offsets."""
    original = path.read_bytes()
    config = json.loads(original)
    if not isinstance(config, dict) or not isinstance(config.get("arms"), dict):
        raise CalibrationError(f"invalid arm configuration: {path}")
    for side, fit in results.items():
        arm = config["arms"].get(side)
        if not isinstance(arm, dict):
            raise CalibrationError(f"config has no {side} arm")
        _finite_vector(arm.get("controller_to_wrist_m"), 3, f"config {side} offset")
        _finite_vector(arm.get("controller_to_wrist_xyzw"), 4, f"config {side} quaternion")
        arm["controller_to_wrist_m"] = fit["controller_to_wrist_m"]
    new_bytes = (json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%dT%H%M%S')}-{time.time_ns()}")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            tmp_path = Path(stream.name)
            stream.write(new_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp_path, stat.S_IMODE(path.stat().st_mode))
        if path.read_bytes() != original:
            raise CalibrationError("config changed during calibration; refusing to overwrite it")
        with backup.open("xb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(backup, stat.S_IMODE(path.stat().st_mode))
        if path.read_bytes() != original:
            raise CalibrationError("config changed while its backup was created; refusing to overwrite it")
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return backup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="tools/record_session.py JSONL output")
    parser.add_argument("--side", choices=("both", "left", "right"), default="both")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--write-config", action="store_true",
                        help="back up and atomically update selected controller_to_wrist_m fields")
    args = parser.parse_args(argv)
    try:
        poses, counts = load_samples(args.recording)
        selected = SIDES if args.side == "both" else (args.side,)
        results = {side: fit_pivot(poses[side], side) for side in selected}
        report = {"recording": str(args.recording), "capture": counts, "fits": results,
                  "written": False}
        if args.write_config:
            backup = write_config(args.config, results)
            report.update({"written": True, "config": str(args.config), "backup": str(backup)})
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (CalibrationError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"标定拒绝：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
