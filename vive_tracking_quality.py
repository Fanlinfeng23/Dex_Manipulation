#!/usr/bin/env python3
"""Read-only, same-frame quality report for the two Tianji wrist Vive Trackers.

Examples:
  python3 vive_tracking_quality.py --mode static --seconds 20
  python3 vive_tracking_quality.py --mode sweep --seconds 30

This program uses only SteamVR/OpenVR poses. It never imports or commands a
robot SDK. A default-layout PASS means tracking is ready for calibration or
preview. A PASS with --expected-baseline-m applies only to this read-only
tracking test. Neither proves active optical contribution from both stations.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
SERIALS = {"left": "LHR-8180F66F", "right": "LHR-00037B1A"}
SAMPLE_HZ = 90.0
MIN_DURATION_S = 10.0
MAX_DURATION_S = 60.0
MIN_BASELINE_M = 0.8
MAX_BASELINE_M = 5.0
EXPECTED_BASELINE_TOLERANCE_M = 0.02
MIN_SIMULTANEOUS_VALID_RATIO = 0.98
MAX_INVALID_STREAK_FRAMES = 2
MAX_SAMPLE_INTERVAL_MS = 50.0
MIN_VALID_TAIL_S = 0.75


def _user_vrserver_running() -> bool:
    """Avoid OpenVR starting a bare vrserver outside the Steam Linux Runtime."""
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            if (process / "comm").read_text().strip() != "vrserver":
                continue
            state = (process / "stat").read_text().rpartition(") ")[2][:1]
            if state and state not in ("Z", "X"):
                return True
        except OSError:
            continue
    return False


def _longest_false_streak(values: list[bool]) -> int:
    longest = current = 0
    for valid in values:
        current = 0 if valid else current + 1
        longest = max(longest, current)
    return longest


def _valid_tail(values: list[bool]) -> int:
    tail = 0
    for valid in reversed(values):
        if not valid:
            break
        tail += 1
    return tail


def _percentile(values: list[float] | np.ndarray, p: float) -> float | None:
    return float(np.percentile(values, p)) if len(values) else None


def _static_jitter(poses: list[np.ndarray]) -> dict | None:
    """Pose displacement around a robust static centre; operator must hold still."""
    if not poses:
        return None
    positions = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float64)
    rotations = np.asarray([pose[:3, :3] for pose in poses], dtype=np.float64)
    centre_position = np.median(positions, axis=0)
    translation_mm = np.linalg.norm(positions - centre_position, axis=1) * 1000.0

    # The orthogonal Procrustes mean is a valid SO(3) centre even when matrix
    # elements have independent numerical noise. The residual is geodesic.
    u, _, vt = np.linalg.svd(rotations.mean(axis=0))
    correction = np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))])
    centre_rotation = u @ correction @ vt
    relative = np.einsum("ij,njk->nik", centre_rotation.T, rotations)
    cosines = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
                      -1.0, 1.0)
    rotation_deg = np.degrees(np.arccos(cosines))

    return {
        "valid_pose_count": len(poses),
        "position_centre_m": [float(x) for x in centre_position],
        "position_rms_mm": float(np.sqrt(np.mean(translation_mm ** 2))),
        "position_p95_mm": _percentile(translation_mm, 95),
        "orientation_rms_deg": float(np.sqrt(np.mean(rotation_deg ** 2))),
        "orientation_p95_deg": _percentile(rotation_deg, 95),
    }


def analyze(samples: list, mode: str, *, duration_s: float,
            baseline_limits_m: tuple[float, float] | None = None) -> dict:
    """Summarize PairSample records; pure math, no OpenVR or robot access."""
    if mode not in ("static", "sweep"):
        raise ValueError("mode must be static or sweep")
    if not samples:
        raise ValueError("no samples")
    if baseline_limits_m is None:
        baseline_limits_m = (MIN_BASELINE_M, MAX_BASELINE_M)
    try:
        baseline_min_m, baseline_max_m = (float(value) for value in baseline_limits_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("baseline_limits_m must contain two finite bounds") from exc
    if (not math.isfinite(baseline_min_m) or not math.isfinite(baseline_max_m)
            or baseline_min_m <= 0 or baseline_min_m >= baseline_max_m):
        raise ValueError("baseline_limits_m must contain two ordered positive bounds")
    close_layout_test = (baseline_min_m, baseline_max_m) != (MIN_BASELINE_M, MAX_BASELINE_M)

    count = len(samples)
    intervals_ms = [1000.0 * (b.time_s - a.time_s)
                    for a, b in zip(samples, samples[1:])]
    base_present = []
    baseline_in_range = []
    full_valid = []
    both_trackers_valid = []
    baselines = []
    side_valid = {side: [] for side in SERIALS}
    side_connected = {side: [] for side in SERIALS}
    side_results = {side: Counter() for side in SERIALS}
    side_poses = {side: [] for side in SERIALS}
    reference_changed_frames = 0

    for sample in samples:
        baseline = sample.base_station_baseline_m
        usable_baseline = bool(baseline is not None and math.isfinite(baseline))
        if usable_baseline:
            baselines.append(float(baseline))
        base_present.append(bool(sample.base_station_count >= 2 and usable_baseline))
        baseline_in_range.append(bool(
            sample.base_station_count >= 2 and usable_baseline
            and baseline_min_m <= baseline <= baseline_max_m))
        reference_changed_frames += bool(sample.reference_changed)
        for side in SERIALS:
            state = sample.states[side]
            connected = bool(state.get("connected"))
            valid = bool(connected and sample.poses[side] is not None
                         and state.get("valid") and not sample.reference_changed)
            side_connected[side].append(connected)
            side_valid[side].append(valid)
            side_results[side][str(state.get("tracking_result"))] += 1
            if valid:
                side_poses[side].append(sample.poses[side])
        both_trackers_valid.append(all(side_valid[side][-1] for side in SERIALS))
        full_valid.append(bool(baseline_in_range[-1]
                               and both_trackers_valid[-1]
                               and not sample.reference_changed))

    longest_invalid = _longest_false_streak(full_valid)
    valid_tail = _valid_tail(full_valid)
    p95_interval = _percentile(intervals_ms, 95)
    max_interval = max(intervals_ms, default=None)
    min_tail_frames = min(count, math.ceil(MIN_VALID_TAIL_S * SAMPLE_HZ))
    reasons = []
    if sum(full_valid) / count < MIN_SIMULTANEOUS_VALID_RATIO:
        reasons.append("双腕与两台指定基站同帧有效比例低于 98%")
    if sum(base_present) != count:
        reasons.append("两台指定基站不是每帧都有有效位置和间距")
    if not all(baseline_in_range):
        reasons.append(f"基站间距缺失或有采样帧超出 {baseline_min_m:.3f}–"
                       f"{baseline_max_m:.3f} 米")
    if longest_invalid > MAX_INVALID_STREAK_FRAMES:
        reasons.append("最长连续无效超过 2 帧")
    if max_interval is None or max_interval > MAX_SAMPLE_INTERVAL_MS or min(intervals_ms) <= 0:
        reasons.append("采样时间间隔超过 50 ms 或非单调")
    if valid_tail < min_tail_frames:
        reasons.append("采样末尾连续有效时间不足 0.75 秒")
    if reference_changed_frames:
        reasons.append("SteamVR 站立坐标参考系在采样中变化")

    report = {
        "schema_version": 1,
        "mode": mode,
        "requested_duration_s": duration_s,
        "sample_hz_target": SAMPLE_HZ,
        "samples": count,
        "elapsed_s": samples[-1].time_s - samples[0].time_s,
        "serials": SERIALS.copy(),
        "reference_changed_frames": reference_changed_frames,
        "sample_interval_ms": {
            "p95": p95_interval,
            "max": max_interval,
            "caveat": "主机采样间隔，不是 Tracker 到机械臂的端到端延迟",
        },
        "base_stations": {
            "both_named_valid_frames": sum(base_present),
            "baseline_in_range_frames": sum(baseline_in_range),
            "baseline_m": ({
                "samples": len(baselines),
                "median": float(statistics.median(baselines)),
                "min": min(baselines),
                "max": max(baselines),
            } if baselines else None),
            "required_baseline_m": [baseline_min_m, baseline_max_m],
            "close_layout_read_only_test": close_layout_test,
            "caveat": "OpenVR 基站位姿有效不能证明其当前对 Tracker 有光学贡献",
        },
        "trackers": {
            side: {
                "serial": SERIALS[side],
                "connected_frames": sum(side_connected[side]),
                "valid_frames": sum(side_valid[side]),
                "valid_ratio": sum(side_valid[side]) / count,
                "longest_invalid_streak_frames": _longest_false_streak(side_valid[side]),
                "tracking_results": dict(side_results[side]),
                **({"static_jitter": _static_jitter(side_poses[side]),
                    "static_jitter_caveat": "包含佩戴者微动，不能单独视为传感器噪声"}
                   if mode == "static" else {}),
            }
            for side in SERIALS
        },
        "simultaneous": {
            "both_trackers_valid_frames": sum(both_trackers_valid),
            "both_trackers_valid_ratio": sum(both_trackers_valid) / count,
            "valid_frames": sum(full_valid),
            "valid_ratio": sum(full_valid) / count,
            "longest_invalid_streak_frames": longest_invalid,
            "valid_tail_frames": valid_tail,
        },
        "calibration_readiness": {
            "pass": not reasons,
            "reasons": reasons,
            "meaning": "仅表示本次只读采样中的追踪质量；不保证之后整个工作区的持续覆盖或机器人运动精度",
        },
    }
    return report


def collect(pair, seconds: float) -> list:
    count = max(1, round(seconds * SAMPLE_HZ))
    start = time.perf_counter()
    samples = []
    for index in range(count):
        samples.append(pair.sample())
        delay = start + (index + 1) / SAMPLE_HZ - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("static", "sweep"))
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--output", type=Path, help="JSON file (default: logs/vive-quality-*.json)")
    parser.add_argument(
        "--expected-baseline-m", type=float,
        help="read-only close-layout test: require every measured baseline within ±0.02 m",
    )
    args = parser.parse_args(argv)
    if not MIN_DURATION_S <= args.seconds <= MAX_DURATION_S:
        parser.error("--seconds must be between 10 and 60")
    baseline_limits_m = None
    if args.expected_baseline_m is not None:
        expected = args.expected_baseline_m
        if not math.isfinite(expected) or expected <= EXPECTED_BASELINE_TOLERANCE_M:
            parser.error("--expected-baseline-m must be finite and greater than 0.02")
        baseline_limits_m = (expected - EXPECTED_BASELINE_TOLERANCE_M,
                             expected + EXPECTED_BASELINE_TOLERANCE_M)
    if not _user_vrserver_running():
        print("SteamVR 服务器未运行；请先从 Steam 启动 SteamVR。未访问追踪器。")
        return 2

    # Imported only after server preflight, so this tool cannot trigger a bare
    # OpenVR server launch when SteamVR has exited.
    from vive_tianji.tracker_input import BASE_STATION_SERIALS, TrackerPair

    try:
        with TrackerPair(SERIALS["left"], SERIALS["right"]) as pair:
            samples = collect(pair, args.seconds)
    except Exception as exc:
        print(f"只读采样失败：{exc}")
        return 2
    report = analyze(samples, args.mode, duration_s=args.seconds,
                     baseline_limits_m=baseline_limits_m)
    report["base_stations"]["serials"] = list(BASE_STATION_SERIALS)
    report["created_at"] = datetime.now().astimezone().isoformat()
    output = args.output or ROOT / "logs" / f"vive-quality-{datetime.now():%Y%m%d-%H%M%S}-{time.time_ns() % 1000000:06d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)

    same = report["simultaneous"]
    baseline = report["base_stations"]["baseline_m"]
    print(f"双腕同帧 6DoF {same['both_trackers_valid_frames']}/{report['samples']} "
          f"({same['both_trackers_valid_ratio']:.1%})；连同两台基站场地位姿有效 "
          f"{same['valid_frames']}/{report['samples']} ({same['valid_ratio']:.1%})，"
          f"最长连续无效 {same['longest_invalid_streak_frames']} 帧")
    for side in SERIALS:
        tracker = report["trackers"][side]
        print(f"{side} {tracker['serial']}：6DoF {tracker['valid_frames']}/{report['samples']} "
              f"({tracker['valid_ratio']:.1%})，最长连续无效 "
              f"{tracker['longest_invalid_streak_frames']} 帧")
    print(f"采样间隔 p95 {report['sample_interval_ms']['p95']:.1f} ms，"
          f"最大 {report['sample_interval_ms']['max']:.1f} ms")
    print(f"基站间距中位数 {baseline['median']:.3f} m，"
          f"范围 {baseline['min']:.3f}–{baseline['max']:.3f} m" if baseline else
          "两台指定基站尚无同帧有效间距")
    if args.mode == "static":
        for side in SERIALS:
            jitter = report["trackers"][side]["static_jitter"]
            if jitter:
                print(f"{side} 静止抖动：位置 RMS {jitter['position_rms_mm']:.2f} mm、"
                      f"p95 {jitter['position_p95_mm']:.2f} mm；"
                      f"姿态 RMS {jitter['orientation_rms_deg']:.3f}°、"
                      f"p95 {jitter['orientation_p95_deg']:.3f}°")
        print("静止抖动含佩戴者微动，仅用于对比安装/遮挡条件；PASS 不代表已达到绝对定位精度。")
    status = report["calibration_readiness"]
    print("PASS：本次只读采样追踪质量通过" if status["pass"] else
          "FAIL：" + "；".join(status["reasons"]))
    print(f"只读报告：{output}")
    return 0 if status["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
