"""Read-only quality summary for a recorded Vive/Tianji teleoperation session.

Usage: python3 -m vive_tianji.analyze_session logs/vive-live-.../stream.jsonl

The target/FK and target/measured comparisons use the robot's own kinematic
model and joint feedback. They are not measurements of absolute end-effector
accuracy against an external reference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


SIDES = ("left", "right")
PERCENTILES = (50, 95, 99)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _percentile(sorted_values: list[float], percent: int) -> float:
    index = (len(sorted_values) - 1) * percent / 100.0
    low = math.floor(index)
    high = math.ceil(index)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (index - low)


def _distribution(values: list[float]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    result: dict[str, int | float | None] = {"samples": len(ordered), "mean": None,
                                                 "p50": None, "p95": None, "p99": None,
                                                 "max": None}
    if ordered:
        result["mean"] = statistics.fmean(ordered)
        for percent in PERCENTILES:
            result[f"p{percent}"] = _percentile(ordered, percent)
        result["max"] = ordered[-1]
    return result


def _position(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    numbers = tuple(_finite_number(item) for item in value)
    return numbers if all(number is not None for number in numbers) else None


def _tracker_pose_valid(value: Any) -> bool:
    if not isinstance(value, dict) or _position(value.get("position_m")) is None:
        return False
    quaternion = value.get("quaternion_xyzw")
    if not isinstance(quaternion, list) or len(quaternion) != 4:
        return False
    components = [_finite_number(item) for item in quaternion]
    return bool(all(component is not None for component in components) and
                abs(sum(component * component for component in components) - 1.0) < 0.01)


def _distance_mm(first: Any, second: Any) -> float | None:
    a, b = _position(first), _position(second)
    if a is None or b is None:
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) * 1000.0


class _TrackingStats:
    def __init__(self) -> None:
        self.valid = 0
        self.total = 0
        self.loss_episodes = 0
        self.longest_loss_frames = 0
        self.longest_loss_s = 0.0
        self._loss_start: float | None = None
        self._loss_frames = 0

    def add(self, valid: bool, timestamp: float) -> None:
        self.total += 1
        if valid:
            self.valid += 1
            if self._loss_start is not None:
                self.longest_loss_s = max(self.longest_loss_s, timestamp - self._loss_start)
            self._loss_start = None
            self._loss_frames = 0
        else:
            if self._loss_start is None:
                self._loss_start = timestamp
                self.loss_episodes += 1
            self._loss_frames += 1
            self.longest_loss_frames = max(self.longest_loss_frames, self._loss_frames)

    def finish(self, final_time: float, sample_period_s: float) -> dict[str, int | float]:
        if self._loss_start is not None:
            self.longest_loss_s = max(
                self.longest_loss_s, final_time - self._loss_start + sample_period_s)
        return {
            "samples": self.total,
            "valid_samples": self.valid,
            "valid_fraction": self.valid / self.total if self.total else 0.0,
            "loss_episodes": self.loss_episodes,
            "longest_loss_frames": self.longest_loss_frames,
            "longest_loss_ms_approx": self.longest_loss_s * 1000.0,
        }


def analyze_stream(path: str | Path) -> dict[str, Any]:
    """Summarize one existing ``stream.jsonl`` without opening any device."""
    source = Path(path).expanduser()
    if source.is_dir():
        source /= "stream.jsonl"
    if source.name != "stream.jsonl":
        raise ValueError("请指定 Vive 会话目录或其中的 stream.jsonl")
    if not source.is_file():
        raise FileNotFoundError(source)

    tracking = {name: _TrackingStats() for name in (*SIDES, "both", "gate")}
    timings: dict[str, list[float]] = {"frame_age_ms": [], "queue_delay_ms": []}
    errors: dict[str, dict[str, list[float]]] = {
        side: {"target_to_fk_mm": [], "target_to_measured_mm": [],
               "joint_tracking_error_deg": []} for side in SIDES
    }
    sample_intervals: list[float] = []
    previous_time: float | None = None
    previous_seq: int | None = None
    missing_sequence_rows = 0
    control_rows = following_rows = 0

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: JSON 行损坏: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: 记录必须是 JSON 对象")
            timestamp = _finite_number(row.get("t"))
            seq = row.get("seq")
            if timestamp is None or isinstance(seq, bool) or not isinstance(seq, int):
                raise ValueError(f"{source}:{line_number}: t 或 seq 缺失/无效")
            if previous_time is not None:
                if timestamp <= previous_time or seq <= previous_seq:
                    raise ValueError(f"{source}:{line_number}: 时间或序号未递增")
                sample_intervals.append(timestamp - previous_time)
                missing_sequence_rows += max(0, seq - previous_seq - 1)
            previous_time, previous_seq = timestamp, seq

            trackers = row.get("tracker")
            trackers = trackers if isinstance(trackers, dict) else {}
            per_side = {side: _tracker_pose_valid(trackers.get(side)) for side in SIDES}
            for side in SIDES:
                tracking[side].add(per_side[side], timestamp)
            tracking["both"].add(all(per_side.values()), timestamp)
            tracking["gate"].add(row.get("valid") is True, timestamp)

            control = row.get("control")
            if not isinstance(control, dict):
                continue
            control_rows += 1
            for key in timings:
                value = _finite_number(control.get(key))
                if value is not None and value >= 0:
                    timings[key].append(value)
            if control.get("following") is not True:
                continue
            following_rows += 1
            for side in SIDES:
                for key, first_name, second_name in (
                    ("target_to_fk_mm", "target_position_m", "fk_position_m"),
                    ("target_to_measured_mm", "target_position_m", "measured_position_m"),
                ):
                    first = control.get(first_name)
                    second = control.get(second_name)
                    first = first.get(side) if isinstance(first, dict) else None
                    second = second.get(side) if isinstance(second, dict) else None
                    value = _distance_mm(first, second)
                    if value is not None:
                        errors[side][key].append(value)
                joint_errors = control.get("tracking_error_deg")
                value = (_finite_number(joint_errors.get(side))
                         if isinstance(joint_errors, dict) else None)
                if value is not None and value >= 0:
                    errors[side]["joint_tracking_error_deg"].append(value)

    if previous_time is None:
        raise ValueError(f"{source}: 没有可分析的记录")
    sample_period_s = statistics.median(sample_intervals) if sample_intervals else 0.0
    report = {
        "source": str(source.resolve()),
        "input_samples": tracking["both"].total,
        "duration_s": sum(sample_intervals),
        "nominal_sample_period_ms": sample_period_s * 1000.0,
        "max_sample_gap_ms": max(sample_intervals, default=0.0) * 1000.0,
        "missing_sequence_rows": missing_sequence_rows,
        "tracking": {name: stats.finish(previous_time, sample_period_s)
                     for name, stats in tracking.items()},
        "control_samples": control_rows,
        "following_control_samples": following_rows,
        "timing": {key: _distribution(values) for key, values in timings.items()},
        "following_internal_error": {
            side: {key: _distribution(values) for key, values in errors[side].items()}
            for side in SIDES
        },
        "interpretation": (
            "target_to_fk 是目标与指令关节正解之差；target_to_measured 是目标与"
            "设备反馈关节正解之差；joint_tracking_error 是设备指令/反馈角差。"
            "这些内部误差与延迟指标不等于外部测量的机械臂末端绝对精度。"
        ),
    }
    return report


def _format_distribution(label: str, values: dict[str, int | float | None], unit: str) -> str:
    if not values["samples"]:
        return f"  {label}: 无有效样本"
    return (f"  {label}: n={values['samples']}，p50={values['p50']:.2f}{unit}，"
            f"p95={values['p95']:.2f}{unit}，p99={values['p99']:.2f}{unit}，"
            f"max={values['max']:.2f}{unit}")


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"会话：{report['source']}",
        f"输入 {report['input_samples']} 帧 / {report['duration_s']:.1f} 秒；"
        f"控制快照 {report['control_samples']} 帧，跟随快照 "
        f"{report['following_control_samples']} 帧",
        f"最大输入间隔 {report['max_sample_gap_ms']:.1f} ms；"
        f"缺失序号 {report['missing_sequence_rows']} 帧",
        "追踪：",
    ]
    for name, label in (("left", "左腕"), ("right", "右腕"),
                        ("both", "双腕同时"), ("gate", "会话有效门禁")):
        row = report["tracking"][name]
        loss_word = "失效" if name == "gate" else "失位"
        lines.append(
            f"  {label}: {row['valid_samples']}/{row['samples']} "
            f"({row['valid_fraction'] * 100:.2f}%)；{loss_word} {row['loss_episodes']} 次，"
            f"最长 {row['longest_loss_frames']} 帧 / 约 "
            f"{row['longest_loss_ms_approx']:.1f} ms"
        )
    lines.append("输入/排队时延（每 5 帧记录一次控制快照）：")
    lines.append(_format_distribution("frame_age", report["timing"]["frame_age_ms"], "ms"))
    lines.append(_format_distribution("queue_delay", report["timing"]["queue_delay_ms"], "ms"))
    lines.append("跟随期间内部误差（仅 control.following=true 的快照）：")
    for side, label in (("left", "左臂"), ("right", "右臂")):
        values = report["following_internal_error"][side]
        lines.append(f"  {label}：")
        lines.append(_format_distribution("目标→指令 FK", values["target_to_fk_mm"], "mm"))
        lines.append(_format_distribution("目标→实测 FK", values["target_to_measured_mm"], "mm"))
        lines.append(_format_distribution("指令→反馈关节", values["joint_tracking_error_deg"], "°"))
    lines.append("说明：" + report["interpretation"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Vive 会话目录或 stream.jsonl 路径")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args(argv)
    try:
        report = analyze_stream(args.session)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"分析失败：{exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
