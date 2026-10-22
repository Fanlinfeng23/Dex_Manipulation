#!/usr/bin/env python3
"""Offline inspection of raw PXREA callback JSONL. Never controls a device.

Quest predictTime is Unity application uptime in microseconds, not a sensor
timestamp. Controller transforms have no per-side tracking-valid flag.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path


POSE_KEYS = {'pose', 'p', 'transform'}
TRACKING_KEYS = {
    'tracked', 'isTracked', 'trackingState', 'trackingStatus',
    'trackingFlags', 'positionTracked', 'orientationTracked',
    'isPoseValid', 'poseValid',
}


def parse_value(payload):
    """The official SDK sends a JSON wrapper whose `value` is another JSON string."""
    root = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(root, dict):
        raise ValueError('payload root is not an object')
    value = root.get('value', root)
    for _ in range(3):
        if not isinstance(value, str):
            break
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError('tracking value is not an object')
    return value


def app_time_ns(value):
    """Application-generated frame time, for dedup only (not sensor time)."""
    raw = value.get('predictTime')
    if isinstance(raw, bool):
        return None
    try:
        micros = float(raw)
    except (TypeError, ValueError):
        return None
    return round(micros * 1000) if math.isfinite(micros) and micros > 0 else None


def wall_timestamp_domain(value, arrival_wall_ns):
    raw = value.get('timeStampNs')
    if isinstance(raw, bool):
        return 'missing'
    try:
        stamp = int(raw)
    except (TypeError, ValueError):
        return 'missing'
    if arrival_wall_ns is not None and abs(stamp - arrival_wall_ns) < 86400 * 10**9:
        return 'unix_utc_ns'
    if 500_000_000_000_000_000 <= stamp < 900_000_000_000_000_000:
        return 'possible_dotnet_ticks_fallback'
    return 'unknown_domain'


def pose_fields(node, path=(), inherited_flags=()):
    if isinstance(node, dict):
        flags = inherited_flags + tuple(
            '.'.join(path + (key,)) for key in TRACKING_KEYS if key in node)
        for key, value in node.items():
            child_path = path + (key,)
            if key.lower() in POSE_KEYS:
                yield child_path, value, flags
            if isinstance(value, (dict, list)):
                yield from pose_fields(value, child_path, flags)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            if isinstance(value, (dict, list)):
                yield from pose_fields(value, path + (f'[{index}]',), inherited_flags)


def numeric_vector(raw, length):
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(',')]
    elif isinstance(raw, (tuple, list)):
        parts = raw
    elif isinstance(raw, dict):
        keys = ('x', 'y', 'z') if length == 3 else ('x', 'y', 'z', 'w')
        parts = [raw.get(key) for key in keys]
    else:
        return None
    if len(parts) != length:
        return None
    try:
        result = tuple(float(part) for part in parts)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(value) for value in result) else None


def parse_pose(raw):
    """Return meter position and xyzw quaternion for documented 7-value poses."""
    if isinstance(raw, dict):
        position = numeric_vector(raw.get('position') or raw.get('translation'), 3)
        rotation = numeric_vector(raw.get('rotation') or raw.get('orientation') or
                                  raw.get('quaternion'), 4)
    else:
        vector = numeric_vector(raw, 7)
        if vector is None:
            return None
        position, rotation = vector[:3], vector[3:]
    if position is None or rotation is None:
        return None
    norm = math.sqrt(sum(value * value for value in rotation))
    if not 0.8 <= norm <= 1.2:
        return None
    return position, tuple(value / norm for value in rotation)


def side_from_path(path):
    lower = '.'.join(path).lower()
    if 'controller' not in lower:
        return None
    if '.left' in lower or 'leftcontroller' in lower or '_left' in lower:
        return 'left'
    if '.right' in lower or 'rightcontroller' in lower or '_right' in lower:
        return 'right'
    return None


def rotation_step_deg(previous, current):
    dot = abs(sum(a * b for a, b in zip(previous, current)))
    return math.degrees(2 * math.acos(min(1.0, max(-1.0, dot))))


def new_stream(device_id, path, side):
    return {
        'device_id': device_id, 'path': path, 'side': side,
        'pose_field_events': 0, 'app_uptime_missing': 0,
        'duplicate_timestamps': 0, 'conflicting_duplicate_poses': 0,
        'app_uptime_regressions': 0, 'unique_samples': 0,
        'valid_pose_samples': 0, 'invalid_pose_samples': 0,
        'app_frame_gaps': [], 'receive_gaps': [], 'pose_jumps': [],
        'tracking_flag_fields': set(), 'app_deltas_ns': [],
        '_seen': {}, '_previous': None, '_epochs': 0,
        '_distinct_poses': set(), '_run_pose': None,
        '_run_start_ns': None, '_run_end_ns': None, '_run_frames': 0,
        '_longest_run_ms': 0.0, '_longest_run_frames': 0,
    }


def finish_identical_pose_run(stream):
    if stream['_run_pose'] is None:
        return
    duration_ms = (stream['_run_end_ns'] - stream['_run_start_ns']) / 1e6
    if (duration_ms > stream['_longest_run_ms'] or
            (duration_ms == stream['_longest_run_ms'] and
             stream['_run_frames'] > stream['_longest_run_frames'])):
        stream['_longest_run_ms'] = duration_ms
        stream['_longest_run_frames'] = stream['_run_frames']
    stream['_run_pose'] = None
    stream['_run_start_ns'] = None
    stream['_run_end_ns'] = None
    stream['_run_frames'] = 0


def extend_identical_pose_run(stream, pose, app_ns):
    if pose is None:
        finish_identical_pose_run(stream)
    elif pose == stream['_run_pose']:
        stream['_run_end_ns'] = app_ns
        stream['_run_frames'] += 1
    else:
        finish_identical_pose_run(stream)
        stream['_run_pose'] = pose
        stream['_run_start_ns'] = app_ns
        stream['_run_end_ns'] = app_ns
        stream['_run_frames'] = 1


def numbered_lines(path):
    with path.open('r', encoding='utf-8') as source:
        yield from enumerate(source, 1)


def analyze(path, *, gap_ms=100.0, jump_m=0.25, jump_deg=45.0):
    """Analyze application frame time, callback gaps, and numeric poses."""
    path = Path(path)
    event_types = Counter()
    wall_domains = Counter()
    focus_events = Counter()
    head_status_events = Counter()
    input_events = Counter()
    streams = {}
    errors = []
    total_lines = 0
    state_events = 0
    state_with_pose = 0
    previous_wall_ns = None
    wall_regressions = 0
    for number, line in numbered_lines(path):
        if not line.strip():
            continue
        total_lines += 1
        try:
            event = json.loads(line)
            event_type = event['event_type']
            event_types[event_type] += 1
            if event_type != 'PXREADeviceStateJson':
                continue
            state_events += 1
            value = parse_value(event['payload'])
            device_id = str(event.get('device_id') or '?')
            arrival_ns = event.get('recv_monotonic_ns')
            if not isinstance(arrival_ns, int):
                arrival_ns = None
            arrival_wall_ns = event.get('recv_wall_ns')
            if not isinstance(arrival_wall_ns, int):
                arrival_wall_ns = None
            wall_domain = wall_timestamp_domain(value, arrival_wall_ns)
            wall_domains[wall_domain] += 1
            if wall_domain == 'unix_utc_ns':
                current_wall_ns = int(value['timeStampNs'])
                if previous_wall_ns is not None and current_wall_ns < previous_wall_ns:
                    wall_regressions += 1
                previous_wall_ns = current_wall_ns
            app_state = value.get('appState')
            app_focus = app_state.get('focus') if isinstance(app_state, dict) else None
            focus_events[str(app_focus).lower() if isinstance(app_focus, bool)
                         else 'unknown'] += 1
            head = value.get('Head')
            if isinstance(head, dict):
                head_status_events[str(head.get('status', 'unknown'))] += 1
            else:
                head_status_events['unknown'] += 1
            input_value = value.get('Input')
            input_events[str(input_value) if input_value is not None else 'missing'] += 1
            timestamp_ns = app_time_ns(value)
            fields = list(pose_fields(value))
            if fields:
                state_with_pose += 1
            for keys, raw_pose, flags in fields:
                stream_path = '.'.join(keys)
                stream_key = (device_id, stream_path)
                stream = streams.setdefault(
                    stream_key, new_stream(device_id, stream_path, side_from_path(keys)))
                stream['pose_field_events'] += 1
                stream['tracking_flag_fields'].update(flags)
                parsed_pose = parse_pose(raw_pose)
                if parsed_pose is None:
                    stream['invalid_pose_samples'] += 1
                else:
                    stream['_distinct_poses'].add(parsed_pose)
                if timestamp_ns is None:
                    stream['app_uptime_missing'] += 1
                    finish_identical_pose_run(stream)
                    continue
                previous = stream['_previous']
                if previous is not None and timestamp_ns < previous[0]:
                    # Unity app restarted: same uptime can be reused in a new epoch.
                    finish_identical_pose_run(stream)
                    stream['app_uptime_regressions'] += 1
                    stream['_epochs'] += 1
                    stream['_seen'].clear()
                    stream['_previous'] = None
                    previous = None
                prior_at_same_time = stream['_seen'].get(timestamp_ns)
                if timestamp_ns in stream['_seen']:
                    stream['duplicate_timestamps'] += 1
                    if prior_at_same_time != parsed_pose:
                        stream['conflicting_duplicate_poses'] += 1
                    continue
                stream['_seen'][timestamp_ns] = parsed_pose
                stream['unique_samples'] += 1
                if parsed_pose is not None:
                    stream['valid_pose_samples'] += 1
                extend_identical_pose_run(stream, parsed_pose, timestamp_ns)
                if previous is not None:
                    previous_ns, previous_arrival_ns, previous_pose = previous
                    elapsed_ns = timestamp_ns - previous_ns
                    if elapsed_ns > 0:
                        stream['app_deltas_ns'].append(elapsed_ns)
                        if elapsed_ns > gap_ms * 1e6:
                            stream['app_frame_gaps'].append({
                                'from_app_uptime_ns': previous_ns,
                                'to_app_uptime_ns': timestamp_ns,
                                'duration_ms': round(elapsed_ns / 1e6, 3),
                            })
                        if previous_pose is not None and parsed_pose is not None:
                            distance = math.dist(previous_pose[0], parsed_pose[0])
                            angle = rotation_step_deg(previous_pose[1], parsed_pose[1])
                            if distance > jump_m or angle > jump_deg:
                                stream['pose_jumps'].append({
                                    'from_app_uptime_ns': previous_ns,
                                    'to_app_uptime_ns': timestamp_ns,
                                    'translation_m': round(distance, 4),
                                    'rotation_deg': round(angle, 2),
                                })
                    if (arrival_ns is not None and previous_arrival_ns is not None and
                            arrival_ns - previous_arrival_ns > gap_ms * 1e6):
                        stream['receive_gaps'].append({
                            'from_ns': previous_arrival_ns, 'to_ns': arrival_ns,
                            'duration_ms': round((arrival_ns - previous_arrival_ns) / 1e6, 3),
                        })
                stream['_previous'] = (timestamp_ns, arrival_ns, parsed_pose)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append({'line': number, 'error': f'{type(exc).__name__}: {exc}'})

    clean_streams = []
    for stream in streams.values():
        finish_identical_pose_run(stream)
        stream['different_pose_count'] = len(stream.pop('_distinct_poses'))
        stream['longest_identical_pose_duration_ms'] = round(
            stream.pop('_longest_run_ms'), 3)
        stream['longest_identical_pose_run_frames'] = stream.pop('_longest_run_frames')
        for private in ('_run_pose', '_run_start_ns', '_run_end_ns', '_run_frames'):
            stream.pop(private)
        stream.pop('_seen')
        stream.pop('_previous')
        stream['app_uptime_epochs'] = stream.pop('_epochs') + (
            1 if stream['unique_samples'] else 0)
        deltas = stream.pop('app_deltas_ns')
        stream['tracking_flag_fields'] = sorted(stream['tracking_flag_fields'])
        stream['median_app_frame_period_ms'] = (
            round(statistics.median(deltas) / 1e6, 3) if deltas else None)
        stream['median_app_frame_rate_hz'] = (
            round(1e9 / statistics.median(deltas), 2) if deltas else None)
        stream['app_frame_gap_count'] = len(stream['app_frame_gaps'])
        stream['receive_gap_count'] = len(stream['receive_gaps'])
        stream['pose_jump_count'] = len(stream['pose_jumps'])
        # Bound detailed diagnostic arrays in longer recordings.
        for key in ('app_frame_gaps', 'receive_gaps', 'pose_jumps'):
            stream[key] = stream[key][:20]
        clean_streams.append(stream)
    clean_streams.sort(key=lambda item: (item['device_id'], item['path']))
    sides_with_numeric_pose = sorted({stream['side'] for stream in clean_streams
                                      if stream['side'] and stream['valid_pose_samples']})
    return {
        'input': str(path), 'raw_event_lines': total_lines,
        'event_types': dict(event_types), 'device_state_events': state_events,
        'state_events_with_pose_fields': state_with_pose,
        'app_focus_events': dict(focus_events),
        'top_level_input_counts': dict(input_events),
        'head_status_events': dict(head_status_events),
        'head_status_semantics': 'Quest client: 3=valid, 0=invalid; do not apply '
                                 'the PICO README 0/1 convention.',
        'top_level_timestamp_domains': dict(wall_domains),
        'top_level_wall_timestamp_regressions': wall_regressions,
        'controller_sides_with_numeric_pose': sides_with_numeric_pose,
        'has_state_frames': state_events > 0,
        'has_both_controller_pose_fields': sides_with_numeric_pose == ['left', 'right'],
        'thresholds': {'gap_ms': gap_ms, 'jump_m': jump_m, 'jump_deg': jump_deg},
        'streams': clean_streams, 'parse_errors': errors[:20],
        'parse_error_count': len(errors),
        'true_6dof_proven': False,
        'optical_tracking_validity': 'unknown',
        'tracking_limitation': 'The Quest client sends controller transforms without '
                               'per-side tracking-valid flags. App output and receive '
                               'continuity cannot prove real optical 6DoF.',
        'time_semantics': 'predictTime is Unity application uptime in microseconds; '
                          'timeStampNs is normally UTC nanoseconds but may use .NET '
                          'Ticks fallback. Neither is sensor-sample time.',
        'identical_pose_semantics': 'Exact numeric pose equality over distinct '
            'application frame times; duration uses app uptime. Stillness and '
            'frozen tracking cannot be distinguished from this metric alone.',
    }


def summary_path(recording):
    return Path(str(recording) + '.summary.json')


def write_summary(recording, output=None, **kwargs):
    result = analyze(recording, **kwargs)
    destination = Path(output) if output is not None else summary_path(recording)
    # A recorder summary contains capture metadata unavailable from the JSONL.
    # Reanalysis must not erase it. Prefer the selected destination, then the
    # default sidecar when writing a separately named analysis file.
    for previous_path in dict.fromkeys((destination, summary_path(recording))):
        if not previous_path.exists():
            continue
        try:
            previous = json.loads(previous_path.read_text(encoding='utf-8'))
            previous_input = Path(previous['input']).resolve()
            if previous_input == Path(recording).resolve() and 'capture' in previous:
                result['capture'] = previous['capture']
                break
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically so an interrupted reanalysis leaves the old sidecar.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8',
                                         dir=destination.parent,
                                         prefix=f'.{destination.name}.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return result, destination


def main(argv=None):
    parser = argparse.ArgumentParser(description='离线分析 XRoboToolkit 原始 JSONL')
    parser.add_argument('input', type=Path)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--gap-ms', type=float, default=100.0)
    parser.add_argument('--jump-m', type=float, default=0.25)
    parser.add_argument('--jump-deg', type=float, default=45.0)
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.gap_ms, args.jump_m, args.jump_deg)):
        parser.error('gap/jump 阈值必须为有限正数')
    result, destination = write_summary(
        args.input, args.summary, gap_ms=args.gap_ms,
        jump_m=args.jump_m, jump_deg=args.jump_deg)
    print(f"分析完成：{destination}；状态事件 {result['device_state_events']}；"
          f"数值位姿侧别 {','.join(result['controller_sides_with_numeric_pose']) or '无'}；"
          '光学追踪有效性未知')
    return 0 if result['has_state_frames'] and not result['parse_error_count'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
