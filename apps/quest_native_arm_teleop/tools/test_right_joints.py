#!/usr/bin/env python3
"""低速逐轴检查天机右臂；左臂始终保持 state=0，且不自动清错。"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quest_teleop.config import load_config
from quest_teleop.hardware import HardwareError, TianjiHardwareOutput
from quest_teleop.kinematics import Kinematics
from quest_teleop.vendor_checks import read_servo_errors


SIDES = ('left', 'right')
AXIS_AMPLITUDES_DEG = (1., 1., 1., 1., 1., .5, .5)


def quintic(value: float) -> float:
    value = min(1., max(0., float(value)))
    return 10 * value ** 3 - 15 * value ** 4 + 6 * value ** 5


def target_at(start: np.ndarray, axis: int, offset_deg: float, progress: float) -> np.ndarray:
    target = np.asarray(start, dtype=float).copy()
    target[axis] += float(offset_deg) * quintic(progress)
    return target


def _telemetry(output: TianjiHardwareOutput):
    packet = output._subscribe()
    feedback = output.read_feedback()
    servo = read_servo_errors(output.robot)
    rows = {}
    for index, side in enumerate(SIDES):
        raw = packet['outputs'][index]
        rows[side] = {
            'joints_deg': list(map(float, raw['fb_joint_pos'])),
            'velocity_deg_s': list(map(float, raw['fb_joint_vel'])),
            'external_joints_deg': list(map(float, raw['fb_joint_posE'])),
            'feedback_command_deg': list(map(float, raw['fb_joint_cmd'])),
            'current_torque': list(map(float, raw['fb_joint_cToq'])),
            'sensor_torque': list(map(float, raw['fb_joint_sToq'])),
            'temperature_c': list(map(float, raw['fb_joint_them'])),
            'servo_error_words': servo[side],
            'state': int(packet['states'][index]['cur_state']),
            'error': int(packet['states'][index]['err_code']),
            'frame_serial': int(raw['frame_serial']),
        }
    return feedback, rows


def _sample_row(elapsed, axis, phase, command, telemetry):
    right = telemetry['right']
    row = {'elapsed_s': elapsed, 'axis': axis + 1, 'phase': phase}
    fields = {
        'command_deg': command,
        'measured_deg': right['joints_deg'],
        'external_deg': right['external_joints_deg'],
        'controller_command_deg': right['feedback_command_deg'],
        'velocity_deg_s': right['velocity_deg_s'],
        'current_torque': right['current_torque'],
        'sensor_torque': right['sensor_torque'],
        'temperature_c': right['temperature_c'],
    }
    for name, values in fields.items():
        for index, value in enumerate(values):
            row[f'{name}_j{index + 1}'] = float(value)
    return row


def run(ip: str, duration_s: float, sample_hz: float, output_json: Path, output_csv: Path):
    config_path = Path(__file__).resolve().parents[1] / 'configs' / 'hardware_right_arm.json'
    config = load_config(config_path, hardware=True)
    config['hardware']['robot_ip'] = ip
    config['hardware']['send_clear_timeout_s'] = .04
    kine = Kinematics(config)
    controller = TianjiHardwareOutput(config)
    report = {
        'started_at': datetime.datetime.now().astimezone().isoformat(),
        'robot_ip': ip,
        'arm': 'right',
        'left_arm_commanded': False,
        'mode': 'position',
        'velocity_ratio_percent': 1,
        'acceleration_ratio_percent': 1,
        'amplitudes_deg': list(AXIS_AMPLITUDES_DEG),
        'segment_duration_s': duration_s,
        'success': False,
        'axes': [],
    }
    samples = []
    connected = False
    start_time = time.monotonic()
    try:
        joints = controller.connect()
        connected = True
        baseline = {side: np.asarray(value, dtype=float) for side, value in joints.items()}
        report['initial_joints_deg'] = {side: value.tolist() for side, value in baseline.items()}
        if not kine.valid_joints('right', baseline['right']):
            raise HardwareError('右臂当前角度不符合已核验的 Lite 关节/腕部耦合限位')

        controller.enable()
        deadline = time.monotonic() + 4.
        while not controller.enabled and time.monotonic() < deadline:
            controller.poll()
            time.sleep(.01)
        if not controller.enabled:
            raise HardwareError('右臂未进入 state=1 位置模式')

        dt = 1. / sample_hz
        for axis, amplitude in enumerate(AXIS_AMPLITUDES_DEG):
            axis_start = controller.read_joints()['right'].copy()
            axis_result = {'joint': axis + 1, 'amplitude_deg': amplitude,
                           'start_deg': float(axis_start[axis]), 'segments': []}
            report['axes'].append(axis_result)
            # Each excursion starts and ends at the freshly measured axis origin.
            for phase, offset_start, offset_end in (
                    ('positive', 0., amplitude), ('positive_return', amplitude, 0.),
                    ('negative', 0., -amplitude), ('negative_return', -amplitude, 0.)):
                segment_start = time.monotonic()
                segment_peak_error = 0.
                segment_peak_other_motion = 0.
                segment_samples = 0
                while True:
                    now = time.monotonic()
                    u = min(1., (now - segment_start) / duration_s)
                    offset = offset_start + (offset_end - offset_start) * quintic(u)
                    right_target = axis_start.copy(); right_target[axis] += offset
                    if not kine.valid_joints('right', right_target):
                        raise HardwareError(f'J{axis + 1} 测试目标触及关节或 J6/J7 耦合限位')
                    targets = {'left': baseline['left'].copy(), 'right': right_target}
                    controller.send_joint_targets(targets)
                    feedback, telemetry = _telemetry(controller)
                    measured = np.asarray(telemetry['right']['joints_deg'], dtype=float)
                    errors = np.abs(right_target - measured)
                    other = np.delete(np.abs(measured - axis_start), axis)
                    segment_peak_error = max(segment_peak_error, float(errors[axis]))
                    segment_peak_other_motion = max(segment_peak_other_motion,
                                                    float(np.max(other)) if other.size else 0.)
                    if telemetry['left']['state'] != 0:
                        raise HardwareError(f'左臂意外离开待机（state={telemetry["left"]["state"]}）')
                    if any(telemetry[side]['servo_error_words'] != [0] * 7 for side in SIDES):
                        raise HardwareError('检测到非零伺服错误字')
                    if errors[axis] > 1.5:
                        raise HardwareError(f'右臂 J{axis + 1} 跟随误差 {errors[axis]:.2f}° 超过 1.5°')
                    if np.any(other > .35):
                        raise HardwareError(f'测试 J{axis + 1} 时其他右臂关节移动超过 0.35°')
                    samples.append(_sample_row(now - start_time, axis, phase,
                                               right_target.tolist(), telemetry))
                    segment_samples += 1
                    if u >= 1.:
                        break
                    wait = segment_start + segment_samples * dt - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                axis_result['segments'].append({
                    'phase': phase,
                    'peak_tracking_error_deg': segment_peak_error,
                    'peak_other_joint_motion_deg': segment_peak_other_motion,
                    'samples': segment_samples,
                })
            final = controller.read_joints()['right']
            axis_result['final_deg'] = float(final[axis])
            axis_result['return_error_deg'] = float(abs(final[axis] - axis_start[axis]))
        report['success'] = True
    except BaseException as exc:
        report['error'] = str(exc)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            report['interrupted'] = True
    finally:
        report['finished_at'] = datetime.datetime.now().astimezone().isoformat()
        report['send_clear_retries'] = int(getattr(controller, 'send_clear_retries', 0))
        if connected:
            try:
                controller.close()
                report['standby_requested'] = True
            except Exception as exc:
                report['standby_error'] = str(exc)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        if samples:
            with output_csv.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
                writer.writeheader(); writer.writerows(samples)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-ip', default='192.168.1.190')
    parser.add_argument('--duration-s', type=float, default=2.)
    parser.add_argument('--sample-hz', type=float, default=25.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--confirmed', action='store_true')
    args = parser.parse_args()
    if not args.confirmed:
        parser.error('缺少现场确认 --confirmed')
    if not 1. <= args.duration_s <= 10. or not 10. <= args.sample_hz <= 50.:
        parser.error('duration-s 必须为 1–10，sample-hz 必须为 10–50')
    report = run(args.robot_ip, args.duration_s, args.sample_hz, args.output, args.csv)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
