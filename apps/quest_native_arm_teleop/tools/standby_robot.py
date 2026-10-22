#!/usr/bin/env python3
"""Request dual-arm standby only; never clear faults or send joint targets."""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import ipaddress
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quest_teleop.config import SDK_ROOT
from quest_teleop.vendor_checks import bind_robot_abi, read_servo_errors, version_info


def _load_robot():
    wrapper = SDK_ROOT / 'SDK_PYTHON/fx_robot.py'
    spec = importlib.util.spec_from_file_location('_standby_tianji_sdk', wrapper)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'无法加载厂家 SDK: {wrapper}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    robot = module.Marvin_Robot()
    bind_robot_abi(robot, module.DCSS)
    return module, robot


def _sample(robot, module, seconds=.5):
    rows = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rows.append(robot.subscribe(module.DCSS()))
        time.sleep(.02)
    if not rows:
        raise RuntimeError('未收到反馈')
    unique = [len({row['outputs'][i]['frame_serial'] for row in rows
                   if row['outputs'][i]['frame_serial'] != 0}) for i in range(2)]
    if not all(count > 1 for count in unique):
        raise RuntimeError(f'双臂反馈未持续更新: {unique}')
    return rows[-1], unique


def _snapshot(robot, row, unique):
    servo = read_servo_errors(robot)
    arms = {}
    for i, side in enumerate(('left', 'right')):
        output = row['outputs'][i]
        temperatures = np.asarray(output['fb_joint_them'], dtype=float)
        arms[side] = {
            **row['states'][i],
            'joints_deg': output['fb_joint_pos'],
            'temperature_c': output['fb_joint_them'],
            'telemetry_valid': bool(temperatures.shape == (7,) and
                                    np.isfinite(temperatures).all() and
                                    np.any(np.abs(temperatures) > 1e-6)),
            'frame_serial': output['frame_serial'],
            'servo_error_words': servo[side],
        }
    return {'unique_feedback_frames': unique, 'arms': arms}


def standby(ip, settle_seconds=3.):
    module, robot = _load_robot()
    connected = False
    report = {
        'checked_at': datetime.datetime.now().astimezone().isoformat(),
        'robot_ip': ip,
        'operation': 'dual_arm_standby',
        'error_clear_sent': False,
        'joint_targets_sent': False,
        'standby_command_sent': False,
        'standby_confirmed': False,
    }
    try:
        connected = bool(robot.connect(ip))
        report['sdk_socket_open'] = connected
        if not connected:
            raise RuntimeError('SDK 端口无法建立；请关闭其他控制程序')
        report.update(version_info(robot))
        before_row, before_unique = _sample(robot, module)
        report['before'] = _snapshot(robot, before_row, before_unique)
        arms = list(report['before']['arms'].values())
        if not all(arm['err_code'] == 0 and arm['telemetry_valid'] and
                   arm['servo_error_words'] == [0] * 7 for arm in arms):
            raise RuntimeError('仅允许将无故障、真实轴反馈在线的双臂转入待机')
        states = [arm['cur_state'] for arm in arms]
        if states == [0, 0]:
            report['standby_confirmed'] = True
            report['after'] = report['before']
            report['message'] = '双臂已处于待机，未发送命令'
            return report
        if states not in ([1, 1], [3, 3]):
            raise RuntimeError(f'只允许将双臂位置/阻抗模式 [1, 1]/[3, 3] 转为待机；当前 {states}')
        if not robot.clear_set():
            raise RuntimeError('无法清空厂家发送缓冲')
        if not robot.set_state('A', 0) or not robot.set_state('B', 0):
            raise RuntimeError('无法设置双臂待机目标')
        if not robot.send_cmd():
            raise RuntimeError('待机命令发送失败')
        report['standby_command_sent'] = True
        after_row, after_unique = _sample(robot, module, settle_seconds)
        report['after'] = _snapshot(robot, after_row, after_unique)
        report['standby_confirmed'] = all(
            arm['cur_state'] == 0 and arm['err_code'] == 0 and arm['telemetry_valid']
            for arm in report['after']['arms'].values())
        if not report['standby_confirmed']:
            raise RuntimeError('双臂未在时限内确认进入待机；请使用现场急停')
        report['message'] = '双臂已确认进入 state=0 待机；未清错、未发送关节目标'
    except Exception as exc:
        report['error'] = str(exc)
    finally:
        if connected:
            robot.release_robot()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-ip', default='192.168.1.190')
    parser.add_argument('--settle-seconds', type=float, default=3.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ipaddress.IPv4Address(args.robot_ip)
    if not .5 <= args.settle_seconds <= 10:
        parser.error('--settle-seconds 必须在 0.5–10 之间')
    report = standby(args.robot_ip, args.settle_seconds)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)
    raise SystemExit(0 if report['standby_confirmed'] else 2)
