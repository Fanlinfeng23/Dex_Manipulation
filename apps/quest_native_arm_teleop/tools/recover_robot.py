#!/usr/bin/env python3
"""Clear only a latched dual-arm E-stop while commanding standby; never enable or move."""
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
from quest_teleop.config import SDK_ROOT, load_config
from quest_teleop.vendor_checks import (bind_robot_abi, controller_model_info,
                                        read_servo_errors, version_info)


def _load_robot():
    wrapper = SDK_ROOT / 'SDK_PYTHON/fx_robot.py'
    spec = importlib.util.spec_from_file_location('_recover_tianji_sdk', wrapper)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'无法加载厂家 SDK: {wrapper}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    robot = module.Marvin_Robot()
    bind_robot_abi(robot, module.DCSS)
    return module, robot


def _sample(robot, module, seconds):
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
        arms[side] = {**row['states'][i],
                      'joints_deg': row['outputs'][i]['fb_joint_pos'],
                      'frame_serial': row['outputs'][i]['frame_serial'],
                      'servo_error_words': servo[side]}
    return {'unique_feedback_frames': unique, 'arms': arms}


def recover(ip, settle_seconds=3.):
    module, robot = _load_robot()
    connected = False
    report = {'checked_at': datetime.datetime.now().astimezone().isoformat(),
              'robot_ip': ip, 'operation': 'clear_latched_estop_and_standby',
              'motion_commands_sent': False, 'servo_enable_sent': False,
              'joint_targets_sent': False, 'recovered': False}
    try:
        connected = bool(robot.connect(ip))
        report['sdk_socket_open'] = connected
        if not connected:
            raise RuntimeError('SDK 端口无法建立；关闭 MarvinPlatform 和其他控制程序')
        report.update(version_info(robot))
        report.update(controller_model_info(robot, load_config(hardware=True)))
        before_row, before_unique = _sample(robot, module, .5)
        report['before'] = _snapshot(robot, before_row, before_unique)
        states = [arm['cur_state'] for arm in report['before']['arms'].values()]
        commands = [arm['cmd_state'] for arm in report['before']['arms'].values()]
        errors = [arm['err_code'] for arm in report['before']['arms'].values()]
        if states == [0, 0] and errors == [0, 0]:
            report['recovered'] = True
            report['message'] = '控制器已经处于双臂无故障待机，无需清错'
            report['after'] = report['before']
            return report
        if states != [100, 100] or errors != [13, 13] or any(cmd not in (-1, 0) for cmd in commands):
            raise RuntimeError('仅允许恢复“双臂急停 13 且目标为待机”；当前状态不符合，拒绝写入')

        # Vendor-prescribed reset, deliberately paired with state=0 only.
        # No position, velocity, impedance, release, brake or servo-reset API is used.
        if not robot.clear_set():
            raise RuntimeError('无法清空厂家发送缓冲')
        robot.clear_error('A')
        robot.clear_error('B')
        if not robot.set_state('A', 0) or not robot.set_state('B', 0):
            raise RuntimeError('无法设置双臂待机目标')
        if not robot.send_cmd():
            raise RuntimeError('待机/清错命令发送失败')
        report['standby_and_clear_sent'] = True

        after_row, after_unique = _sample(robot, module, settle_seconds)
        report['after'] = _snapshot(robot, after_row, after_unique)
        report['recovered'] = all(
            arm['cur_state'] == 0 and arm['err_code'] == 0 and
            np.isfinite(arm['joints_deg']).all() and
            not np.allclose(arm['joints_deg'], 0., atol=1e-6, rtol=0)
            for arm in report['after']['arms'].values())
        if report['recovered']:
            report['message'] = '双臂已恢复为无故障待机'
        else:
            after_arms = list(report['after']['arms'].values())
            if all(arm['cur_state'] == 100 and arm['cmd_state'] == 0 and
                   arm['err_code'] == 13 and arm['servo_error_words'] == [0] * 7
                   for arm in after_arms):
                report['failure_class'] = 'hardware_safety_chain_still_asserted'
                report['message'] = ('厂家清错和双臂待机命令已执行，但控制器公共急停输入仍有效；'
                                     '不要重复软件清错，请断电后检查 GPIO 急停插头 3/4/5 号线、'
                                     '急停按钮常闭触点、安全回路和 48V 驱动供电')
            else:
                report['failure_class'] = 'standby_feedback_not_valid'
                report['message'] = ('未获得“双臂无故障待机 + 有效非全零关节反馈”；'
                                     '请检查控制器状态和驱动供电')
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
    parser.add_argument('--confirm-physical-estop-released', action='store_true', required=True,
                        help='确认现场物理急停已释放且双臂工作区安全')
    args = parser.parse_args()
    ipaddress.IPv4Address(args.robot_ip)
    if not .5 <= args.settle_seconds <= 10:
        parser.error('--settle-seconds 必须在 0.5–10 之间')
    report = recover(args.robot_ip, args.settle_seconds)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)
    sys.exit(0 if report['recovered'] else 2)
