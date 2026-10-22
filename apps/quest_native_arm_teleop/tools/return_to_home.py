#!/usr/bin/env python3
"""Guarded, slow return from a small joint-limit overrun to the teleop startup pose.

Planning is offline. Execution always reconnects, checks fresh feedback, and
requests standby on every exit. No fault clearing is performed here.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
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
MAX_JOINT_OVERRUN_DEG = 0.5
MAX_WRIST_OVERRUN_DEG = 3.0
MAX_SPEED_DEG_S = 5.0
MAX_ACCEL_DEG_S2 = 8.0
SAMPLE_HZ = 25
FOLLOW_ERROR_DEG = 3.0


def violation(kine, side, q):
    limits = kine.arms[side]['limits']
    axis = np.maximum(limits[:, 0] - q, 0) + np.maximum(q - limits[:, 1], 0)
    wrist = np.array([max(0., -float(np.rad2deg(slack)))
                      for _, slack in kine._wrist_faces(side, q)])
    return axis, wrist


def limit_diagnostics(kine, side, joints):
    """Describe the model constraints in degrees without changing any limit."""
    q = np.asarray(joints, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError(f'{side}: 关节反馈无效')
    axis, wrist = violation(kine, side, q)
    limits = kine.arms[side]['limits']
    faces = kine._wrist_faces(side, q)
    return {
        'joints_deg': q.tolist(),
        'within_model_limits': kine.valid_joints(side, q),
        'max_joint_overrun_deg': float(np.max(axis)),
        'max_wrist_overrun_deg': float(np.max(wrist)) if wrist.size else 0.,
        'joint_violations': [
            {'joint': i + 1, 'actual_deg': float(q[i]),
             'lower_deg': float(limits[i, 0]), 'upper_deg': float(limits[i, 1]),
             'overrun_deg': float(axis[i])}
            for i in range(7) if axis[i] > 1e-7],
        'wrist_constraints': [
            {'joint6_deg': float(q[5]), 'joint7_deg': float(q[6]),
             'bound': 'upper' if row[6] > 0 else 'lower',
             'joint7_bound_deg': float(q[6] + row[6] * np.rad2deg(slack)),
             'margin_deg': float(np.rad2deg(slack)),
             'overrun_deg': max(0., -float(np.rad2deg(slack)))}
            for row, slack in faces],
        'rescue_tolerance_deg': {'joint': MAX_JOINT_OVERRUN_DEG,
                                 'wrist': MAX_WRIST_OVERRUN_DEG},
    }


def overrun_description(kine, side, start):
    info = limit_diagnostics(kine, side, start)
    reasons = [f"J{v['joint']}={v['actual_deg']:.3f}°，允许 "
               f"[{v['lower_deg']:.3f}, {v['upper_deg']:.3f}]°，越限 {v['overrun_deg']:.3f}°"
               for v in info['joint_violations']]
    for v in info['wrist_constraints']:
        if v['overrun_deg'] > 1e-7:
            sign = '≤' if v['bound'] == 'upper' else '≥'
            reasons.append(f"腕部耦合：J6={v['joint6_deg']:.3f}° 时 J7 应 {sign} "
                           f"{v['joint7_bound_deg']:.3f}°，实际 {v['joint7_deg']:.3f}°，"
                           f"越限 {v['overrun_deg']:.3f}°")
    return '；'.join(reasons)


def rescue_waypoint(kine, side, start):
    """Find a one-axis inward move with monotonically shrinking violations."""
    axis, wrist = violation(kine, side, start)
    if np.max(axis) > MAX_JOINT_OVERRUN_DEG or (wrist.size and np.max(wrist) > MAX_WRIST_OVERRUN_DEG):
        raise ValueError(
            f'{side}: {overrun_description(kine, side, start)}。'
            f'超出仅可低速复位的轻微越限范围（单轴 {MAX_JOINT_OVERRUN_DEG:g}° / '
            f'腕部耦合 {MAX_WRIST_OVERRUN_DEG:g}°）；未发送运动指令。'
            '请由熟悉该机械臂的人员使用厂家软件现场恢复到模型限位内，再执行 home。')
    if kine.valid_joints(side, start):
        return None
    candidates = []
    for joint in range(7):
        for direction in (-1., 1.):
            for distance in np.arange(.25, 5.001, .25):
                end = start.copy()
                end[joint] += direction * distance
                if not kine.valid_joints(side, end):
                    continue
                limits = kine.arms[side]['limits']
                if np.min(np.minimum(end - limits[:, 0], limits[:, 1] - end)) < 1.:
                    continue
                faces = kine._wrist_faces(side, end)
                if faces and min(float(np.rad2deg(slack)) for _, slack in faces) < 1.:
                    continue
                previous = float(np.sum(axis) + np.sum(wrist))
                safe = True
                for u in np.linspace(0., 1., 101):
                    q = start + u * (end - start)
                    a, w = violation(kine, side, q)
                    if np.max(a) > MAX_JOINT_OVERRUN_DEG + 1e-6 or (w.size and np.max(w) > MAX_WRIST_OVERRUN_DEG + 1e-6):
                        safe = False; break
                    current = float(np.sum(a) + np.sum(w))
                    if current > previous + 1e-5:
                        safe = False; break
                    previous = current
                if safe:
                    candidates.append((float(distance), joint, direction, end))
                break
    if not candidates:
        raise ValueError(f'{side}: 找不到单轴、单调回到模型限位内的轨迹；请使用厂家软件现场处理')
    _, joint, direction, end = min(candidates, key=lambda row: (row[0], row[1]))
    return {'joint': joint + 1, 'direction': int(direction), 'target': end}


def duration_for(start, goal):
    distance = max(float(np.max(np.abs(goal[s] - start[s]))) for s in SIDES)
    # Quintic max velocity = 1.875 D/T, max acceleration ~= 5.774 D/T^2.
    return max(2., 1.875 * distance / MAX_SPEED_DEG_S,
               math.sqrt(5.774 * distance / MAX_ACCEL_DEG_S2))


def plan(report, config, kine):
    if not report.get('ready_for_wrist_test') or not report.get('read_only') or report.get('motion_commands_sent'):
        raise ValueError('只读基础诊断未通过')
    if report.get('model_verified') != config['model']:
        raise ValueError('控制柜型号与运动学模型不一致')
    start = {}
    for side in SIDES:
        arm = report['arms'][side]
        if arm.get('cur_state') != 0 or arm.get('err_code') != 0 or arm.get('servo_error_words') != [0] * 7:
            raise ValueError(f'{side}: 非无故障待机，拒绝复位')
        if not arm.get('telemetry_valid'):
            raise ValueError(f'{side}: 遥测无效')
        q = np.asarray(arm['joints_deg'], dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError(f'{side}: 关节反馈无效')
        start[side] = q
    rescue = {s: rescue_waypoint(kine, s, start[s]) for s in SIDES}
    affected = [s for s in SIDES if rescue[s] is not None]
    if len(affected) > 1:
        raise ValueError('双臂同时越限，不执行自动复位；请使用厂家软件现场处理')
    intermediate = {s: (rescue[s]['target'] if rescue[s] else start[s]).copy() for s in SIDES}
    goal = {s: np.asarray(config['hardware']['startup_joints_deg'][s], dtype=float) for s in SIDES}
    for side in SIDES:
        if goal[side].shape != (7,) or not np.isfinite(goal[side]).all():
            raise ValueError(f'{side}: 初始姿态目标无效')
        for u in np.linspace(0., 1., 1001):
            q = intermediate[side] + u * (goal[side] - intermediate[side])
            if not kine.valid_joints(side, q):
                raise ValueError(f'{side}: 回初始姿态路径触及关节或腕部限位')
        singular = float(np.linalg.svd(kine.jacobian(side, goal[side]), compute_uv=False)[-1])
        if singular < config['hardware']['minimum_jacobian_singular_value']:
            raise ValueError(f'{side}: 初始姿态接近奇异位')
    phases = []
    if affected:
        side = affected[0]
        phases.append({'name': f'{side} 轻微越限回退', 'control_sides': [side],
                       'start': {s: start[s].tolist() for s in SIDES},
                       'goal': {s: intermediate[s].tolist() for s in SIDES},
                       'duration_s': duration_for(start, intermediate),
                       'joint': rescue[side]['joint']})
    phases.append({'name': '双臂回遥操初始姿态', 'control_sides': list(SIDES),
                   'start': {s: intermediate[s].tolist() for s in SIDES},
                   'goal': {s: goal[s].tolist() for s in SIDES},
                   'duration_s': duration_for(intermediate, goal)})
    return {'source_report': report.get('checked_at'), 'robot_ip': report['robot_ip'],
            'start_joints_deg': {s: start[s].tolist() for s in SIDES},
            'goal_joints_deg': {s: goal[s].tolist() for s in SIDES},
            'rescue_side': affected[0] if affected else None,
            'phases': phases, 'estimated_motion_s': sum(p['duration_s'] for p in phases),
            'warning': '仅验证关节/腕部限位；无法感知现场障碍物，须由现场人员确认整段运动空间和急停。'}


def phase_config(config, sides):
    result = copy.deepcopy(config)
    settings = result['hardware']
    settings.update(control_sides=list(sides), servo_mode='position', vel_ratio=10,
                    acc_ratio=10, startup_move_enabled=False)
    return result


def run_phase(config, kine, phase, robot_ip, phase_log):
    controller = TianjiHardwareOutput(phase_config(config, phase['control_sides']))
    controller.settings['robot_ip'] = robot_ip
    start = {s: np.asarray(phase['start'][s], dtype=float) for s in SIDES}
    goal = {s: np.asarray(phase['goal'][s], dtype=float) for s in SIDES}
    duration = float(phase['duration_s'])
    try:
        actual = controller.connect()
        allowed_drift = .2 if 'joint' in phase else .6
        if any(float(np.max(np.abs(actual[s] - start[s]))) > allowed_drift for s in SIDES):
            raise HardwareError(f'实际关节位置偏离复位计划超过 {allowed_drift:.1f}°，请重新诊断和规划')
        # Replan this phase from measured joints. This is essential after the
        # first phase, whose endpoint can settle within the 0.5° tolerance.
        start = {s: actual[s].copy() for s in SIDES}
        if 'joint' in phase:
            side = phase['control_sides'][0]
            current_rescue = rescue_waypoint(kine, side, start[side])
            if current_rescue is None or current_rescue['joint'] != phase['joint']:
                raise HardwareError('实测越限位置已变化，请重新规划')
            goal[side] = current_rescue['target']
        if phase['control_sides'] == list(SIDES):
            for side in SIDES:
                if not kine.valid_joints(side, start[side]):
                    raise HardwareError(f'{side}: 回退后实测姿态仍越限')
                if not all(kine.valid_joints(side, start[side] + u * (goal[side] - start[side]))
                           for u in np.linspace(0., 1., 1001)):
                    raise HardwareError(f'{side}: 实测位置至初始姿态的路径越限')
        phase_log['measured_start_joints_deg'] = {s: start[s].tolist() for s in SIDES}
        duration = max(duration, duration_for(start, goal))
        phase_log['duration_s'] = duration
        controller.enable()
        deadline = time.monotonic() + 5.
        while not controller.enabled and time.monotonic() < deadline:
            feedback = controller.poll()
            if any(float(np.max(np.abs(feedback['joints'][s] - start[s]))) > .5 for s in SIDES):
                controller.stop('使能期间出现意外关节运动')
                raise HardwareError('使能期间出现意外关节运动，已请求待机')
            time.sleep(.01)
        if not controller.enabled:
            raise HardwareError('未能进入位置模式')
        stable = controller.read_joints()
        if any(float(np.max(np.abs(stable[s] - start[s]))) > .5 for s in SIDES):
            raise HardwareError('使能后出现意外关节运动，已请求待机')
        phase_log['started_at'] = dt.datetime.now().astimezone().isoformat()
        started = time.monotonic()
        samples = max(1, math.ceil(duration * SAMPLE_HZ))
        peak_follow_error = 0.
        for index in range(samples + 1):
            u = index / samples
            blend = 10 * u**3 - 15 * u**4 + 6 * u**5
            target = {s: start[s] + blend * (goal[s] - start[s]) for s in SIDES}
            controller.send_joint_targets(target)
            measured = controller.read_joints()
            if 'joint' in phase:
                side = phase['control_sides'][0]
                initial_axis, initial_wrist = violation(kine, side, start[side])
                axis, wrist = violation(kine, side, measured[side])
                if (np.max(axis) > np.max(initial_axis) + .2 or
                        (wrist.size and np.max(wrist) > np.max(initial_wrist) + .2)):
                    raise HardwareError('越限程度在回退中增大，已请求待机')
            error = max(float(np.max(np.abs(measured[s] - target[s]))) for s in phase['control_sides'])
            peak_follow_error = max(peak_follow_error, error)
            if error > FOLLOW_ERROR_DEG:
                raise HardwareError(f'跟随误差 {error:.2f}° 超过 {FOLLOW_ERROR_DEG:.1f}°')
            for side in SIDES:
                if side not in phase['control_sides'] and float(np.max(np.abs(measured[side] - start[side]))) > .35:
                    raise HardwareError(f'{side}: 未受控手臂移动超过 0.35°')
            if index % SAMPLE_HZ == 0 or index == samples:
                servo = read_servo_errors(controller.robot)
                if any(servo[s] != [0] * 7 for s in SIDES):
                    raise HardwareError('检测到伺服错误字，已请求待机')
                print(f"{phase['name']}: {u:.0%}，最大跟随误差 {error:.2f}°", flush=True)
            wait = started + index * duration / samples - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        deadline = time.monotonic() + 2.
        while time.monotonic() < deadline:
            measured = controller.read_joints()
            if all(float(np.max(np.abs(measured[s] - goal[s]))) < .5 for s in phase['control_sides']):
                break
            time.sleep(.02)
        else:
            raise HardwareError('复位目标未到位（0.5° 容差）')
        if 'joint' in phase and not kine.valid_joints(phase['control_sides'][0], measured[phase['control_sides'][0]]):
            raise HardwareError('回退后实测关节仍越限')
        phase_log['final_joints_deg'] = {s: measured[s].tolist() for s in SIDES}
        phase_log['peak_follow_error_deg'] = peak_follow_error
        phase_log['success'] = True
    finally:
        controller.close()
        phase_log['standby_requested'] = bool(controller._motion_started)


def execute(plan_data, config, report_path, output_path):
    run_log = {'started_at': dt.datetime.now().astimezone().isoformat(),
               'plan': plan_data, 'success': False, 'phases': []}
    try:
        source = dt.datetime.fromisoformat(plan_data['source_report'])
        if (dt.datetime.now().astimezone() - source).total_seconds() > 120:
            raise ValueError('诊断报告已超过 120 秒，请重新执行 home')
        kine = Kinematics(config)
        for phase in plan_data['phases']:
            phase_log = {'name': phase['name'], 'success': False}
            run_log['phases'].append(phase_log)
            run_phase(config, kine, phase, plan_data['robot_ip'], phase_log)
        run_log['success'] = True
    except BaseException as exc:
        run_log['error'] = str(exc)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            run_log['interrupted'] = True
    finally:
        run_log['finished_at'] = dt.datetime.now().astimezone().isoformat()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return run_log


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirmed', action='store_true')
    args = parser.parse_args(argv)
    if args.execute and (not args.confirmed or args.output is None):
        parser.error('运动模式必须有 --confirmed 和 --output')
    report = {}
    kine = None
    try:
        report = json.loads(args.report.read_text(encoding='utf-8'))
        if not isinstance(report, dict):
            raise ValueError('诊断报告必须为 JSON 对象')
        config = load_config(args.config, hardware=True)
        kine = Kinematics(config)
        plan_data = plan(report, config, kine)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        failure = {'success': False, 'plan_ready': False, 'motion_commands_sent': False,
                   'source_report': str(args.report), 'error': str(exc),
                   'limit_diagnostics': {}}
        if kine is not None and isinstance(report, dict):
            for side in SIDES:
                try:
                    failure['limit_diagnostics'][side] = limit_diagnostics(
                        kine, side, report['arms'][side]['joints_deg'])
                except (ValueError, KeyError, TypeError):
                    pass
        text = json.dumps(failure, ensure_ascii=False, indent=2) + '\n'
        if args.execute and args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding='utf-8')
        print(text, end='')
        print(f'复位规划未通过：{exc}', file=sys.stderr)
        return 2
    if not args.execute:
        print(json.dumps(plan_data, ensure_ascii=False, indent=2))
        return 0
    result = execute(plan_data, config, args.report, args.output)
    if result['success']:
        print(f'复位轨迹完成，执行报告：{args.output}')
    else:
        print(f'复位停止：{result.get("error", "未知故障")}；执行报告：{args.output}', file=sys.stderr)
    return 0 if result['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
