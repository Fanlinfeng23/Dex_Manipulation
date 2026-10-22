#!/usr/bin/env python3
"""Offline kinematic gate for a previously captured read-only robot report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quest_teleop.config import load_config
from quest_teleop.kinematics import Kinematics


def check(report_path: Path, config_path: Path):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    config = load_config(config_path, hardware=True)
    settings = config['hardware']
    sides = tuple(settings['control_sides'])
    result = {
        'source_report': str(report_path),
        'config': str(config_path),
        'control_sides': list(sides),
        'minimum_required': settings['minimum_jacobian_singular_value'],
        'ready_for_full_arm': False,
        'arms': {},
    }
    if not report.get('ready_for_wrist_test'):
        result['message'] = '基础真机门禁未通过'
        return result
    kine = Kinematics(config)
    for side in sides:
        q = np.asarray(report['arms'][side]['joints_deg'], dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            result['failure_class'] = 'invalid_joint_feedback'
            result['message'] = f'{side}: 关节反馈不是有限的 7 轴角度'
            return result
        if not kine.valid_joints(side, q):
            limits = kine.arms[side]['limits']
            violations = np.maximum(limits[:, 0] - q, 0) + np.maximum(q - limits[:, 1], 0)
            wrist = [max(0., -float(np.rad2deg(slack)))
                     for _, slack in kine._wrist_faces(side, q)]
            result['arms'][side] = {
                'joints_deg': q.tolist(),
                'joint_limit_overrun_deg': violations.tolist(),
                'wrist_coupling_overrun_deg': wrist,
                'passed': False,
            }
            result['failure_class'] = 'joint_feedback_out_of_limits'
            axis = int(np.argmax(violations))
            detail = f'J{axis + 1} 越界 {violations[axis]:.3f}°' if violations[axis] else '关节单轴限位内'
            if wrist and max(wrist) > 0:
                detail += f'，J6/J7 耦合越界 {max(wrist):.3f}°'
            result['message'] = f'{side}: 关节反馈越限（{detail}）；可在现场执行 ./tianji_quest.sh home'
            return result
        singular = np.linalg.svd(kine.jacobian(side, q), compute_uv=False)
        minimum = float(singular[-1])
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float('inf')
        passed = minimum >= settings['minimum_jacobian_singular_value']
        result['arms'][side] = {
            'joints_deg': q.tolist(),
            'minimum_jacobian_singular_value': minimum,
            'jacobian_condition_number': condition,
            'passed': passed,
        }
    if settings.get('startup_move_enabled', False):
        result['startup_move'] = {'enabled': True, 'trajectory_valid': False, 'arms': {}}
        samples = 301
        for side in sides:
            start = np.asarray(report['arms'][side]['joints_deg'], dtype=float)
            goal = np.asarray(settings['startup_joints_deg'][side], dtype=float)
            valid = all(kine.valid_joints(side, start + u * (goal - start))
                        for u in np.linspace(0., 1., samples))
            singular = np.linalg.svd(kine.jacobian(side, goal), compute_uv=False)
            minimum = float(singular[-1])
            passed = valid and minimum >= settings['minimum_jacobian_singular_value']
            result['startup_move']['arms'][side] = {
                'target_joints_deg': goal.tolist(),
                'joint_path_samples': samples,
                'joint_path_within_limits': valid,
                'target_minimum_jacobian_singular_value': minimum,
                'passed': passed,
            }
        result['startup_move']['trajectory_valid'] = all(
            arm['passed'] for arm in result['startup_move']['arms'].values())
        result['ready_for_full_arm'] = result['startup_move']['trajectory_valid']
        if result['ready_for_full_arm']:
            result['message'] = ('当前双臂可按 Wuji 官方同步五次轨迹进入初始姿态；'
                                 '初始姿态通过关节限位和奇异位门禁')
        else:
            result['failure_class'] = 'invalid_startup_trajectory'
            result['message'] = '官方初始姿态或关节空间路径超出当前 Lite 模型限制'
        return result
    result['ready_for_full_arm'] = all(arm['passed'] for arm in result['arms'].values())
    if result['ready_for_full_arm']:
        result['message'] = '所有受控机械臂起始姿态通过奇异位门禁'
    else:
        failed = [side for side, arm in result['arms'].items() if not arm['passed']]
        result['failure_class'] = 'near_kinematic_singularity'
        result['failed_sides'] = failed
        result['message'] = (
            f'{", ".join(failed)} 当前姿态接近运动学奇异位；'
            '请用厂家软件在现场低速调整对应手臂的肘部关节，'
            '保留充足环境间隙后重试。')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    outcome = check(args.report, args.config)
    text = json.dumps(outcome, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)
    raise SystemExit(0 if outcome['ready_for_full_arm'] else 2)
