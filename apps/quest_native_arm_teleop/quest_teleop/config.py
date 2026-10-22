from __future__ import annotations

import json
import math
import ipaddress
import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]
SDK_ROOT = Path(os.environ.get('TIANJI_SDK_ROOT', str(REPO_ROOT / 'TJ_FX_ROBOT_CONTRL_SDK')))
MODELS = {
    'ccs_m6_40': 'CCS M6 · 4.0 示例',
    'ccs_m6_31': 'CCS M6 · 3.1 示例',
    'ccs_m3': 'CCS M3 示例',
    'ccs_m3_2.0_0510': 'M6S Lite · CCS-615',
    'srs': 'SRS 示例',
}


def load_config(path: str | Path | None = None, *, hardware=False) -> dict:
    default = 'hardware_wrist.json' if hardware else 'preview.json'
    config = json.loads(Path(path or APP_ROOT / 'configs' / default).read_text(encoding='utf-8'))
    if config['model'] not in MODELS:
        raise ValueError('Unsupported preview model')
    config['model_label'] = MODELS[config['model']]
    for name in ('control_hz', 'broadcast_hz', 'input_timeout_s', 'frame_age_limit_s',
                 'filter_time_s', 'max_target_speed_m_s', 'max_target_rotation_rad_s',
                 'max_joint_speed_deg_s', 'max_joint_step_deg', 'workspace_radius_m',
                 'controller_jump_m', 'controller_jump_rad', 'position_scale'):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} 必须为正有限数值')
    if not .05 <= config['position_scale'] <= 1:
        raise ValueError('position_scale 必须在 0.05–1.0 之间')
    heading_source = config.setdefault('heading_source', 'head')
    if heading_source not in ('head', 'fixed'):
        raise ValueError('heading_source 必须为 head 或 fixed')
    input_filter = config.setdefault('input_filter', {'type': 'none'})
    if not isinstance(input_filter, dict) or input_filter.get('type') not in ('one_euro', 'none'):
        raise ValueError('input_filter.type 必须为 one_euro 或 none')
    if input_filter['type'] == 'one_euro':
        for name, default in (('min_cutoff_hz', 1.), ('beta', .7),
                              ('derivative_cutoff_hz', 1.), ('position_beta_scale', 20.),
                              ('rotation_beta_scale', 4.)):
            value = input_filter.setdefault(name, default)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'input_filter.{name} 必须为正有限数值')
    for name, default in (('reset_duration_s', 2.), ('max_reset_speed_deg_s', 30.), ('max_reset_accel_deg_s2', 60.)):
        value = config.setdefault(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} 必须是正有限数值')
    grace = config.setdefault('tracking_loss_grace_s', 0.)
    if type(grace) not in (int, float) or not math.isfinite(grace) or not 0 <= grace <= 1.:
        raise ValueError('tracking_loss_grace_s 必须在 0–1 秒之间')
    auto_recover = config.setdefault('keyboard_auto_recover_tracking', False)
    if type(auto_recover) is not bool:
        raise ValueError('keyboard_auto_recover_tracking 必须是布尔值')
    auto_resume = config.setdefault('keyboard_auto_resume_tracking', False)
    if type(auto_resume) is not bool:
        raise ValueError('keyboard_auto_resume_tracking 必须是布尔值')
    arms = config.get('arms')
    if not isinstance(arms, dict) or not all(side in arms for side in ('left', 'right')):
        raise ValueError('arms 必须包含 left/right')
    for side in ('left', 'right'):
        mount = arms[side]
        if not isinstance(mount, dict):
            raise ValueError(f'arms.{side} 必须是对象')
        for name, size in (('controller_to_wrist_m', 3), ('controller_to_wrist_xyzw', 4)):
            values = mount.get(name)
            if (not isinstance(values, list) or len(values) != size or
                    any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
                raise ValueError(f'arms.{side}.{name} 必须为 {size} 个有限数值')
        if math.sqrt(sum(value * value for value in mount['controller_to_wrist_m'])) > .35:
            raise ValueError(f'arms.{side}.controller_to_wrist_m 超过 35 cm；请复查连接件标定')
        norm = math.sqrt(sum(value * value for value in mount['controller_to_wrist_xyzw']))
        if not .99 <= norm <= 1.01:
            raise ValueError(f'arms.{side}.controller_to_wrist_xyzw 必须是单位四元数')
    for name, default in (('ik_position_tolerance_m', .00002), ('ik_rotation_tolerance_rad', .0002),
                          ('ik_rotation_weight_m', .18), ('ik_damping', .002), ('ik_centering', .08)):
        value = config.setdefault(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} 必须是正有限数值')
    count = config.setdefault('ik_max_iterations', 3)
    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError('ik_max_iterations 必须为 1–32 的整数')
    if config.get('visual_model'):
        if config['visual_model'] != 'marvin_m6s_lite_615_v1' or config['model'] != 'ccs_m3_2.0_0510':
            raise ValueError('Lite URDF 需使用 ccs_m3_2.0_0510 运动学；其他模型请将 visual_model 设为 null')
    hardware = config.setdefault('hardware', {})
    if not isinstance(hardware, dict):
        raise ValueError('hardware 必须是对象')
    hardware.setdefault('enabled', False)
    hardware.setdefault('robot_ip', '192.168.1.190')
    hardware.setdefault('vel_ratio', 20)
    hardware.setdefault('acc_ratio', 20)
    hardware.setdefault('motion_scope', 'full')
    hardware.setdefault('servo_mode', 'impedance')
    hardware.setdefault('control_sides', ['left', 'right'])
    if hardware['motion_scope'] not in ('wrist', 'full'):
        raise ValueError('hardware.motion_scope 必须为 wrist 或 full')
    if hardware['servo_mode'] not in ('position', 'impedance'):
        raise ValueError('hardware.servo_mode 必须为 position 或 impedance')
    control_sides = hardware['control_sides']
    if (not isinstance(control_sides, list) or not control_sides or
            any(side not in ('left', 'right') for side in control_sides) or
            len(set(control_sides)) != len(control_sides)):
        raise ValueError('hardware.control_sides 必须是不重复的 left/right 列表')
    for name, default in (('wrist_max_excursion_deg', 3.), ('wrist_max_speed_deg_s', 2.),
                          ('locked_joint_tolerance_deg', .3)):
        value = hardware.setdefault(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'hardware.{name} 必须为正有限数值')
    if hardware['motion_scope'] == 'wrist':
        if hardware['servo_mode'] != 'position':
            raise ValueError('手腕调试使用厂家位置跟随模式')
        if hardware['wrist_max_excursion_deg'] > 3 or hardware['wrist_max_speed_deg_s'] > 2:
            raise ValueError('手腕调试上限为 ±3°、2°/s')
        if hardware['locked_joint_tolerance_deg'] > .5:
            raise ValueError('J1–J5 反馈偏移门限不能超过 0.5°')
    if len(control_sides) == 1:
        if hardware['servo_mode'] != 'position':
            raise ValueError('单臂验证只允许使用厂家位置模式')
    if hardware['motion_scope'] == 'full' and hardware['servo_mode'] == 'position':
        # A bounded startup envelope is useful for staged/single-arm validation,
        # but is deliberately optional for the final dual-arm profile.  Joint
        # limits, velocity limiting and tracking-error checks remain active.
        excursion = hardware.setdefault('joint_excursion_deg', 3.) if len(control_sides) == 1 \
            else hardware.get('joint_excursion_deg')
        if excursion is not None and (type(excursion) not in (int, float) or
                not math.isfinite(excursion) or not 0 < excursion <= 30):
            raise ValueError('整臂位置模式 hardware.joint_excursion_deg 必须为 0–30°')
        inactive_tolerance = hardware.setdefault('inactive_joint_tolerance_deg', .3)
        if (type(inactive_tolerance) not in (int, float) or not math.isfinite(inactive_tolerance) or
                not 0 < inactive_tolerance <= .5):
            raise ValueError('非控制臂反馈偏移门限必须为 0–0.5°')
    if hardware['motion_scope'] == 'full':
        minimum_singular = hardware.get('minimum_jacobian_singular_value')
        if minimum_singular is not None and (type(minimum_singular) not in (int, float) or
                not math.isfinite(minimum_singular) or not 0 < minimum_singular <= .2):
            raise ValueError('整臂奇异值门限必须为 0–0.2')
        startup = hardware.setdefault('startup_move_enabled', False)
        if type(startup) is not bool:
            raise ValueError('hardware.startup_move_enabled 必须为布尔值')
        if startup:
            if control_sides != ['left', 'right']:
                raise ValueError('官方初始姿态轨迹只允许同步控制双臂')
            for name, default, lower, upper in (
                    ('startup_duration_s', 3., .5, 30.), ('startup_settle_s', 3., 0., 10.),
                    ('startup_dt_s', .01, .005, .05), ('startup_tolerance_deg', 5., .1, 5.)):
                value = hardware.setdefault(name, default)
                if (type(value) not in (int, float) or not math.isfinite(value) or
                        not lower <= value <= upper):
                    raise ValueError(f'hardware.{name} 必须在 {lower}–{upper} 之间')
            targets = hardware.get('startup_joints_deg')
            if not isinstance(targets, dict) or set(targets) != {'left', 'right'}:
                raise ValueError('hardware.startup_joints_deg 必须同时包含 left/right')
            for side, values in targets.items():
                if (not isinstance(values, list) or len(values) != 7 or any(
                        type(v) not in (int, float) or not math.isfinite(v) for v in values)):
                    raise ValueError(f'hardware.startup_joints_deg.{side} 必须为 7 个有限角度')
            if hardware['servo_mode'] == 'impedance':
                for name, size in (('tool_kine', 6), ('tool_dyn', 10)):
                    values = hardware.get(name)
                    if (not isinstance(values, list) or len(values) != size or any(
                            type(v) not in (int, float) or not math.isfinite(v) for v in values)):
                        raise ValueError(f'hardware.{name} 必须为 {size} 个有限数值')
    if type(hardware['enabled']) is not bool:
        raise ValueError('hardware.enabled 必须为布尔值')
    ipaddress.IPv4Address(hardware['robot_ip'])
    for name, default in (('feedback_timeout_s', .2), ('max_tracking_error_deg', 8.),
                          ('tracking_error_timeout_s', .2), ('send_clear_timeout_s', .04)):
        value = hardware.setdefault(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'hardware.{name} 必须为正有限数值')
    for name, default in (('joint_k', [2, 2, 2, 1.6, 1, 1, 1]),
                          ('joint_d', [.3, .3, .3, .2, .2, .2, .2])):
        values = hardware.setdefault(name, default)
        if not isinstance(values, list) or len(values) != 7 or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f'hardware.{name} 必须有 7 个正有限数值')
    for name in ('vel_ratio', 'acc_ratio'):
        value = hardware[name]
        if type(value) is not int or not 1 <= value <= 100:
            raise ValueError(f'hardware.{name} 必须为 1–100 的整数')
        if hardware['motion_scope'] == 'wrist' and value != 1:
            raise ValueError('手腕调试固定使用 1% SDK 速度 / 加速度比例')
        if (hardware['motion_scope'] == 'full' and hardware['servo_mode'] == 'position'
                and value != 1):
            raise ValueError('整臂位置模式固定使用 1% SDK 速度 / 加速度比例')
    return config


def configure_runtime(config: dict, *, hardware: bool = False,
                      robot_ip: str | None = None,
                      vel_ratio: int | None = None,
                      acc_ratio: int | None = None) -> dict:
    """Apply command-line-only runtime choices without changing preview.json."""
    settings = config.setdefault('hardware', {})
    settings['enabled'] = bool(hardware)
    if robot_ip is not None:
        ipaddress.IPv4Address(robot_ip)
        settings['robot_ip'] = robot_ip
    if vel_ratio is not None:
        settings['vel_ratio'] = int(vel_ratio)
    if acc_ratio is not None:
        settings['acc_ratio'] = int(acc_ratio)
    if settings.get('motion_scope') == 'wrist' and (settings['vel_ratio'] != 1 or settings['acc_ratio'] != 1):
        raise ValueError('手腕调试固定使用 1% SDK 速度 / 加速度比例')
    return config
