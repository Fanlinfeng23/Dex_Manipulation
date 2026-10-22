#!/usr/bin/env python3
"""Read robot versions, model and feedback. Never enable, clear errors or move."""
import argparse
import datetime
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quest_teleop.config import SDK_ROOT, load_config
from quest_teleop.vendor_checks import (bind_robot_abi, controller_model_info,
                                        read_servo_errors, version_info)


def diagnose(ip, seconds=3):
    wrapper = SDK_ROOT / 'SDK_PYTHON/fx_robot.py'
    spec = importlib.util.spec_from_file_location('_readonly_tianji_sdk', wrapper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    robot = module.Marvin_Robot(); bind_robot_abi(robot, module.DCSS)
    connected = False
    result = {'checked_at': datetime.datetime.now().astimezone().isoformat(), 'robot_ip': ip,
              'read_only': True, 'motion_commands_sent': False, 'ready_for_wrist_test': False}
    try:
        connected = bool(robot.connect(ip))
        result['sdk_socket_open'] = connected
        if not connected:
            raise RuntimeError('SDK 端口无法建立；检查是否已有厂家软件占用 UDP 4730')
        result.update(version_info(robot))
        result.update(controller_model_info(robot, load_config(hardware=True)))
        rows = []; deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rows.append(robot.subscribe(module.DCSS())); time.sleep(.02)
        result['samples'] = len(rows)
        result['unique_feedback_frames'] = [len({r['outputs'][i]['frame_serial'] for r in rows
                                                if r['outputs'][i]['frame_serial'] != 0}) for i in range(2)]
        result['connected'] = all(n > 1 for n in result['unique_feedback_frames'])
        servo_errors = read_servo_errors(robot)
        result['arms'] = {}
        for i, side in enumerate(('left', 'right')):
            state = rows[-1]['states'][i]; data = rows[-1]['outputs'][i]
            inputs = rows[-1]['inputs'][i]
            temperatures = np.asarray(data['fb_joint_them'], dtype=float)
            telemetry_arrays = [np.asarray(data[name], dtype=float) for name in
                                ('fb_joint_pos', 'fb_joint_posE', 'fb_joint_cmd',
                                 'fb_joint_cToq', 'fb_joint_sToq', 'fb_joint_them')]
            # A changing output-frame counter alone is not proof that EtherCAT
            # axis feedback is alive: the controller also emits frames with all
            # axis fields zero when the 48 V drive side is unavailable.  Seven
            # exactly-zero temperatures are therefore treated as placeholder
            # telemetry.  A real calibrated zero pose remains valid once live
            # (non-placeholder) drive temperatures are present.
            telemetry_valid = (all(a.shape == (7,) and np.isfinite(a).all()
                                   for a in telemetry_arrays) and
                               np.any(np.abs(temperatures) > 1e-6) and
                               np.all((-20 < temperatures) & (temperatures < 150)))
            result['arms'][side] = {**state, 'error_text': '急停' if state['err_code'] == 13 else
                                   '正常' if state['err_code'] == 0 else '请查询厂家错误表',
                                   'joints_deg': data['fb_joint_pos'], 'velocity_deg_s': data['fb_joint_vel'],
                                   'external_joints_deg': data['fb_joint_posE'],
                                   'feedback_command_deg': data['fb_joint_cmd'],
                                   'temperature_c': data['fb_joint_them'],
                                   'current_torque': data['fb_joint_cToq'],
                                   'sensor_torque': data['fb_joint_sToq'],
                                   'telemetry_valid': bool(telemetry_valid),
                                   'frame_serial': data['frame_serial'],
                                   'input_frame_serial': inputs['in_frame_serial'],
                                   'system_cycle': inputs['sys_cyc'],
                                   'servo_error_words': servo_errors[side],
                                   'tool_dyn': inputs['tool_dyn']}
        result['standby_estop_recovery_candidate'] = all(
            arm['cur_state'] == 100 and arm['cmd_state'] in (-1, 0) and
            arm['err_code'] == 13 and arm['servo_error_words'] == [0] * 7
            for arm in result['arms'].values())
        result['ready_for_wrist_test'] = result['connected'] and all(
            arm['cur_state'] == 0 and arm['err_code'] == 0 and
            arm['servo_error_words'] == [0] * 7 and arm['telemetry_valid']
            for arm in result['arms'].values())
        if not result['ready_for_wrist_test']:
            if all(arm['cur_state'] == 0 and arm['err_code'] == 0 and
                   arm['servo_error_words'] == [0] * 7 and not arm['telemetry_valid']
                   for arm in result['arms'].values()):
                result['failure_class'] = 'axis_telemetry_placeholder'
                result['message'] = ('控制器通信和急停状态正常，但双臂轴反馈仍为占位零值；'
                                     '请检查机械臂 DC48V 驱动电源、EtherCAT/电机电缆，并在重新上电后再检查。')
            elif all(arm['cur_state'] in (1, 3) and arm['err_code'] == 0 and
                     arm['servo_error_words'] == [0] * 7 and arm['telemetry_valid']
                     for arm in result['arms'].values()):
                result['failure_class'] = 'active_mode_requires_standby'
                result['message'] = ('双臂轴反馈正常，但仍在 state=1/3 运动模式；'
                                     '启动新遥操服务前必须先确认转入 state=0 待机。')
    except Exception as exc:
        result['error'] = str(exc)
    finally:
        if connected: robot.release_robot()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-ip', default='192.168.1.190')
    parser.add_argument('--seconds', type=float, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import ipaddress
    ipaddress.IPv4Address(args.robot_ip)
    if not .1 <= args.seconds <= 30: parser.error('--seconds 必须在 0.1–30 之间')
    report = diagnose(args.robot_ip, args.seconds)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)
    sys.exit(0 if report['ready_for_wrist_test'] else 2)
