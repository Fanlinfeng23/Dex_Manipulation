#!/usr/bin/env python3
"""XRoboToolkit → Tianji launcher, with optional bilateral Wuji teleoperation."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request

APP = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get('TIANJI_WORKSPACE', str(APP.parents[1])))
CAPTURE = Path(os.environ.get('TIANJI_QUEST_CAPTURE_DIR', str(WORKSPACE / 'capture')))
RUNTIME = APP / 'runtime'
STATE = RUNTIME / 'session.json'
CAPTURE_OWNER = RUNTIME / 'capture-owner.json'
PYTHON = Path(os.environ.get('TIANJI_ARM_PYTHON', str(APP / '.venv-linux/bin/python')))
HAND_PYTHON = Path(os.environ.get('TIANJI_HAND_PYTHON', str(WORKSPACE / '.venv/bin/python')))


def load_capture():
    spec = importlib.util.spec_from_file_location('_native_capture_manager', CAPTURE / 'native_test.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.use_tianji_client()
    return module


def read_json(path):
    try:
        value = Path(path).read_text()
        result = json.loads(value)
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


def save_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def process_identity(pid):
    try:
        base = Path('/proc') / str(int(pid))
        stat = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': int(stat[19]),
                'exe': str((base / 'exe').resolve(strict=True))}
    except (OSError, ValueError, IndexError, TypeError):
        return None


def alive(identity):
    return bool(identity and process_identity(identity.get('pid')) == identity)


def run(command, **kwargs):
    timeout = kwargs.pop('timeout', 30)
    return subprocess.run(list(map(str, command)), check=True, timeout=timeout, **kwargs)


def profile(env=None):
    """Exactly the joint arm profile; no hand SDK or calibration dependencies."""
    env = os.environ if env is None else env
    config = read_json(APP / 'configs/hardware_dual_arm.json')
    config['keyboard_auto_resume_tracking'] = True
    config['position_scale'] = 1.0
    config['heading_source'] = env.get('TIANJI_JOINT_HEADING_SOURCE', 'fixed')
    if config['heading_source'] not in ('fixed', 'head'):
        raise ValueError('TIANJI_JOINT_HEADING_SOURCE 只能为 fixed 或 head')
    yaw = float(env.get('TIANJI_JOINT_OPERATOR_YAW_DEG', config['operator_yaw_deg']))
    if not math.isfinite(yaw):
        raise ValueError('TIANJI_JOINT_OPERATOR_YAW_DEG 必须为有限数值')
    config['operator_yaw_deg'] = yaw
    config['initial_source'] = 'quest'
    return config


def health(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=1) as response:
        return json.load(response)


def service_ready(value, hardware):
    common = (value.get('ok') and value.get('hardware') == hardware
              and value.get('input_backend') == 'xrobotoolkit'
              and value.get('source') == 'quest' and not value.get('hardware_fault'))
    return bool(common and (not hardware or (
        value.get('hardware_connected') and value.get('motion_scope') == 'full'
        and set(value.get('control_sides') or []) == {'left', 'right'})))


def require_port_free(port):
    with socket.socket() as sock:
        # Match aiohttp's Linux listener: a recently closed connection in
        # TIME_WAIT must not prevent restarting. An active listener still
        # rejects bind; SO_REUSEPORT is deliberately not enabled.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', port))
        except OSError as exc:
            raise RuntimeError(f'端口 {port} 已被占用，请先退出该服务') from exc


def vive_live_running(proc_root=None):
    """Detect the other arm writer before opening a Quest hardware session."""
    proc_root = Path('/proc') if proc_root is None else Path(proc_root)
    for proc in proc_root.iterdir():
        if not proc.name.isdecimal() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if b'vive_tianji.teleop' in argv and b'live' in argv:
            return True
    return False


def require_idle(capture):
    if vive_live_running():
        raise RuntimeError('已有 Vive 双腕实机遥操；请先退出再启动 Quest 原生会话')
    state = read_json(STATE)
    if any(alive(state.get(key)) for key in ('launcher', 'hand', 'server', 'bridge', 'keyboard')):
        raise RuntimeError('已有原生遥操会话；请先 ./tianji_quest.sh stop')
    capture.assert_no_teleop()
    if capture.active_recorder_pids():
        raise RuntimeError('官方原始采集器仍在录制；请先结束 record 再启动双臂版')


def capture_key(state):
    return {key: state.get(key) for key in ('pid', 'start_ticks', 'serial')}


def start_capture(capture, serial):
    """Reuse only our retained capture session; never alter a foreign one."""
    before = capture.read_state()
    borrowed = capture.service_alive(before)
    if borrowed and read_json(CAPTURE_OWNER) != capture_key(before):
        raise RuntimeError('独立原始采集服务仍运行；请先 ./quest_native_test.sh stop，再启动原生双臂版')
    try:
        state = capture.start(serial)
    except BaseException:
        # The helper persists its child before configuring USB. Recover our
        # partial startup, but never claim a service that was already running.
        after = capture.read_state()
        if not borrowed and capture.service_alive(after) and capture_key(after) != capture_key(before):
            save_json(CAPTURE_OWNER, capture_key(after))
        raise
    if not borrowed:
        save_json(CAPTURE_OWNER, capture_key(state))
    return state


def stop_owned_capture(capture):
    owner = read_json(CAPTURE_OWNER)
    if not owner:
        return False
    current = capture.read_state()
    if capture_key(current) != owner:
        CAPTURE_OWNER.unlink(missing_ok=True)
        return False
    if capture.active_recorder_pids():
        print('PC Service 正被独立原始录制使用，已保留；结束 record 后再执行 stop。', flush=True)
        return False
    capture.stop(owner.get('serial'))
    CAPTURE_OWNER.unlink(missing_ok=True)
    return True


def check_network(ip, interface):
    address = subprocess.check_output(['ip', '-4', 'addr', 'show', 'dev', interface], text=True)
    route = subprocess.check_output(['ip', 'route', 'get', ip], text=True)
    host_cidr = os.environ.get('TIANJI_HOST_CIDR', '192.168.1.50/24')
    if host_cidr not in address or f'dev {interface}' not in route:
        raise RuntimeError(f'机器人专用网卡 {interface} 需有 {host_cidr} 且承载到 {ip} 的路由')
    run(['ping', '-c', '1', '-W', '1', ip], stdout=subprocess.DEVNULL)


def diagnose(args, config_path, log_dir):
    check_network(args.robot_ip, args.interface)
    report = log_dir / 'tianji-readonly.json'
    pose = log_dir / 'tianji-full-arm-pose.json'
    print('只读检查控制柜、双臂反馈与整臂姿态；不清错、不使能。', flush=True)
    run([PYTHON, APP / 'tools/diagnose_robot.py', '--robot-ip', args.robot_ip, '--output', report])
    run([PYTHON, APP / 'tools/check_full_arm_pose.py', '--report', report,
         '--config', config_path, '--output', pose])


def child(command, log, env=None, terminal=False):
    # Each child has its own process group. Only its owner sends shutdown signals.
    with Path(log).open('ab') as out:
        process = subprocess.Popen(list(map(str, command)), cwd=APP, env=env,
                                   stdin=None if terminal else subprocess.DEVNULL,
                                   stdout=None if terminal else out,
                                   stderr=None if terminal else subprocess.STDOUT,
                                   start_new_session=True)
    deadline = time.monotonic() + 1
    while process.poll() is None and time.monotonic() < deadline:
        identity = process_identity(process.pid)
        if identity:
            return process, identity
        time.sleep(.01)
    return process, process_identity(process.pid)


def finish_child(process, name, first_signal=signal.SIGTERM, timeout=12):
    if process is None or process.poll() is not None:
        return
    process.send_signal(first_signal)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f'{name} 未及时退出，正在终止；请确认机械臂实际状态。', file=sys.stderr)
        process.kill()
        process.wait(timeout=3)


def interrupted(_signum, _frame):
    raise KeyboardInterrupt


def hand_environment():
    env = os.environ.copy()
    env.setdefault('WUJI_TELEMETRY_DISABLE', '1')
    return env


def scan_hands(args, log_dir):
    output = log_dir / 'hand-devices.json'
    command = [HAND_PYTHON, WORKSPACE / 'wuji_joint_hand.py', '--check-only', '--device-output', output]
    for side in ('left', 'right'):
        for kind in ('hand', 'glove'):
            serial = getattr(args, f'{side}_{kind}_sn', '')
            if serial:
                command += [f'--{side}-{kind}-sn', serial]
    print('只读识别左右 Hand2 和左右 Glove；四台设备齐全后选择双手标定。', flush=True)
    run(command, env=hand_environment(), timeout=60)
    devices = read_json(output)
    serials = {}
    for side in ('left', 'right'):
        for kind in ('hand', 'glove'):
            value = devices.get(side, {}).get(f'{kind}_sn')
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(f'{side} {kind} 序列号检查未通过')
            serials[f'--{side}-{kind}-sn'] = value
    return serials


def prepare_hands(args, log_dir):
    if os.environ.get('WUJI_DEFAULT_MODEL') == '1':
        raise ValueError('完整联合模式要求左右手均已标定，不接受 Default 模型')
    selected = args.model_user or args.calibrate_user
    model = args.model_user if selected else os.environ.get('WUJI_MODEL_USER')
    calibrate = args.calibrate_user if selected else os.environ.get('WUJI_CALIBRATE_USER')
    if model and calibrate:
        raise ValueError('已有用户与重新标定只能选择一项')
    serials = scan_hands(args, log_dir)
    output = log_dir / 'hand-profile.json'
    command = [HAND_PYTHON, WORKSPACE / 'wuji_hand_profile.py', '--output', output,
               '--left-glove-sn', serials['--left-glove-sn'],
               '--right-glove-sn', serials['--right-glove-sn']]
    if model:
        command += ['--user', model]
    elif calibrate:
        command += ['--calibrate-user', calibrate]
    # Inherit this terminal: its c / user menu and both calibration countdowns
    # are interactive, and can legitimately take several minutes.
    run(command, env=hand_environment(), timeout=None)
    selection = read_json(output)
    user_id = selection.get('user_id')
    if not isinstance(user_id, str) or not user_id.strip() or selection.get('is_default'):
        raise RuntimeError('未获得左右手已标定的具名用户，联合启动已取消')
    model_paths = selection.get('model_paths')
    if not isinstance(model_paths, dict):
        raise RuntimeError('双手模型选择结果缺少左右标定路径')
    paths = []
    for side in ('left', 'right'):
        raw_path = model_paths.get(side)
        path = Path(raw_path) if isinstance(raw_path, str) and raw_path else None
        if path is None or not path.is_file() or path.name != f'{side}_hand.urdf':
            raise RuntimeError(f'{side} 手标定模型缺失或侧别不匹配')
        paths.append(path.resolve())
    if paths[0].parent != paths[1].parent or (paths[0].parent / 'joint_calibration.pending.json').exists():
        raise RuntimeError('左右手模型不是同一操作员，或上次双手标定未完成')
    return {'serials': serials, 'user_id': user_id}


def hand_alive(process, ready_file=None):
    if process is not None and process.poll() is not None:
        raise RuntimeError('左右手通道已退出，联合遥操停止；查看本次 hand.log')
    if ready_file is not None and not ready_file.is_file():
        raise RuntimeError('左右手就绪信号已丢失，联合遥操停止；查看本次 hand.log')


def await_hand_ready(process, ready_file, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hand_alive(process)
        if ready_file.is_file() and ready_file.stat().st_size:
            return
        time.sleep(.05)
    raise RuntimeError('等待左右 Glove 有效骨架超时；机械臂尚未启动，查看本次 hand.log')


def close_hand_gate(state):
    if state.get('hand_gate'):
        Path(state['hand_gate']).unlink(missing_ok=True)


def launch(args):
    joint = args.action == 'joint'
    hardware = args.action in ('start', 'joint')
    if hardware and not sys.stdin.isatty():
        raise RuntimeError('实机启动需在现场交互式终端执行，首次启动会移动双臂至原联合版初始姿态')
    if not hardware and args.seconds is None and not sys.stdin.isatty():
        raise RuntimeError('交互预演需要终端；自动检查可用 preview --seconds 8')
    # Physical teleoperation always waits for an explicit Space press. Old
    # shell profiles may still export AUTO_START=1; they must not defeat this.
    auto = '0' if hardware else os.environ.get('TIANJI_NATIVE_AUTO_START', os.environ.get('WUJI_JOINT_AUTO_START', '1'))
    if auto not in ('0', '1'):
        raise ValueError('TIANJI_NATIVE_AUTO_START / WUJI_JOINT_AUTO_START 只能为 0 或 1')
    capture = load_capture()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / 'launcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('另一个原生双臂启动器正在运行') from exc
        require_idle(capture)
        require_port_free(args.port)
        config = profile()
        stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        log_dir = WORKSPACE / 'logs' / (('native-joint-' if joint else 'native-arm-') + stamp)
        log_dir.mkdir(parents=True)
        config_path = log_dir / 'config.json'
        save_json(config_path, config)
        # Validate the exact profile before any ADB/robot startup side effects.
        from quest_teleop.config import load_config
        load_config(config_path, hardware=hardware)
        hand_selection = prepare_hands(args, log_dir) if joint else None
        if hardware:
            diagnose(args, config_path, log_dir)
            print('\n保留联合版：双臂各 7 轴、1:1 位移、阻抗模式、SDK 60%、3 秒初始轨迹。')
            if joint:
                print('原生联合模式：天机双臂 + 左右 Glove / Hand2；同一键盘统一开始、暂停和退出。')
            else:
                print('不连接 Glove / Hand 控制。末端仍安装灵巧手，沿用原工具与载荷参数。')
            print('官方 APK 未报告每侧追踪有效性；这一版使用数值/冻结门禁，不能保证识别全部失追。')
            print('双臂到达初始姿态后固定保持；准备完毕在本终端按空格，才开始双臂/双手跟随。')
            confirmation = 'JOINT-MOVE' if joint else 'NATIVE-MOVE'
            answer = input(f'确认双臂与手部全路径无障碍、急停在手边，输入 {confirmation} 启动：').strip()
            if answer != confirmation:
                raise RuntimeError('已取消原生双臂启动')
        else:
            print('原生双臂预演：不会连接或驱动机械臂。', flush=True)
        token_path = log_dir / 'session.token'
        with open(token_path, 'x', opener=lambda path, flags: os.open(path, flags, 0o600)) as stream:
            stream.write(secrets.token_urlsafe(32))
        native_status = log_dir / 'native-status.json'
        state = {'launcher': process_identity(os.getpid()), 'hardware': hardware, 'joint': joint,
                 'port': args.port, 'log_dir': str(log_dir), 'config': str(config_path),
                 'native_status': str(native_status), 'started_at': dt.datetime.now().astimezone().isoformat()}
        if joint:
            state.update(hand_gate=str(log_dir / 'hand.follow'), hand_ready=str(log_dir / 'hand.ready'),
                         hand_selection=hand_selection)
        save_json(STATE, state)
        server = bridge = keyboard = hand = None
        capture_started = False
        old_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for sig in old_handlers:
            signal.signal(sig, interrupted)
        outcome = 0
        try:
            # Set ownership before start so even a partial ADB/service launch is cleaned up.
            capture_started = True
            capture_state = start_capture(capture, args.serial)
            verification = capture_state.get('apk_verification', {})
            if not (verification.get('approved_client') or verification.get('official_apk_unmodified')):
                raise RuntimeError('Quest 客户端未通过批准版本的 APK 校验')
            state['serial'] = capture_state['serial']
            save_json(STATE, state)
            if joint:
                # Capture helper checks for other teleop workers; start it before
                # our hand worker. The latter never enables motors while gated.
                hand_args = [HAND_PYTHON, '-u', WORKSPACE / 'wuji_joint_hand.py',
                             '--gate', state['hand_gate'], '--gate-timeout', '0.5',
                             '--ready-file', state['hand_ready'], '--model-user-id', hand_selection['user_id']]
                for option, value in hand_selection['serials'].items():
                    hand_args += [option, value]
                hand, state['hand'] = child(hand_args, log_dir / 'hand.log', env=hand_environment())
                save_json(STATE, state)
                await_hand_ready(hand, Path(state['hand_ready']))
                hand_alive(hand, Path(state['hand_ready']))
            server_args = [PYTHON, '-u', '-m', 'quest_teleop.native_server', '--config', config_path,
                           '--port', args.port, '--native-status-file', native_status,
                           '--token-file', token_path]
            if hardware:
                server_args += ['--hardware', '--robot-ip', args.robot_ip]
            if joint:
                server_args += ['--joint-gate', state['hand_gate']]
            server, state['server'] = child(server_args, log_dir / 'arm.log')
            save_json(STATE, state)
            deadline = time.monotonic() + 50
            ready = False
            while time.monotonic() < deadline:
                if joint:
                    hand_alive(hand, Path(state['hand_ready']))
                if server.poll() is not None:
                    raise RuntimeError(f'双臂服务退出；查看 {log_dir / "arm.log"}')
                try:
                    h = health(args.port)
                    if service_ready(h, hardware):
                        ready = True
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            if not ready:
                raise RuntimeError(f'双臂服务未就绪；查看 {log_dir / "arm.log"}')
            bridge_args = [PYTHON, '-u', '-m', 'quest_teleop.native_bridge', '--port', args.port,
                           '--status-file', native_status, '--log-file', log_dir / 'native-events.jsonl',
                           '--token-file', token_path]
            bridge, state['bridge'] = child(bridge_args, log_dir / 'bridge.log', env=capture.service_env())
            save_json(STATE, state)
            print(f'电脑状态页：http://127.0.0.1:{args.port}/\n本次日志：{log_dir}', flush=True)
            if capture_state.get('auto_setup', {}).get('head_controller_send_verified'):
                print('Quest 默认连接与 Head / Controller / Send 已自动配置并验证。', flush=True)
            else:
                print('官方应用：PC Service=127.0.0.1；Head、Controller、Send 打勾；Switch w/ A Button 关闭。', flush=True)
            print('请唤醒并轻微移动两只手柄；Input=2 不再单独拦截，按两侧 Controller 实际数据检查。', flush=True)
            if not args.no_browser and args.seconds is None:
                with contextlib.suppress(OSError):
                    subprocess.Popen(['xdg-open', f'http://127.0.0.1:{args.port}/'],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if args.seconds is None:
                keyboard_args = [PYTHON, '-u', '-m', 'quest_teleop.keyboard', '--full-arm', '--auto-calibrate',
                                 '--native-arm-mode', '--token-file', token_path,
                                 '--config', config_path, '--port', args.port, '--robot-ip', args.robot_ip]
                if hardware:
                    keyboard_args.append('--hardware')
                if joint:
                    keyboard_args += ['--joint-gate', state['hand_gate']]
                if auto == '1' and not args.manual:
                    keyboard_args.append('--auto-follow-on-ready')
                keyboard, state['keyboard'] = child(keyboard_args, log_dir / 'keyboard.log', terminal=True)
                save_json(STATE, state)
            end = time.monotonic() + args.seconds if args.seconds is not None else math.inf
            last_reason = None
            while time.monotonic() < end:
                if joint:
                    hand_alive(hand, Path(state['hand_ready']))
                if server.poll() is not None or bridge.poll() is not None:
                    raise RuntimeError(f'原生输入或双臂服务退出，已停止本次遥操；日志 {log_dir}')
                if keyboard is not None and keyboard.poll() is not None:
                    outcome = keyboard.returncode
                    break
                status = read_json(native_status)
                reason = status.get('reason_zh') or status.get('reason')
                if reason and reason != last_reason:
                    print('原生采集：' + str(reason), flush=True)
                    last_reason = reason
                time.sleep(.1)
            if args.seconds is not None:
                result = {'health': health(args.port), 'native': read_json(native_status)}
                result['numeric_input_ready'] = bool(service_ready(result['health'], False)
                                                     and result['native'].get('eligible')
                                                     and result['native'].get('acked_frames', 0) > 0)
                result['optical_tracking_verified'] = False
                save_json(log_dir / 'preview-result.json', result)
                if not result['numeric_input_ready']:
                    print('限时预演尚未收到门禁合格的双侧输入；请完成官方应用设置后重试。', flush=True)
                    outcome = 1
        except KeyboardInterrupt:
            outcome = 130
        except BaseException:
            outcome = 2
            raise
        finally:
            # Ignore repeated Ctrl+C while each owner closes its devices normally.
            for sig in old_handlers:
                signal.signal(sig, signal.SIG_IGN)
            close_hand_gate(state)
            finish_child(keyboard, '键盘控制', signal.SIGINT)
            # Closing the keyboard first prevents reopening the gate during exit.
            close_hand_gate(state)
            finish_child(bridge, '原生输入', signal.SIGINT)
            finish_child(server, '机械臂服务', signal.SIGTERM, timeout=20)
            finish_child(hand, '左右 Wuji 手部', signal.SIGINT)
            if joint:
                Path(state['hand_ready']).unlink(missing_ok=True)
            keep_capture = bool(getattr(args, 'keep_capture', False) and outcome == 0)
            if capture_started and not keep_capture:
                with contextlib.suppress(Exception):
                    stop_owned_capture(capture)
            state.update(ended_at=dt.datetime.now().astimezone().isoformat(), exit_code=outcome,
                         capture_kept=keep_capture)
            save_json(log_dir / 'session.json', state)
            current = read_json(STATE)
            if current.get('launcher') == state['launcher']:
                STATE.unlink(missing_ok=True)
            token_path.unlink(missing_ok=True)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            print('原生遥操会话已结束；实机模式已请求待机，记录保留。', flush=True)
            if keep_capture:
                print('已保留 Quest 原生应用和 PC Service，便于直接 start；不用时执行 ./tianji_native.sh stop。', flush=True)
        return outcome


def stop():
    state = read_json(STATE)
    close_hand_gate(state)
    if alive(state.get('launcher')):
        os.kill(state['launcher']['pid'], signal.SIGINT)
        deadline = time.monotonic() + 45
        while alive(state['launcher']) and time.monotonic() < deadline:
            time.sleep(.1)
        if alive(state['launcher']):
            raise RuntimeError('启动器仍在退出；请查看现场状态及日志，不自动接管其他进程')
        print('原生双臂会话已退出。')
        return
    # Recover only processes whose PID, start time and executable still match.
    for key in ('keyboard', 'bridge', 'server', 'hand'):
        identity = state.get(key)
        if not alive(identity):
            continue
        os.kill(identity['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 20
        while alive(identity) and time.monotonic() < deadline:
            time.sleep(.1)
        if alive(identity):
            raise RuntimeError(f'{key} 未退出，请检查现场；未停止其他服务')
    stopped_capture = stop_owned_capture(load_capture())
    STATE.unlink(missing_ok=True)
    if not state and not stopped_capture:
        print('没有本入口拥有的运行会话；未停止其他采集服务。')


def main(argv=None):
    parser = argparse.ArgumentParser(description='XRoboToolkit Quest → 天机双臂；joint 同步左右 Wuji Glove / Hand2')
    parser.add_argument('action', choices=['joint', 'start', 'preview', 'capture', 'tracking', 'check', 'status', 'stop', 'home', 'standby', 'recover', 'right-joints'])
    parser.add_argument('--port', type=int, default=int(os.environ.get('TIANJI_NATIVE_PORT', os.environ.get('TIANJI_QUEST_PORT', '8766'))))
    parser.add_argument('--robot-ip', default=os.environ.get('TIANJI_ROBOT_IP', '192.168.1.190'))
    parser.add_argument('--interface', default=os.environ.get('TIANJI_DEVICE_INTERFACE', 'enxec1ac3115da6'))
    parser.add_argument('--serial')
    parser.add_argument('--manual', action='store_true', help='预演首次也等待空格；实机始终按空格开始')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--keep-capture', action='store_true', help='仅 preview：正常退出后保留原生应用设置，便于随后 start')
    parser.add_argument('--seconds', type=float, help='仅 preview：无人交互限时验证，不开启跟随')
    users = parser.add_mutually_exclusive_group()
    users.add_argument('--model-user', '--model-user-id', dest='model_user')
    users.add_argument('--calibrate-user')
    for side in ('left', 'right'):
        for kind in ('hand', 'glove'):
            names = [f'--{side}-{kind}-sn']
            if side == 'right':
                names += [f'--{kind}-sn']
            default = os.environ.get(f'WUJI_{side.upper()}_{kind.upper()}_SN',
                                     os.environ.get(f'WUJI_{kind.upper()}_SN', '') if side == 'right' else '')
            parser.add_argument(*names, dest=f'{side}_{kind}_sn', default=default)
    parser.add_argument('--check-hands', action='store_true', help='check 同时只读检查左右 Hand2 与 Glove')
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error('port 必须在 1024–65535')
    if args.seconds is not None and (args.action != 'preview' or not math.isfinite(args.seconds) or not 0 < args.seconds <= 600):
        parser.error('--seconds 只适用于 preview，取值 0–600 秒')
    if args.keep_capture and args.action != 'preview':
        parser.error('--keep-capture 只适用于 preview')
    try:
        if args.action in ('joint', 'start', 'preview'):
            return launch(args)
        if args.action == 'stop':
            stop()
        elif args.action == 'status':
            state = read_json(STATE)
            result = {'session': state, 'launcher_running': alive(state.get('launcher'))}
            if state.get('native_status'):
                result['native'] = read_json(state['native_status'])
            with contextlib.suppress(OSError, ValueError):
                result['health'] = health(state.get('port', args.port))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.action == 'tracking':
            # Observe an existing session only; this must never start hardware.
            state = read_json(STATE)
            if not state.get('native_status') or not alive(state.get('bridge')):
                raise RuntimeError('请先运行 ./tianji_quest.sh preview --keep-capture 或 joint，再在另一终端执行 tracking')
            samples = []
            for _ in range(50):
                value = health(state.get('port', args.port))
                samples.append(bool(value.get('native_eligible')))
                time.sleep(.1)
            print(f'5 秒状态采样：数值门禁通过 {sum(samples)}/{len(samples)}；不代表光学追踪覆盖率。')
        elif args.action == 'capture':
            capture = load_capture()
            require_idle(capture)
            RUNTIME.mkdir(parents=True, exist_ok=True)
            start_capture(capture, args.serial)
        elif args.action == 'check':
            capture = load_capture()
            require_idle(capture)
            capture.device(args.serial)
            directory = WORKSPACE / 'logs' / ('native-check-' + dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            directory.mkdir(parents=True)
            config_path = directory / 'config.json'
            save_json(config_path, profile())
            if args.check_hands:
                scan_hands(args, directory)
            diagnose(args, config_path, directory)
            capture.status(args.serial)
        else:
            require_idle(load_capture())
            env = os.environ.copy()
            env['TIANJI_ROBOT_IP'] = args.robot_ip
            env['TIANJI_DEVICE_INTERFACE'] = args.interface
            env['TIANJI_QUEST_PORT'] = str(args.port)
            env['TIANJI_QUEST_APP'] = str(APP)
            run([WORKSPACE / 'tianji_maintenance.sh', args.action], env=env, timeout=None)
        return 0
    except subprocess.CalledProcessError as exc:
        print(f'原生入口：{args.action} 未完成（退出码 {exc.returncode}），原因见上方设备或复位提示。',
              file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'原生双臂：{exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
