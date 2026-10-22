#!/usr/bin/env python3
"""Capture-only launcher; official by default, with an opt-in Tianji adapter."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
SERVICE = Path(os.environ.get('TIANJI_XROBOT_SERVICE_DIR',
                              str(ROOT / 'native/opt/apps/roboticsservice')))
EXE = SERVICE / 'RoboticsServiceProcess'
SDK = SERVICE / 'SDK/x64/libPXREARobotSDK.so'
STATE = ROOT / 'service-state.json'
RECORD_STATE = ROOT / 'record-state.json'
ASSETS = ROOT / 'official-assets.json'
RECORDER = ROOT / 'raw_recorder.py'
PACKAGE = 'com.xrobotoolkit.client.quest'
COMPONENT = PACKAGE + '/com.unity3d.player.UnityPlayerActivity'
EXPECTED_APK_VERSION = '1.0.1'
CUSTOM_PROFILE = None


def use_tianji_client():
    """Opt in to the separately installed, hash-pinned startup adapter."""
    global PACKAGE, COMPONENT, CUSTOM_PROFILE
    path = ROOT / 'tianji-client.json'
    mode = os.environ.get('TIANJI_QUEST_CLIENT', 'auto')
    if mode not in ('auto', 'official'):
        raise ValueError('TIANJI_QUEST_CLIENT 只能为 auto 或 official')
    if mode == 'official' or not path.exists():
        return
    profile = json.loads(path.read_text())
    apk_path = Path(profile['apk'])
    if not apk_path.is_absolute():
        apk_path = ROOT / apk_path
    if (profile.get('package') != 'com.xrobotoolkit.client.quest.tianji' or
            profile.get('component') != 'com.xrobotoolkit.client.quest.tianji/com.tianji.xrobotoolkit.AutoSetupActivity' or
            file_sha256(apk_path) != profile.get('sha256')):
        raise RuntimeError('自动配置客户端清单或 APK 校验失败；未启动任何采集/运动')
    profile['apk'] = str(apk_path)
    CUSTOM_PROFILE = profile
    PACKAGE, COMPONENT = profile['package'], profile['component']


def run(args, *, check=True):
    return subprocess.run(args, text=True, capture_output=True, timeout=20, check=check)


def read_state():
    try:
        state = json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(state, dict):
        return {}
    # One-time migration of a service started before PID start ticks were saved.
    if 'pid' in state and 'start_ticks' not in state:
        ticks = process_start_ticks(state['pid'], EXE)
        if ticks is not None:
            state['start_ticks'] = ticks
            save_state(state)
    return state


def save_state(state):
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temp.replace(STATE)


def read_record_state():
    try:
        state = json.loads(RECORD_STATE.read_text())
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def save_record_state(state):
    temp = RECORD_STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temp.replace(RECORD_STATE)


def proc_stat(pid):
    """Return /proc state and start time; comm may contain spaces or parentheses."""
    try:
        raw = Path(f'/proc/{int(pid)}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return fields[0], int(fields[19])  # fields 3 and 22 of proc(5)
    except (OSError, ValueError, IndexError, TypeError):
        return None


def process_start_ticks(pid, executable):
    stat = proc_stat(pid)
    if stat is None or stat[0] == 'Z':
        return None
    try:
        if Path(f'/proc/{int(pid)}/exe').resolve(strict=True) == Path(executable).resolve(strict=True):
            return stat[1]
    except (OSError, ValueError, TypeError):
        pass
    return None


def process_alive(pid, start_ticks, executable, required_arg=None):
    if start_ticks is None or process_start_ticks(pid, executable) != start_ticks:
        return False
    if required_arg is None:
        return True
    try:
        return required_arg in Path(f'/proc/{int(pid)}/cmdline').read_bytes().split(b'\0')
    except (OSError, ValueError, TypeError):
        return False


def wait_start_ticks(proc, executable, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and proc.poll() is None:
        ticks = process_start_ticks(proc.pid, executable)
        if ticks is not None:
            return ticks
        time.sleep(.01)
    return None


def service_alive(state):
    try:
        return process_alive(state['pid'], state['start_ticks'], EXE)
    except (KeyError, ValueError, TypeError):
        return False


def collector_alive(state):
    try:
        return process_alive(state['collector_pid'], state['collector_start_ticks'],
                             sys.executable, os.fsencode(str(RECORDER)))
    except (KeyError, ValueError, TypeError):
        return False


def logcat_alive(state):
    try:
        return process_alive(state['logcat_pid'], state['logcat_start_ticks'],
                             state['logcat_executable'], b'logcat')
    except (KeyError, ValueError, TypeError):
        return False


def service_env():
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = ':'.join(map(str, [SERVICE, SERVICE / 'lib', SERVICE / 'SDK/x64']))
    env['QT_PLUGIN_PATH'] = str(SERVICE / 'plugins')
    env['QT_QML_PATH'] = str(SERVICE / 'qml')
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env['XDG_DATA_HOME'] = str(ROOT / 'runtime/data')
    (Path(env['XDG_DATA_HOME']) / 'PICOBusinessSuitData').mkdir(parents=True, exist_ok=True)
    return env


def device(preferred=None):
    rows = run(['adb', 'devices']).stdout.splitlines()[1:]
    states = {}
    for row in rows:
        fields = row.split()
        if len(fields) >= 2:
            states[fields[0]] = 'no permissions' if row.find('no permissions') >= 0 else fields[1]
    connected = [serial for serial, state in states.items() if state == 'device']

    def unavailable(serial):
        state = states[serial]
        if state == 'no permissions':
            return (f'Quest {serial} 已通过 USB 检出，但 ADB 无 USB 权限（no permissions）；'
                    '请按 Quest天机遥操启动指南.md 安装 Quest udev 规则后重试')
        if state == 'unauthorized':
            return f'Quest {serial} 的 ADB 尚未授权；请在头显中允许这台电脑的 USB 调试'
        if state == 'offline':
            return f'Quest {serial} 的 ADB 处于 offline；请检查 USB 线、唤醒头显并重新连接'
        return f'Quest {serial} 的 ADB 状态为 {state}；请运行 adb devices -l 检查'

    if preferred:
        if preferred not in connected:
            if preferred in states:
                raise RuntimeError(unavailable(preferred))
            raise RuntimeError(f'Quest {preferred} 未被 ADB 检出；请运行 adb devices -l 检查连接')
        return preferred
    if len(connected) > 1:
        raise RuntimeError('检测到多台已授权 ADB 设备；请使用 --serial 指定 Quest')
    if not connected:
        if len(states) == 1:
            raise RuntimeError(unavailable(next(iter(states))))
        if states:
            raise RuntimeError('没有可用的 ADB 设备；请运行 adb devices -l 检查各设备状态')
        raise RuntimeError('ADB 未检出 Quest；请检查 USB 数据线、头显开机与开发者模式')
    return connected[0]


def adb(serial, *args, check=True):
    return run(['adb', '-s', serial, *args], check=check)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def official_apk_reference():
    try:
        assets = json.loads(ASSETS.read_text())['assets']
        asset = next(item for item in assets if item['url'].endswith(
            f'/v{EXPECTED_APK_VERSION}/XRoboToolkit-Quest-{EXPECTED_APK_VERSION}.apk'))
        path = Path(asset['file'])
        if not path.is_absolute():
            path = ROOT / path
        digest = file_sha256(path)
        if digest != asset['sha256'].lower():
            raise ValueError('官方 APK 本地文件与资产清单 SHA256 不符')
        return digest, None
    except (OSError, ValueError, KeyError, StopIteration, TypeError) as exc:
        return None, f'{type(exc).__name__}: {exc}'


def verify_installed_apk(serial, installed_stdout):
    paths = [line.partition(':')[2].strip() for line in installed_stdout.splitlines()
             if line.startswith('package:')]
    selected = next((path for path in paths if path.endswith('/base.apk')),
                    paths[0] if paths else None)
    reference_sha, reference_error = official_apk_reference()
    result = {'installed_apk_path': selected, 'installed_apk_sha256': None,
              'official_apk_sha256': reference_sha, 'apk_version': None,
              'official_apk_unmodified': False}
    errors = []
    if reference_error:
        errors.append(f'本地官方资产校验失败：{reference_error}')
    if not selected:
        errors.append('Quest 未报告 APK 路径')
    else:
        remote = adb(serial, 'shell', 'sha256sum', selected, check=False)
        match = re.match(r'^([0-9a-fA-F]{64})\s', remote.stdout.strip())
        if match and remote.returncode == 0:
            result['installed_apk_sha256'] = match.group(1).lower()
        else:
            errors.append('无法读取 Quest 已安装 APK 的 SHA256')
    package_info = adb(serial, 'shell', 'dumpsys', 'package', PACKAGE, check=False)
    version = re.search(r'\bversionName=([^\s]+)', package_info.stdout)
    if version and package_info.returncode == 0:
        result['apk_version'] = version.group(1)
    else:
        errors.append('无法读取 Quest 已安装 APK 的版本')
    result['official_apk_unmodified'] = bool(
        reference_sha and result['installed_apk_sha256'] == reference_sha
        and result['apk_version'] == EXPECTED_APK_VERSION)
    result['approved_client'] = result['official_apk_unmodified']
    result['client_kind'] = 'official'
    if CUSTOM_PROFILE is not None:
        result['official_apk_unmodified'] = False
        result['client_kind'] = 'tianji_auto_setup'
        result['approved_client'] = bool(reference_sha and not reference_error and
            CUSTOM_PROFILE.get('upstream_sha256') == reference_sha and
            result['installed_apk_sha256'] == CUSTOM_PROFILE['sha256'] and
            result['apk_version'] == CUSTOM_PROFILE['version'])
        result['expected_client_sha256'] = CUSTOM_PROFILE['sha256']
    if not result['approved_client'] and not errors:
        errors.append('Quest 已安装 APK 与选定客户端的批准资产或版本不符')
    result['apk_verification_error'] = '；'.join(errors) if errors else None
    return result


def unity_log(serial, pid):
    return adb(serial, 'logcat', '-d', '--pid=' + pid, '-v', 'brief',
               '-s', 'Unity:I', 'TianjiAutoSetup:I', 'AndroidRuntime:E', '*:S').stdout


def configure_tianji_client(serial, timeout_s=35.):
    """Wait for a Unity scene echo, then apply defaults once per app process.

    The native Activity owns the latch; retries never flip toggles twice. This
    helper does not send controller or robot motion commands.
    """
    nonce = 'TianjiAutoProbe_' + uuid.uuid4().hex
    deadline = time.monotonic() + timeout_s
    pid = ''
    while time.monotonic() < deadline:
        pid = adb(serial, 'shell', 'pidof', PACKAGE, check=False).stdout.strip()
        if pid.isdigit():
            adb(serial, 'shell', 'am', 'start', '-n', COMPONENT, '--es', 'tianji_probe', nonce)
            time.sleep(.2)
            if 'AndroidCall:' + nonce in unity_log(serial, pid):
                break
        time.sleep(.3)
    else:
        raise RuntimeError('Quest 自动配置应用未进入前台；请处理系统锁屏/安全边界后重试。机械臂尚未启动。')
    adb(serial, 'shell', 'am', 'start', '-n', COMPONENT, '--ez', 'tianji_defaults', 'true')
    deadline = time.monotonic() + 8.
    while time.monotonic() < deadline:
        if adb(serial, 'shell', 'pidof', PACKAGE, check=False).stdout.strip() != pid:
            raise RuntimeError('Quest 自动配置期间进程重启，请重新启动本次采集')
        log = unity_log(serial, pid)
        if 'AndroidCall:TianjiAutoConfigured|v1' in log:
            if any(marker in log for marker in ('InternalToggle has no receiver', 'InternalToggle has no receiver!')):
                raise RuntimeError('Quest 无法应用默认开关；未启动机械臂')
            return {'pid': int(pid), 'scene_probe': nonce, 'defaults_requested': True}
        time.sleep(.1)
    raise RuntimeError('Quest 默认配置没有收到场景回执；未启动机械臂')


def verify_tianji_stream():
    """Prove Head + both Controller fields arrive with an advancing app clock."""
    output = ROOT / 'logs' / ('auto-setup-' + dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.jsonl')
    result = subprocess.run([sys.executable, str(RECORDER), '--seconds', '.5',
                             '--wait-seconds', '12', '--output', str(output)],
                            env=service_env(), text=True, capture_output=True, timeout=20)
    last_time = None
    new_frames = 0

    def numeric_pose(node):
        if not isinstance(node, dict):
            return False
        raw = node.get('pose')
        if not isinstance(raw, str):
            return False
        try:
            values = [float(value) for value in raw.split(',')]
        except ValueError:
            return False
        return (len(values) == 7 and all(math.isfinite(value) for value in values) and
                .95 <= math.sqrt(sum(value * value for value in values[3:])) <= 1.05)
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            if row.get('event_type') != 'PXREADeviceStateJson':
                continue
            try:
                value = json.loads(row['payload'])
                if value.get('functionName') != 'Tracking':
                    continue
                value = value.get('value')
                for _ in range(3):
                    if not isinstance(value, str):
                        break
                    value = json.loads(value)
                if (isinstance(value, dict) and numeric_pose(value.get('Head')) and
                        isinstance(value.get('Controller'), dict) and
                        all(numeric_pose(value['Controller'].get(side)) for side in ('left', 'right')) and
                        type(value.get('predictTime')) in (int, float) and math.isfinite(value['predictTime'])):
                    stamp = value['predictTime']
                    if last_time is not None and stamp < last_time:
                        new_frames = 0
                    if last_time is None or stamp != last_time:
                        new_frames += 1
                    last_time = stamp
            except (KeyError, ValueError, TypeError):
                continue
    if result.returncode != 0 or new_frames < 3:
        raise RuntimeError(f'Quest 自动设置后未收到完整 Head / Controller 新帧；保持机器人未启动。记录：{output}')
    return {'head_controller_send_verified': True, 'distinct_app_frames': new_frames, 'recording': str(output)}


def listener(port):
    """Check LISTEN sockets without opening a connection to PC Service."""
    for name in ('tcp', 'tcp6'):
        try:
            rows = Path(f'/proc/net/{name}').read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            parts = row.split()
            try:
                if parts[3] == '0A' and int(parts[1].rsplit(':', 1)[1], 16) == port:
                    return True
            except (IndexError, ValueError):
                continue
    return False


def assert_no_teleop():
    # Inspect executable arguments, rather than matching arbitrary shell text.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if any(Path(arg).name in {'wuji_joint_hand.py', 'joint_teleop.sh'} or arg == 'quest_teleop'
               or arg.startswith('quest_teleop.') for arg in args if arg):
            raise RuntimeError(f'检测到原遥操进程 PID {proc.name}，请先正常退出再启动原生采集测试')


def start(preferred=None):
    assert_no_teleop()
    serial = device(preferred)
    installed = adb(serial, 'shell', 'pm', 'path', PACKAGE, check=False)
    if 'package:' not in installed.stdout:
        raise RuntimeError(f'未安装所选 Quest 客户端：{PACKAGE}，请按指南安装对应 APK')
    apk_verification = verify_installed_apk(serial, installed.stdout)
    if CUSTOM_PROFILE is not None and not apk_verification['approved_client']:
        raise RuntimeError('已安装的自动配置客户端不在批准的 SHA256 清单中，请重装已保存的定制 APK')
    state = read_state()
    if state.get('serial') and state['serial'] != serial and service_alive(state):
        raise RuntimeError('已有另一台 Quest 的测试会话，请先 stop')
    if not service_alive(state):
        if listener(63901) or listener(60061):
            raise RuntimeError('63901 或 60061 已被其他服务占用；未停止其他服务')
        if not EXE.is_file() or not SDK.is_file():
            raise RuntimeError('官方 PC Service 文件缺失')
        logs = ROOT / 'logs'
        logs.mkdir(exist_ok=True)
        log = logs / ('pc-service-' + dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '.log')
        with log.open('ab') as out:
            proc = subprocess.Popen([str(EXE)], cwd=SERVICE, env=service_env(),
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        ticks = wait_start_ticks(proc, EXE)
        if ticks is None:
            raise RuntimeError(f'PC Service 未能启动，日志：{log}')
        state.update(pid=proc.pid, start_ticks=ticks, executable=str(EXE),
                     log=str(log), serial=serial)
        save_state(state)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and proc.poll() is None:
            if listener(63901) and listener(60061):
                break
            time.sleep(.2)
        if proc.poll() is not None or not (listener(63901) and listener(60061)):
            raise RuntimeError(f'PC Service 未就绪，日志：{log}；可执行 stop 清理本次进程')
    state['serial'] = serial
    state['apk_verification'] = apk_verification
    state['package'] = PACKAGE
    state.pop('auto_setup', None)
    reverse = adb(serial, 'reverse', '--list').stdout
    entries = [row.split() for row in reverse.splitlines()]
    existing = next((row[-1] for row in entries if len(row) >= 3 and row[-2] == 'tcp:63901'), None)
    if existing and existing != 'tcp:63901':
        raise RuntimeError('Quest 63901 已转发至其他地址；未覆盖现有配置')
    if not existing:
        adb(serial, 'reverse', 'tcp:63901', 'tcp:63901')
        state['owns_reverse'] = True
    state['offface_override'] = True
    save_state(state)
    adb(serial, 'shell', 'am', 'broadcast', '-a', 'com.oculus.vrpowermanager.prox_close')
    adb(serial, 'shell', 'input', 'keyevent', '224')
    if not adb(serial, 'shell', 'pidof', PACKAGE, check=False).stdout.strip():
        adb(serial, 'shell', 'am', 'start', '-n', COMPONENT)
    if CUSTOM_PROFILE is not None:
        state['auto_setup'] = {**configure_tianji_client(serial), **verify_tianji_stream()}
        save_state(state)
        print('Quest 已自动设置 127.0.0.1 + Head / Controller / Send，并验证收到完整新帧；机械臂尚未启动。', flush=True)
    else:
        print('官方原生采集已就绪（仅输入服务）。', flush=True)
        print('Quest 原版 UI：PC Service 填写 127.0.0.1；勾选 Head、Controller 和 Send。', flush=True)
    if not apk_verification['approved_client']:
        print('警告：Quest APK 未通过官方 SHA256/版本验证；记录将如实标记为未验证。', flush=True)
    return state


def wait_until_gone(alive, state, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not alive(state):
            return True
        time.sleep(.1)
    return not alive(state)


def stop_collector(record_state):
    if not collector_alive(record_state):
        return True
    pid = record_state['collector_pid']
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        return True
    if wait_until_gone(collector_alive, record_state, 15):
        return True
    if collector_alive(record_state):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
    if wait_until_gone(collector_alive, record_state, 5):
        return False
    if collector_alive(record_state):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return False
    wait_until_gone(collector_alive, record_state, 3)
    return False


def stop(preferred=None):
    state = read_state()
    record_state = read_record_state()
    serial = state.get('serial') or record_state.get('serial')
    if preferred and serial and preferred != serial:
        raise RuntimeError(f'--serial {preferred} 与当前测试会话 {serial} 不符；未停止任何进程')
    if preferred and not serial:
        raise RuntimeError('没有对应 --serial 的测试会话；未停止任何进程')
    graceful = stop_collector(record_state)
    if record_state and not graceful:
        print('警告：采集器未能在 SIGINT 后正常落盘，已请求终止。', file=sys.stderr)
    # The record command writes its session metadata before removing this file.
    deadline = time.monotonic() + 5
    while RECORD_STATE.exists() and time.monotonic() < deadline:
        if not collector_alive(record_state) and not process_alive(
                record_state.get('owner_pid'), record_state.get('owner_start_ticks'),
                sys.executable, os.fsencode(str(Path(__file__).resolve()))):
            break
        time.sleep(.1)
    if RECORD_STATE.exists() and not collector_alive(record_state):
        RECORD_STATE.unlink(missing_ok=True)
    if logcat_alive(record_state):
        try:
            os.kill(record_state['logcat_pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
        wait_until_gone(logcat_alive, record_state, 3)
        if logcat_alive(record_state):
            try:
                os.kill(record_state['logcat_pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
    serial = serial or preferred
    if serial:
        for args in [('shell', 'am', 'force-stop', state.get('package', PACKAGE)),
                     ('shell', 'am', 'broadcast', '-a', 'com.oculus.vrpowermanager.automation_disable')]:
            try:
                adb(serial, *args, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if state.get('owns_reverse'):
            try:
                current = adb(serial, 'reverse', '--list', check=False).stdout
                if any(row.split()[-2:] == ['tcp:63901', 'tcp:63901'] for row in current.splitlines()):
                    adb(serial, 'reverse', '--remove', 'tcp:63901', check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
    if service_alive(state):
        try:
            os.kill(state['pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
        wait_until_gone(service_alive, state, 3)
        if service_alive(state):
            try:
                os.kill(state['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
    STATE.unlink(missing_ok=True)
    print('原生测试服务已停止；已请求恢复 Quest 离脸休眠。已保存记录保留。')


def status(preferred=None):
    state = read_state()
    record_state = read_record_state()
    if preferred and state.get('serial') and preferred != state['serial']:
        raise RuntimeError(f'--serial {preferred} 与当前测试会话 {state["serial"]} 不符')
    result = {**state, 'capture_only': True, 'service_running': service_alive(state),
              'collector_running': collector_alive(record_state),
              'recording': record_state if record_state else None,
              'device_tcp_63901': listener(63901), 'sdk_grpc_60061': listener(60061)}
    try:
        serial = device(preferred or state.get('serial'))
        result['adb_serial'] = serial
        result['quest_app_pid'] = adb(serial, 'shell', 'pidof', PACKAGE, check=False).stdout.strip()
        result['reverse'] = adb(serial, 'reverse', '--list').stdout.strip()
    except (RuntimeError, subprocess.SubprocessError) as exc:
        result['adb_error'] = str(exc)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def active_recorder_pids():
    found = []
    expected = os.fsencode(str(RECORDER))
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if expected in args:
            found.append(int(proc.name))
    return found


def prepare_record(args):
    if not all(math.isfinite(value) and value > 0 for value in
               (args.seconds, args.wait_seconds)) or args.seconds > 86400 or args.wait_seconds > 3600:
        raise RuntimeError('seconds 需为 (0,86400]，wait-seconds 需为 (0,3600] 的有限数')
    if not RECORDER.is_file() or not SDK.is_file():
        raise RuntimeError('采集器或官方 C SDK 文件未就绪')
    output = (args.output or (ROOT / 'logs' / ('quest-native-' + dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '.jsonl'))).resolve()
    for artifact in (output, Path(str(output) + '.summary.json'),
                     output.with_suffix('.session.json'), output.with_suffix('.quest-logcat.txt')):
        if artifact.exists():
            raise RuntimeError(f'输出已存在，未覆盖：{artifact}')
    active = active_recorder_pids()
    if active:
        raise RuntimeError(f'已有原生采集器 PID {active}，请先等待或结束当前记录')
    previous = read_record_state()
    if previous and collector_alive(previous):
        raise RuntimeError('已有原生采集会话，请先结束当前记录')
    if RECORD_STATE.exists():
        RECORD_STATE.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    args.output = output
    return output


def record(args, state):
    output = args.output
    print(f'原始记录：{output}\n等待头显启用 Send；结束后执行 ./quest_native_test.sh stop。', flush=True)
    logcat = None
    collector = None
    record_state = None
    quest_log = output.with_suffix('.quest-logcat.txt')
    metadata = {'started_at': dt.datetime.now().astimezone().isoformat(), 'serial': state['serial'],
                'capture_only': True,
                'requested_seconds': args.seconds, 'wait_seconds': args.wait_seconds,
                'data': str(output), 'quest_logcat': str(quest_log),
                'tracking_validity': 'unknown: official APK does not send per-controller validity'}
    metadata.update(state.get('apk_verification') or {
        'official_apk_unmodified': False, 'apk_verification_error': '本会话未执行安装包验证'})
    result = 2
    try:
        with quest_log.open('x') as log:
            pid = adb(state['serial'], 'shell', 'pidof', PACKAGE, check=False).stdout.strip().split()
            if pid:
                logcat = subprocess.Popen(['adb', '-s', state['serial'], 'logcat', '-T', '1', '--pid=' + pid[0]],
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            collector = subprocess.Popen([sys.executable, str(RECORDER),
                                          '--sdk-lib', str(SDK), '--seconds', str(args.seconds),
                                          '--wait-seconds', str(args.wait_seconds), '--output', str(output)],
                                         env=service_env(), cwd=ROOT, start_new_session=True)
            ticks = wait_start_ticks(collector, sys.executable)
            if ticks is not None:
                record_state = {'collector_pid': collector.pid, 'collector_start_ticks': ticks,
                                'owner_pid': os.getpid(),
                                'owner_start_ticks': process_start_ticks(os.getpid(), sys.executable),
                                'serial': state['serial'], 'output': str(output)}
                if logcat:
                    record_state.update(logcat_pid=logcat.pid,
                                        logcat_start_ticks=wait_start_ticks(logcat, shutil.which('adb')),
                                        logcat_executable=shutil.which('adb'))
                save_record_state(record_state)
            try:
                result = collector.wait()
            except KeyboardInterrupt:
                if record_state:
                    stop_collector(record_state)
                elif collector.poll() is None:
                    collector.send_signal(signal.SIGINT)
                collector.wait()
                result = 130
    finally:
        if collector and collector.poll() is None:
            if record_state:
                stop_collector(record_state)
            else:
                collector.send_signal(signal.SIGINT)
            try:
                collector.wait(timeout=5)
            except subprocess.TimeoutExpired:
                collector.kill()
                collector.wait()
        if collector and result == 2 and collector.returncode is not None:
            result = collector.returncode
        if logcat and logcat.poll() is None:
            logcat.terminate()
            try:
                logcat.wait(timeout=3)
            except subprocess.TimeoutExpired:
                logcat.kill()
                logcat.wait()
        metadata.update(ended_at=dt.datetime.now().astimezone().isoformat(), recorder_exit_code=result)
        try:
            output.with_suffix('.session.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        finally:
            if record_state:
                current = read_record_state()
                if (current.get('collector_pid') == record_state['collector_pid'] and
                        current.get('collector_start_ticks') == record_state['collector_start_ticks']):
                    RECORD_STATE.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'status', 'record', 'stop'])
    parser.add_argument('--serial')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--wait-seconds', type=float, default=120)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.action == 'stop':
            stop(args.serial)
        elif args.action == 'status':
            status(args.serial)
        else:
            if args.action == 'record':
                prepare_record(args)
            state = start(args.serial)
            if args.action == 'record':
                return record(args, state)
        return 0
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f'原生采集测试：{exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
