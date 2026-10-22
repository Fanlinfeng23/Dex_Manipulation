"""Supervise one Vive dual-arm session and both Wuji Glove/Hand2 channels.

The arm process owns the *only* writer of ``hand.follow``.  This launcher
starts the hand worker with that gate closed, waits for fresh skeletons from
both gloves, then starts the interactive arm process.  A lost hand worker
stops the arm process; any exit closes the hand gate first.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
HAND_PYTHON = Path(os.environ.get('TIANJI_HAND_PYTHON', str(ROOT / '.venv/bin/python')))
HAND_WORKER = ROOT / 'wuji_joint_hand.py'
HAND_PROFILE = ROOT / 'wuji_hand_profile.py'
ARM_LAUNCHER = ROOT / 'tianji_vive.sh'
QUEST_STATE = Path(os.environ.get('TIANJI_QUEST_STATE_FILE',
                                  str(ROOT / 'apps/quest_native_arm_teleop/runtime/session.json')))
SIDES = ('left', 'right')
KINDS = ('hand', 'glove')


def _read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return result if isinstance(result, dict) else {}


def _run_checked(command: list[str | Path], *, timeout: float | None = None) -> None:
    env = os.environ.copy()
    env.setdefault('WUJI_TELEMETRY_DISABLE', '1')
    result = subprocess.run([str(part) for part in command], cwd=ROOT,
                            env=env, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f'Wuji 设备或标定步骤未完成（退出码 {result.returncode}）')


def _serial_options(args: argparse.Namespace) -> list[str]:
    result = []
    for side in SIDES:
        for kind in KINDS:
            value = getattr(args, f'{side}_{kind}_sn')
            if value:
                result.extend((f'--{side}-{kind}-sn', value))
    return result


def prepare_hands(args: argparse.Namespace, session: Path) -> tuple[dict[str, str], str]:
    """Reuse the proven Quest bilateral discovery and hand-model workflow."""
    if os.environ.get('WUJI_DEFAULT_MODEL') == '1':
        raise RuntimeError('双手联合遥操需要同一操作员的左右标定模型，不接受 Default 模型')
    device_file = session / 'hand-devices.json'
    print('只读确认左右 Wuji Glove 与 Hand2 四台设备。', flush=True)
    _run_checked([HAND_PYTHON, HAND_WORKER, '--check-only', '--device-output', device_file,
                  *_serial_options(args)], timeout=60)
    device_data = _read_json(device_file)
    serials: dict[str, str] = {}
    for side in SIDES:
        for kind in KINDS:
            value = device_data.get(side, {}).get(f'{kind}_sn')
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(f'{side} {kind} 序列号检查未通过')
            serials[f'--{side}-{kind}-sn'] = value
    for kind in KINDS:
        if serials[f'--left-{kind}-sn'] == serials[f'--right-{kind}-sn']:
            raise RuntimeError(f'左右 {kind} 序列号重复')

    user_file = session / 'hand-profile.json'
    model = args.model_user if args.model_user or args.calibrate_user else os.environ.get('WUJI_MODEL_USER')
    calibrate = (args.calibrate_user if args.model_user or args.calibrate_user
                 else os.environ.get('WUJI_CALIBRATE_USER'))
    if model and calibrate:
        raise RuntimeError('已有操作员与重新标定只能选择一项')
    command: list[str | Path] = [HAND_PYTHON, HAND_PROFILE, '--output', user_file,
                                 '--left-glove-sn', serials['--left-glove-sn'],
                                 '--right-glove-sn', serials['--right-glove-sn']]
    if model:
        command += ['--user', model]
    elif calibrate:
        command += ['--calibrate-user', calibrate]
    print('选择已完成左右手标定的操作员；如需新标定，在本终端模型菜单输入 c。', flush=True)
    _run_checked(command)
    selection = _read_json(user_file)
    user_id = selection.get('user_id')
    if not isinstance(user_id, str) or not user_id.strip() or selection.get('is_default'):
        raise RuntimeError('未选定已标定的具名 Wuji 操作员')
    models = selection.get('model_paths')
    if not isinstance(models, dict):
        raise RuntimeError('操作员模型缺少左右手路径')
    paths = []
    for side in SIDES:
        raw = models.get(side)
        path = Path(raw).resolve() if isinstance(raw, str) and raw else None
        if path is None or not path.is_file() or path.name != f'{side}_hand.urdf':
            raise RuntimeError(f'{side} 手模型不存在或侧别不符')
        paths.append(path)
    if (paths[0].parent != paths[1].parent or
            (paths[0].parent / 'joint_calibration.pending.json').exists()):
        raise RuntimeError('左右手不是同一操作员模型，或双手标定尚未完成')
    return serials, user_id


def _quest_live() -> bool:
    state = _read_json(QUEST_STATE)
    identity = state.get('launcher')
    if not isinstance(identity, dict):
        return False
    try:
        pid = int(identity['pid'])
        stat = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return stat[0] != 'Z' and int(stat[19]) == int(identity['start_ticks'])
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def _other_vive_live() -> bool:
    """Reject a separately started Vive arm controller on this host."""
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal() or int(path.name) == os.getpid():
            continue
        try:
            args = (path / 'cmdline').read_bytes().split(b'\0')
            if b'vive_tianji.teleop' in args and b'live' in args:
                return True
        except (OSError, PermissionError):
            continue
    return False


def _worker_ready(worker: subprocess.Popen, ready: Path) -> bool:
    if worker.poll() is not None:
        raise RuntimeError(f'左右 Wuji Hand2/Glove 通道退出（退出码 {worker.returncode}）')
    try:
        return ready.is_file() and ready.stat().st_size > 0
    except FileNotFoundError:
        return False


def _await_hand_ready(worker: subprocess.Popen, ready: Path, *, timeout_s: float = 15.) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _worker_ready(worker, ready):
            return
        time.sleep(.05)
    raise RuntimeError('15 秒内未同时收到左右 Wuji Glove 新鲜骨架帧；天机臂未启动')


def _stop_process(process: subprocess.Popen | None, *, label: str,
                  timeout_s: float = 30.) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'{label} 在 {timeout_s:g} 秒内未确认退出；请现场核对设备状态与急停') from exc


def _arm_command(args: argparse.Namespace, gate: Path, ready: Path) -> list[str]:
    command = [str(ARM_LAUNCHER), 'live', '--side', 'both', '--scale', str(args.scale),
               '--full-range', '--seconds', str(args.seconds),
               '--expected-baseline-m', str(args.expected_baseline_m),
               '--joint-gate', str(gate), '--joint-ready-file', str(ready)]
    return command


def run_joint(args: argparse.Namespace) -> int:
    if not sys.stdin.isatty():
        raise RuntimeError('实机联合遥操需在现场交互式终端执行')
    if not HAND_PYTHON.is_file() or not ARM_LAUNCHER.is_file():
        raise RuntimeError('Wuji SDK 虚拟环境或 Vive 启动脚本缺失')
    calibration = Path(os.environ.get('TIANJI_VIVE_CALIBRATION_FILE',
                                      ROOT / 'vive_tianji_calibration.json'))
    if not calibration.is_file():
        raise RuntimeError('双腕标定文件缺失；先运行 ./tianji_vive.sh calibrate --expected-baseline-m '
                           f'{args.expected_baseline_m:g}')
    if _quest_live() or _other_vive_live():
        raise RuntimeError('已有 Quest/Vive 实机遥操会话；请先退出原会话')
    log_root = ROOT / 'logs'
    log_root.mkdir(parents=True, exist_ok=True)
    with (log_root / '.vive-joint-launch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('已有 Vive 双腕联合启动器正在运行') from exc
        if _quest_live() or _other_vive_live():
            raise RuntimeError('已有 Quest/Vive 实机遥操会话；请先退出原会话')
        session = log_root / f'vive-joint-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}'
        session.mkdir()
        gate, ready = session / 'hand.follow', session / 'hand.ready'
        worker = arm = None
        errors: list[str] = []
        try:
            serials, user_id = prepare_hands(args, session)
            hand_command: list[str | Path] = [HAND_PYTHON, '-u', HAND_WORKER,
                                               '--gate', gate, '--gate-timeout', '0.5',
                                               '--ready-file', ready,
                                               '--model-user-id', user_id]
            for option, value in serials.items():
                hand_command.extend((option, value))
            env = os.environ.copy()
            env.setdefault('WUJI_TELEMETRY_DISABLE', '1')
            with (session / 'hand.log').open('ab') as hand_log:
                worker = subprocess.Popen([str(part) for part in hand_command],
                                          cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                          stdout=hand_log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            _await_hand_ready(worker, ready)
            print('左右 Wuji 骨架已就绪；灵巧手保持。现在进入 Vive 双臂实机会话。', flush=True)
            arm = subprocess.Popen(_arm_command(args, gate, ready), cwd=ROOT,
                                   start_new_session=True)
            while arm.poll() is None:
                if not _worker_ready(worker, ready):
                    raise RuntimeError(f'左右 Wuji 通道故障；详见 {session / "hand.log"}')
                time.sleep(.05)
            if not _worker_ready(worker, ready):
                raise RuntimeError(f'左右 Wuji 通道在双臂会话结束前失效；详见 {session / "hand.log"}')
            if arm.returncode:
                raise RuntimeError(f'Vive 双臂会话退出码 {arm.returncode}；详见 {session}')
            return 0
        except KeyboardInterrupt:
            errors.append('操作员请求退出')
            return 130
        except Exception as exc:
            errors.append(str(exc))
            raise
        finally:
            # Never leave a stale gate that can re-enable the hand worker.
            gate.unlink(missing_ok=True)
            shutdown_errors = []
            try:
                _stop_process(arm, label='Vive 双臂')
            except Exception as exc:
                shutdown_errors.append(str(exc))
            gate.unlink(missing_ok=True)
            try:
                _stop_process(worker, label='Wuji 双手')
            except Exception as exc:
                shutdown_errors.append(str(exc))
            ready.unlink(missing_ok=True)
            (session / 'supervisor.json').write_text(json.dumps({
                'finished_at': datetime.now().astimezone().isoformat(),
                'arm_exit_code': arm.poll() if arm else None,
                'hand_exit_code': worker.poll() if worker else None,
                'errors': errors + shutdown_errors,
            }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(f'Vive+Wuji 联合会话记录：{session}', flush=True)
            if shutdown_errors:
                raise RuntimeError('会话停机未确认：' + '；'.join(shutdown_errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='双 Vive Tracker + 双 Wuji Glove/Hand2 联合遥操')
    user = parser.add_mutually_exclusive_group()
    user.add_argument('--model-user', help='使用已完成左右手标定的操作员名称或 ID')
    user.add_argument('--calibrate-user', help='为操作员依次重新标定左右手')
    for side in SIDES:
        for kind in KINDS:
            parser.add_argument(f'--{side}-{kind}-sn', default='')
    parser.add_argument('--scale', type=float, default=1., help='双腕到双臂位移比例（0.2–1.0）；默认与 Wuji 官方米制 TF 一致，为 1:1')
    parser.add_argument('--seconds', type=float, default=0., help='自动结束时长；0 表示按 Q 退出')
    parser.add_argument('--expected-baseline-m', type=float, default=.240,
                        help='已实测两基站中心距；当前布局默认 0.240 m')
    args = parser.parse_args(argv)
    if not math.isfinite(args.scale) or not .2 <= args.scale <= 1.:
        parser.error('--scale 必须在 0.2–1.0 之间')
    if not math.isfinite(args.seconds) or not 0 <= args.seconds <= 3600.:
        parser.error('--seconds 必须在 0–3600 秒之间')
    if not math.isfinite(args.expected_baseline_m) or not .15 <= args.expected_baseline_m <= 5.:
        parser.error('--expected-baseline-m 必须在 0.15–5.0 m 之间')
    try:
        return run_joint(args)
    except Exception as exc:
        print(f'Vive+Wuji 联合遥操未启动或已停止：{exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
