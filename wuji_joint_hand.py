#!/usr/bin/env python3
"""Bilateral Wuji Glove -> Hand2 runner with an external arm-follow gate.

The retargeting math and command format are the official SDK path.  The gate
only coordinates start/pause with the Quest/Tianji keyboard controller; it is
not a replacement for the hand's own emergency stop.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wuji_sdk import (DeviceType, HandModel, Handedness, JointCommand,
                      RetargetSession, SdkManager)
import wuji_hand_profile as hand_profile

FPS = 120
GLOVE_FRAME_TIMEOUT_S = 0.6
RESUME_MAX_JOINT_STEP_RAD = 0.025
RESUME_MAX_DURATION_S = 2.0
MAX_DRAIN_FRAMES = 256
DISCOVERY_SCAN_ATTEMPTS = 3
DISCOVERY_RETRY_DELAY_S = 0.35
SIDES = ('left', 'right')
KINDS = ('hand', 'glove')
KIND_LABELS = {'hand': 'Wuji Hand2', 'glove': 'Wuji Glove'}
SIDE_LABELS = {'left': '左', 'right': '右'}


def glove_frame_stale(last_frame_time, now):
    return last_frame_time is None or now - last_frame_time > GLOVE_FRAME_TIMEOUT_S


def limited_resume_target(target, last_sent, max_step=RESUME_MAX_JOINT_STEP_RAD):
    """Limit one reconnect command relative to the last command actually sent."""
    target = np.asarray(target, dtype=np.float64)
    last_sent = np.asarray(last_sent, dtype=np.float64)
    if (target.shape != (20,) or last_sent.shape != (20,) or
            not np.isfinite(target).all() or not np.isfinite(last_sent).all()):
        raise RuntimeError('Wuji 灵巧手重接目标无效')
    sent = last_sent + np.clip(target - last_sent, -max_step, max_step)
    caught_up = np.all(np.abs(target - sent) <= 1e-9)
    return sent, bool(caught_up)


def latest_keypoints(subscription):
    # SDK Subscription.recv() is non-blocking and returns None immediately when
    # drained (wuji_sdk/__init__.pyi); this keeps the stream-age watchdog live.
    latest = None
    drained = 0
    while True:
        frame = subscription.recv()
        if frame is None:
            break
        drained += 1
        if drained > MAX_DRAIN_FRAMES:
            raise RuntimeError('Wuji Glove 骨架流积压过多，无法确认当前手势')
        latest = frame
    if latest is None:
        return None
    keypoints = np.array([joint.pose.position for joint in latest.joints], dtype=np.float32)
    if keypoints.shape != (21, 3) or not np.isfinite(keypoints).all():
        raise RuntimeError(f'Wuji Glove 骨架帧无效：shape={keypoints.shape}')
    identity = (int(latest.header.seq), int(latest.header.timestamp_us))
    return keypoints, identity


@dataclass
class DeviceMatch:
    descriptor: object
    handle: object
    side: str


@dataclass
class SideChannel:
    side: str
    hand: object
    publisher: object
    subscription: object
    session: object
    last_frame_time: float | None = None
    last_frame_identity: tuple[int, int] | None = None
    current_keypoints: np.ndarray | None = None
    last_sent_qpos: np.ndarray | None = None
    resume_ramp_started: float | None = None


def normalized_side(raw, *, sn, kind):
    value = str(raw).strip().lower().rsplit('.', 1)[-1]
    if value not in SIDES:
        raise RuntimeError(f'{KIND_LABELS[kind]} SN={sn} 侧别未知：{raw!r}')
    return value


def requested_serials(args):
    if args.hand_sn and args.right_hand_sn and args.hand_sn != args.right_hand_sn:
        raise ValueError('--hand-sn 与 --right-hand-sn 指定了不同的右手序列号')
    if args.glove_sn and args.right_glove_sn and args.glove_sn != args.right_glove_sn:
        raise ValueError('--glove-sn 与 --right-glove-sn 指定了不同的右手套序列号')
    requested = {
        'left': {'hand': args.left_hand_sn, 'glove': args.left_glove_sn},
        'right': {'hand': args.right_hand_sn or args.hand_sn,
                  'glove': args.right_glove_sn or args.glove_sn},
    }
    for kind in KINDS:
        left, right = requested['left'][kind], requested['right'][kind]
        if left and right and left == right:
            raise ValueError(f'左右 {KIND_LABELS[kind]} 不能使用同一序列号：{left}')
    return requested


def resolve_device_pairs(identified, requested):
    """Reject a wrong side or ambiguity; return all still-missing requirements."""
    pairs = {side: {} for side in SIDES}
    missing = []
    for kind in KINDS:
        for side in SIDES:
            wanted_sn = requested[side][kind]
            matches = [item for item in identified[kind].values()
                       if item.side == side and (not wanted_sn or item.descriptor.sn == wanted_sn)]
            if not matches:
                if wanted_sn:
                    actual = identified[kind].get(wanted_sn)
                    if actual is not None:
                        raise RuntimeError(
                            f'指定的 {SIDE_LABELS[side]}侧 {KIND_LABELS[kind]} '
                            f'SN={wanted_sn} 实际为 {actual.side} 侧')
                    missing.append(f'未找到指定的{SIDE_LABELS[side]}侧 {KIND_LABELS[kind]} SN={wanted_sn}')
                else:
                    missing.append(f'未找到{SIDE_LABELS[side]}侧 {KIND_LABELS[kind]}')
                continue
            if len(matches) > 1:
                serials = ', '.join(item.descriptor.sn for item in matches)
                raise RuntimeError(
                    f'检测到多只{SIDE_LABELS[side]}侧 {KIND_LABELS[kind]}：{serials}；'
                    f'请指定 --{side}-{kind}-sn')
            pairs[side][kind] = matches[0]
    return pairs, missing


def discovered_summary(identified):
    found = [f"{SIDE_LABELS[item.side]}侧 {KIND_LABELS[kind]}={item.descriptor.sn}"
             for kind in KINDS for item in identified[kind].values()]
    return '、'.join(found) if found else '无'


def discover_devices(manager, requested, *, scan_attempts=DISCOVERY_SCAN_ATTEMPTS,
                     retry_delay=DISCOVERY_RETRY_DELAY_S, sleep=time.sleep):
    """Bounded read-only discovery, retaining verified handles across scans."""
    if scan_attempts < 1:
        raise ValueError('设备扫描次数必须至少为 1')
    scanned = {kind: {} for kind in KINDS}
    handles = {}
    identified = {kind: {} for kind in KINDS}
    connection_errors = {}
    scan_errors = []
    for attempt in range(1, scan_attempts + 1):
        verified_this_round = set()
        print(f'Wuji 设备扫描 {attempt}/{scan_attempts}：', flush=True)
        try:
            devices = manager.scan()
        except Exception as exc:
            scan_errors.append(str(exc))
            print(f'  本次扫描失败：{exc}', flush=True)
            devices = []
        this_scan = {kind: set() for kind in KINDS}
        for device in devices:
            print(f'  SN={device.sn}, Type={device.device_type}, Address={device.address}', flush=True)
            kind = ('hand' if device.device_type == DeviceType.WujiHand2 else
                    'glove' if device.device_type == DeviceType.WujiGlove else None)
            if kind is None:
                continue
            if not device.sn:
                raise RuntimeError(f'{KIND_LABELS[kind]} 扫描到空序列号，无法确认设备身份')
            if device.sn in this_scan[kind]:
                raise RuntimeError(f'{KIND_LABELS[kind]} 扫描到重复序列号 {device.sn}')
            this_scan[kind].add(device.sn)
            other_kind = 'glove' if kind == 'hand' else 'hand'
            if device.sn in scanned[other_kind]:
                raise RuntimeError(f'SN={device.sn} 在不同扫描中被识别为两种设备类型')
            scanned[kind].setdefault(device.sn, device)
        for kind in KINDS:
            pinned = {requested[side][kind] for side in SIDES if requested[side][kind]}
            for sn, device in scanned[kind].items():
                if all(requested[side][kind] for side in SIDES) and sn not in pinned:
                    continue
                if sn in identified[kind]:
                    continue
                key = kind, sn
                try:
                    handle = handles.get(key)
                    if handle is None:
                        handle = manager.connect(sn=sn, device_name=sn)
                        handles[key] = handle
                    raw_side = (handle.handedness().get() if kind == 'hand' else
                                handle.hand_side().get())
                except Exception as exc:
                    connection_errors[key] = str(exc)
                    continue
                side = normalized_side(raw_side, sn=sn, kind=kind)
                identified[kind][sn] = DeviceMatch(device, handle, side)
                verified_this_round.add(key)
                connection_errors.pop(key, None)
        pairs, missing = resolve_device_pairs(identified, requested)
        if not missing:
            # A device found in an earlier scan may have disappeared while a
            # later device was being discovered. Recheck cached handles before
            # treating the union of scan results as a complete live set.
            for side in SIDES:
                for kind in KINDS:
                    item = pairs[side][kind]
                    sn = item.descriptor.sn
                    key = kind, sn
                    if key in verified_this_round:
                        continue
                    try:
                        raw_side = (item.handle.handedness().get() if kind == 'hand' else
                                    item.handle.hand_side().get())
                    except Exception as exc:
                        identified[kind].pop(sn, None)
                        connection_errors[key] = str(exc)
                        continue
                    current_side = normalized_side(raw_side, sn=sn, kind=kind)
                    if current_side != item.side:
                        raise RuntimeError(
                            f'{KIND_LABELS[kind]} SN={sn} 侧别变化：'
                            f'先前为 {item.side} 侧，当前为 {current_side} 侧')
                    verified_this_round.add(key)
                    connection_errors.pop(key, None)
            pairs, missing = resolve_device_pairs(identified, requested)
            if not missing:
                return pairs
        if attempt < scan_attempts:
            print(f'  尚未齐备：{"、".join(missing)}；{retry_delay:g} 秒后重试。', flush=True)
            sleep(retry_delay)
    details = [f'扫描 {scan_attempts} 次后设备未齐：{"、".join(missing)}',
               f'已识别：{discovered_summary(identified)}']
    if connection_errors:
        details.append('连接或侧别读取失败：' + '、'.join(
            f'{KIND_LABELS[kind]} SN={sn}（{error}）'
            for (kind, sn), error in connection_errors.items()))
    if scan_errors:
        details.append(f'最近扫描异常：{scan_errors[-1]}')
    details.append('请检查缺失设备的供电、网线或 USB、IP 同网段后重试')
    raise RuntimeError('；'.join(details))


def device_serials(pairs):
    return {side: {'hand_sn': pairs[side]['hand'].descriptor.sn,
                   'glove_sn': pairs[side]['glove'].descriptor.sn}
            for side in SIDES}


def write_device_output(path, pairs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as temp:
        json.dump(device_serials(pairs), temp, ensure_ascii=False)
        temp.write('\n')
        temp_path = Path(temp.name)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def configure_glove_models(pairs, models):
    """Bind each connected glove to the selected side's calibrated URDF."""
    for side in SIDES:
        expected = Path(models[side]).expanduser().resolve()
        glove = pairs[side]['glove'].handle
        try:
            resource = glove.hand_model_path()
            resource.set(str(expected))
            actual_raw = resource.get()
        except Exception as exc:
            raise RuntimeError(
                f'{SIDE_LABELS[side]}侧 Wuji Glove 手型模型设置或回读失败：{exc}') from exc
        actual = Path(actual_raw).expanduser().resolve() if actual_raw else None
        if actual != expected:
            raise RuntimeError(
                f'{SIDE_LABELS[side]}侧 Wuji Glove 手型模型不匹配：'
                f'期望 {expected}，实际 {actual_raw!r}')
        print(f'{SIDE_LABELS[side]}侧 Wuji Glove 已加载手型模型：{actual}', flush=True)


def poll_channel(channel, now):
    channel.current_keypoints = None
    sample = latest_keypoints(channel.subscription)
    if sample is not None and sample[1] != channel.last_frame_identity:
        channel.current_keypoints, channel.last_frame_identity = sample
        channel.last_frame_time = now


def streams_ready(channels, now):
    return all(not glove_frame_stale(channels[side].last_frame_time, now) for side in SIDES)


def gate_heartbeat_active(gate, timeout_s, now_wall):
    """Treat a stale keyboard gate as closed; zero retains the legacy file gate."""
    if not gate.exists():
        return False
    if timeout_s == 0:
        return True
    try:
        age = now_wall - gate.stat().st_mtime
    except OSError:
        return False
    # A small wall-clock skew is harmless, but a far-future mtime must not
    # keep an abandoned gate open after the keyboard or launcher exits.
    return math.isfinite(age) and -0.1 <= age <= timeout_s


def require_fresh_streams(channels, now):
    for side in SIDES:
        if glove_frame_stale(channels[side].last_frame_time, now):
            raise RuntimeError(
                f'{SIDE_LABELS[side]}侧 Wuji Glove 骨架流超过 {GLOVE_FRAME_TIMEOUT_S} 秒无新帧；联合遥操将停止')


def prepare_channel_target(channel, now):
    kp = channel.current_keypoints
    if kp is None:
        return None
    qpos = np.asarray(channel.session.step(kp), dtype=np.float64)
    if qpos.shape != (20,) or not np.isfinite(qpos).all():
        raise RuntimeError(f'{SIDE_LABELS[channel.side]}侧 Wuji 重定向输出不是有效的 20 维关节目标')
    caught_up = False
    if channel.resume_ramp_started is not None:
        qpos, caught_up = limited_resume_target(qpos, channel.last_sent_qpos)
        if not caught_up and now - channel.resume_ramp_started > RESUME_MAX_DURATION_S:
            raise RuntimeError(f'{SIDE_LABELS[channel.side]}侧 Wuji 灵巧手重接超过 2 秒仍未追上手势')
    return qpos, caught_up


def publish_channel_target(channel, prepared):
    if prepared is None:
        return
    qpos, caught_up = prepared
    channel.publisher.send([JointCommand(float(position), 0.0, 0.0) for position in qpos])
    channel.last_sent_qpos = qpos.copy()
    if caught_up:
        channel.resume_ramp_started = None
        print(f'{SIDE_LABELS[channel.side]}侧 Wuji 灵巧手：重接平滑段完成。', flush=True)


def send_channel_target(channel, now):
    publish_channel_target(channel, prepare_channel_target(channel, now))


def run_joint_loop(channels, gate, ready_file, enabled_hands, *, gate_timeout=0.0,
                   clock=time.monotonic, gate_clock=time.time, sleep=time.sleep):
    budget = 1.0 / FPS
    ready_announced = False
    enabled = False
    gate_wait_start = None
    while True:
        started = clock()
        for side in SIDES:
            poll_channel(channels[side], started)
        if not ready_announced and streams_ready(channels, started):
            if ready_file is not None:
                ready_file.write_text('ready\n', encoding='utf-8')
            ready_announced = True
            print('左右 Wuji Glove 均已收到有效骨架帧。', flush=True)
        if ready_announced:
            require_fresh_streams(channels, started)

        gate_open = gate_heartbeat_active(gate, gate_timeout, gate_clock())
        if gate_open and not enabled:
            if not streams_ready(channels, started):
                if gate_wait_start is None:
                    gate_wait_start = started
                    print('等待左右 Wuji Glove 新鲜骨架帧，暂不使能灵巧手。', flush=True)
                elif started - gate_wait_start > GLOVE_FRAME_TIMEOUT_S:
                    raise RuntimeError('左右 Wuji Glove 骨架流未就绪；联合遥操将停止')
                sleep(.02)
                continue
            gate_wait_start = None
            for side in SIDES:
                channel = channels[side]
                if all(hand is not channel.hand for hand in enabled_hands):
                    enabled_hands.append(channel.hand)
                channel.hand.enable()
                channel.resume_ramp_started = started if channel.last_sent_qpos is not None else None
            enabled = True
            print('左右 Wuji 灵巧手：跟随已开始', flush=True)
        elif not gate_open and enabled:
            enabled = False
            gate_wait_start = None
            for side in SIDES:
                channels[side].resume_ramp_started = None
            print('左右 Wuji 灵巧手：已暂停并保持', flush=True)

        if enabled:
            # Validate both sides before publishing either command. A bad
            # retargeting result must not advance just one hand this frame.
            targets = {side: prepare_channel_target(channels[side], started)
                       for side in SIDES}
            for side in SIDES:
                publish_channel_target(channels[side], targets[side])
        elapsed = clock() - started
        if elapsed < budget:
            sleep(budget - elapsed)


def _interrupt_for_cleanup(_signum, _frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description='双手 Wuji Glove/Hand2 联合遥操通道')
    parser.add_argument('--gate', default='', help='keyboard follow gate file')
    parser.add_argument('--gate-timeout', type=float, default=0.0,
                        help='gate 最近一次心跳允许的秒数；0 保持旧版文件门控')
    parser.add_argument('--check-only', action='store_true',
                        help='只读连接并确认左右两只 Hand2 和两只 Glove 的侧别')
    parser.add_argument('--left-hand-sn', default='')
    parser.add_argument('--right-hand-sn', default='')
    parser.add_argument('--left-glove-sn', default='')
    parser.add_argument('--right-glove-sn', default='')
    parser.add_argument('--hand-sn', default='', help='--right-hand-sn 的兼容别名')
    parser.add_argument('--glove-sn', default='', help='--right-glove-sn 的兼容别名')
    parser.add_argument('--device-output', default=None, type=Path,
                        help='将确认的左右设备序列号写入 JSON 文件')
    parser.add_argument('--model-user-id', default=None,
                        help='本次联合启动选定的具名 SDK 用户 ID')
    parser.add_argument('--ready-file', default='',
                        help='左右手套均有新鲜骨架帧后创建的就绪文件')
    args = parser.parse_args()
    if not args.check_only and not args.gate:
        parser.error('--gate 是联合运行必需参数')
    if not math.isfinite(args.gate_timeout) or args.gate_timeout < 0:
        parser.error('--gate-timeout 必须是有限的非负秒数')
    try:
        requested = requested_serials(args)
    except ValueError as exc:
        parser.error(str(exc))
    manager = SdkManager.instance()
    previous_sigint = signal.signal(signal.SIGINT, _interrupt_for_cleanup)
    previous_sigterm = signal.signal(signal.SIGTERM, _interrupt_for_cleanup)
    enabled_hands = []
    publishers = []
    ready_file = Path(args.ready_file) if args.ready_file and not args.check_only else None
    previous_user = None
    try:
        if not args.check_only:
            if args.model_user_id is None:
                selected_user = manager.current_user()
            else:
                previous_user = manager.current_user()
                selected_user = hand_profile.select_user(
                    manager, hand_profile.find_user(manager, args.model_user_id))
            models = hand_profile.checked_model_paths(selected_user)
            print(f"Wuji 模型用户：{selected_user['display_name']}，"
                  f"左模型：{models['left']}，右模型：{models['right']}", flush=True)

        # Side identity is read from each connected device. Hand motors are not
        # enabled and command publishers are not opened during --check-only.
        pairs = discover_devices(manager, requested)
        serials = device_serials(pairs)
        for side in SIDES:
            print(f"{SIDE_LABELS[side]}侧设备：Hand2={serials[side]['hand_sn']}，"
                  f"Glove={serials[side]['glove_sn']}", flush=True)
        if args.device_output:
            write_device_output(args.device_output, pairs)
        if args.check_only:
            print('左右 Wuji Hand2/Glove 只读侧别检查通过。', flush=True)
            return 0
        # The SDK can retain a custom online-IK path from an earlier session.
        # Confirm each glove uses the selected side's URDF before any Hand2
        # parameter writes, command publishers, or motor enable.
        configure_glove_models(pairs, models)
        channels = {}
        for side in SIDES:
            hand = pairs[side]['hand'].handle
            glove = pairs[side]['glove'].handle
            hand.effort_limit().set(1.5)
            hand.mit_params().set((5.0, 0.1))
            publisher = hand.joint_command().publish()
            publishers.append(publisher)
            subscription = glove.hand_skeleton().subscribe()
            handedness = Handedness.Left if side == 'left' else Handedness.Right
            session = RetargetSession.for_hand(HandModel.WujiHand2, side=handedness)
            channels[side] = SideChannel(side, hand, publisher, subscription, session)
        print('左右 Wuji 手部联合通道已连接；等待 Quest 追踪和联合开始。', flush=True)
        run_joint_loop(channels, Path(args.gate), ready_file, enabled_hands,
                       gate_timeout=args.gate_timeout)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f'Wuji 灵巧手通道故障：{exc}', file=sys.stderr, flush=True)
        return 1
    finally:
        # Gate pauses hold the last target; disabling happens only on exit.
        if ready_file is not None:
            with contextlib.suppress(Exception):
                ready_file.unlink(missing_ok=True)
        for hand in reversed(enabled_hands):
            with contextlib.suppress(Exception):
                hand.disable()
        for publisher in reversed(publishers):
            with contextlib.suppress(Exception):
                publisher.close()
        with contextlib.suppress(Exception):
            manager.disconnect_all()
        if previous_user is not None:
            try:
                if previous_user.get('is_default'):
                    manager.switch_to_default_user()
                else:
                    manager.switch_user(previous_user['user_id'])
            except Exception as exc:
                print(f'恢复先前 Wuji SDK 用户失败：{exc}', file=sys.stderr, flush=True)
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    raise SystemExit(main())
