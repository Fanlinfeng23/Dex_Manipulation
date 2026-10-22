"""Turn XRoboToolkit Quest JSON into protocol-v1 frames, without robot access.

The official Quest APK does not publish controller optical validity. `tracked`
below is only a conservative numeric gate for compatibility with the existing
arm engine; status always states whether validity was actually reported.
"""

from __future__ import annotations

import json
import hashlib
import math
import uuid


SIDES = ('left', 'right')
VALIDITY_KEYS = ('isPoseValid', 'poseValid', 'tracked', 'isTracked',
                 'positionTracked', 'orientationTracked')
COMPLETE_VALIDITY_KEYS = ('isPoseValid', 'poseValid', 'tracked', 'isTracked')
IDENTITY_POSE = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
REASON_ZH = {
    'waiting_for_native_frames': '等待 Quest 原生数据',
    'native_device_discovered': '已发现 Quest 数据，等待稳定',
    'native_device_changed': 'Quest 设备已变化，重新建立映射',
    'native_app_clock_reset': 'Quest 应用时钟回退，重新建立映射',
    'native_app_frame_gap': 'Quest 应用帧间隔过长，重新建立映射',
    'native_stream_gap': 'Quest 原生数据中断，重新建立映射',
    'native_stream_timeout': 'Quest 原生数据超时，已暂停',
    'native_input_not_controllers': '客户端未报告手柄或手部输入；请唤醒双手柄',
    'native_focus_lost': 'Quest 应用失焦，已暂停',
    'native_head_invalid': '头显位姿无效，已暂停',
    'native_explicit_tracking_invalid': '手柄明确报告追踪无效，已暂停',
    'native_controller_pose_invalid': '手柄姿态数值无效，已暂停',
    'native_controller_jump': '手柄姿态跳变，等待重新稳定',
    'native_controller_frozen': '手柄姿态长时间完全相同，疑似停滞',
    'waiting_for_distinct_controller_poses': '等待左右手柄各出现两种不同姿态',
    'waiting_for_stable_native_stream': '等待原生数据稳定约 0.6 秒',
    'numeric_gate_ready': '数值门禁已就绪；光学追踪有效性未报告',
    'invalid_native_json': '原生数据格式错误，已暂停',
    'missing_app_uptime': '原生应用时间缺失，已暂停',
    'sdk_server_disconnected': 'XRoboToolkit 服务已断开',
    'sdk_device_missing': 'Quest 原生设备已离线',
    'sdk_callback_error': '原生 SDK 回调异常，已暂停',
    'bridge_websocket_connected': '输入桥已连接，等待新数据',
    'bridge_websocket_disconnected': '输入桥已断开，已暂停',
    'bridge_stopped': '输入桥已停止',
}


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _pose(value):
    if isinstance(value, str):
        parts = value.split(',')
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        return None
    if len(parts) != 7:
        return None
    numbers = [_number(part) for part in parts]
    if any(part is None for part in numbers):
        return None
    position, quaternion = numbers[:3], numbers[3:]
    norm = math.sqrt(sum(q * q for q in quaternion))
    if max(map(abs, position)) > 30 or not 0.95 <= norm <= 1.05:
        return None
    return position, [q / norm for q in quaternion]


def _rotation_delta(a, b):
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return 2 * math.acos(min(1.0, max(-1.0, dot)))


def _same_pose(a, b):
    if a[0] != b[0]:
        return False
    return a[1] == b[1] or all(x == -y for x, y in zip(a[1], b[1]))


def _unwrap(payload):
    root = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(root, dict):
        raise ValueError('native payload root is not an object')
    if 'functionName' in root and root['functionName'] != 'Tracking':
        return None
    value = root.get('value', root)
    for _ in range(3):
        if not isinstance(value, str):
            break
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError('native tracking value is not an object')
    return value


class NativeAdapter:
    STABLE_S = 0.6
    FROZEN_S = 0.35
    MAX_POSITION_STEP_M = 0.10
    MAX_ROTATION_STEP_RAD = 0.5
    STREAM_GAP_S = 0.20
    REORDER_TOLERANCE_MS = 50.0

    def __init__(self, *, bridge_uuid=None):
        self.bridge_uuid = bridge_uuid or uuid.uuid4().hex[:12]
        self.device_id = None
        self.epoch = 0
        self.session = None
        self.seq = 0
        self.last_app_ms = None
        self.last_rx = None  # Only distinct, monotonic application frames refresh this.
        self.last_packet = None
        self.stable_since = None
        self.last_pose = {side: None for side in SIDES}
        self.last_pose_change = {side: None for side in SIDES}
        self.changed = {side: False for side in SIDES}
        self.first_valid_pose = {side: None for side in SIDES}
        self.eligible = False
        self.reason = 'waiting_for_native_frames'
        self.tracking_validity = 'unreported'
        self.reported_sides = []
        self.raw_trigger = {side: None for side in SIDES}
        self.input_value = None
        self.head_status = None
        self.focus = None
        self.duplicate_count = 0
        self.out_of_order_count = 0
        self.clock_reset_count = 0

    @property
    def status(self):
        reason_key = self.reason.split(':', 1)[0]
        return {
            'eligible': self.eligible,
            'tracking_validity': self.tracking_validity,
            'reported_sides': list(self.reported_sides),
            'raw_trigger': dict(self.raw_trigger),
            'reason': self.reason,
            'reason_zh': REASON_ZH.get(reason_key, self.reason),
            'last_rx': self.last_rx,
            'last_packet': self.last_packet,
            'device_id': self.device_id,
            'session': self.session,
            'epoch': self.epoch,
            'seq': self.seq,
            'input': self.input_value,
            'input_note': ('手部模型可见；Input=2 不代表手柄断开，按双侧 Controller 数据检查'
                           if self.input_value == 2 else None),
            'head_status': self.head_status,
            'focus': self.focus,
            'duplicate_app_frames': self.duplicate_count,
            'out_of_order_app_frames': self.out_of_order_count,
            'clock_resets': self.clock_reset_count,
            'optical_6dof_proven': False,
            'validity_note': 'Quest APK controller validity is unreported unless '
                             'explicit per-side flags appear; numeric gate can '
                             'misclassify a stationary controller as frozen.',
        }

    def reset_epoch(self, reason, *, device_id=None):
        if device_id is not None:
            self.device_id = str(device_id)
        self.epoch += 1
        device_tag = hashlib.sha1((self.device_id or '').encode()).hexdigest()[:8]
        self.session = f'native-{self.bridge_uuid}-{device_tag}-{self.epoch}'
        self.seq = 0
        self.last_app_ms = None
        self.last_rx = None
        self.stable_since = None
        self.last_pose = {side: None for side in SIDES}
        self.last_pose_change = {side: None for side in SIDES}
        self.changed = {side: False for side in SIDES}
        self.first_valid_pose = {side: None for side in SIDES}
        self.eligible = False
        self.reason = reason

    def invalidate(self, reason, *, rotate=False):
        if rotate:
            self.reset_epoch(reason)
        else:
            self.eligible = False
            self.stable_since = None
            self.changed = {side: False for side in SIDES}
            self.first_valid_pose = {side: None for side in SIDES}
            self.reason = reason

    def check_stale(self, now_s):
        if self.last_rx is not None and now_s - self.last_rx > self.STREAM_GAP_S:
            self.invalidate('native_stream_timeout', rotate=True)
            return True
        return False

    def consume(self, payload, device_id, received_s):
        """Return a v1 frame or None for duplicate/out-of-order source frames."""
        if not math.isfinite(received_s):
            raise ValueError('received_s must be finite')
        self.last_packet = float(received_s)
        device_id = str(device_id)
        if self.device_id != device_id:
            self.reset_epoch('native_device_changed' if self.device_id is not None
                             else 'native_device_discovered', device_id=device_id)
        try:
            value = _unwrap(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            self.invalidate('invalid_native_json')
            return None
        if value is None:
            return None
        app_us = _number(value.get('predictTime'))
        if app_us is None or app_us < 0:
            self.invalidate('missing_app_uptime')
            return None
        app_ms = app_us / 1000.0  # Unity Time.unscaledTimeAsDouble * 1e6.
        if self.last_app_ms is not None and app_ms <= self.last_app_ms:
            if app_ms == self.last_app_ms:
                self.duplicate_count += 1
                return None
            if self.last_app_ms - app_ms <= self.REORDER_TOLERANCE_MS:
                self.out_of_order_count += 1
                return None
            self.clock_reset_count += 1
            self.reset_epoch('native_app_clock_reset')
        elif (self.last_app_ms is not None and
              app_ms - self.last_app_ms > self.STREAM_GAP_S * 1000):
            self.reset_epoch('native_app_frame_gap')
        if self.last_rx is not None and received_s - self.last_rx > self.STREAM_GAP_S:
            self.reset_epoch('native_stream_gap')
        self.last_app_ms = app_ms
        self.last_rx = float(received_s)
        self.seq += 1

        self.input_value = value.get('Input')
        app_state = value.get('appState')
        self.focus = app_state.get('focus') if isinstance(app_state, dict) else None
        head_value = value.get('Head')
        self.head_status = head_value.get('status') if isinstance(head_value, dict) else None
        head_pose = _pose(head_value.get('pose')) if isinstance(head_value, dict) else None
        head_ok = (type(self.head_status) is int and self.head_status == 3 and
                   head_pose is not None)

        raw_controllers = value.get('Controller')
        raw_controllers = raw_controllers if isinstance(raw_controllers, dict) else {}
        controllers = {}
        current_poses = {}
        bad_pose = []
        explicit_invalid = []
        jumps = []
        frozen = []
        complete_sides = []
        for side in SIDES:
            source = raw_controllers.get(side)
            source = source if isinstance(source, dict) else {}
            parsed = _pose(source.get('pose'))
            flags = {key: source[key] for key in VALIDITY_KEYS if key in source}
            if flags:
                if (any(key in flags for key in COMPLETE_VALIDITY_KEYS) or
                        ('positionTracked' in flags and 'orientationTracked' in flags)):
                    complete_sides.append(side)
                if any(type(flag) is not bool or not flag for flag in flags.values()):
                    explicit_invalid.append(side)
            if parsed is None:
                bad_pose.append(side)
                parsed = self.last_pose[side] or IDENTITY_POSE
            else:
                old = self.last_pose[side]
                if old is None:
                    self.last_pose_change[side] = received_s
                elif _same_pose(parsed, old):
                    if (self.last_pose_change[side] is not None and
                            received_s - self.last_pose_change[side] >= self.FROZEN_S):
                        frozen.append(side)
                else:
                    self.last_pose_change[side] = received_s
                    if (math.dist(parsed[0], old[0]) > self.MAX_POSITION_STEP_M or
                            _rotation_delta(parsed[1], old[1]) > self.MAX_ROTATION_STEP_RAD):
                        jumps.append(side)
                self.last_pose[side] = parsed
            current_poses[side] = parsed
            trigger = _number(source.get('trigger'))
            self.raw_trigger[side] = (trigger if trigger is not None and
                                      0 <= trigger <= 1 else None)
            controllers[side] = {
                'position': list(parsed[0]), 'quaternion': list(parsed[1]),
                'tracked': False,
                'grip': 0.0,
            }
        self.reported_sides = complete_sides
        self.tracking_validity = ('reported' if set(complete_sides) == set(SIDES)
                                  else 'unreported')
        problem = None
        # Quest GetActiveInputDevice() prioritizes HandVisual.IsVisible over
        # controller.IsConnected. Input=2 is not exclusive hand-only mode.
        # Both 1 and 2 still require the same bilateral numeric/temporal gates;
        # neither value proves controller optical validity. Never substitute
        # Hand joint poses for the Controller fields.
        if type(self.input_value) is not int or self.input_value not in (1, 2):
            problem = 'native_input_not_controllers'
        elif self.focus is not True:
            problem = 'native_focus_lost'
        elif not head_ok:
            problem = 'native_head_invalid'
        elif explicit_invalid:
            problem = 'native_explicit_tracking_invalid:' + ','.join(explicit_invalid)
        elif bad_pose:
            problem = 'native_controller_pose_invalid:' + ','.join(bad_pose)
        elif jumps:
            problem = 'native_controller_jump:' + ','.join(jumps)
        elif frozen:
            problem = 'native_controller_frozen:' + ','.join(frozen)

        if problem:
            self.invalidate(problem)
        else:
            for side in SIDES:
                if self.first_valid_pose[side] is None:
                    self.first_valid_pose[side] = current_poses[side]
                elif not _same_pose(self.first_valid_pose[side], current_poses[side]):
                    self.changed[side] = True
            if self.stable_since is None:
                self.stable_since = received_s
            if not all(self.changed.values()):
                self.eligible = False
                self.reason = 'waiting_for_distinct_controller_poses'
            elif received_s - self.stable_since < self.STABLE_S:
                self.eligible = False
                self.reason = 'waiting_for_stable_native_stream'
            else:
                self.eligible = True
                self.reason = 'numeric_gate_ready'
        for side in SIDES:
            controllers[side]['tracked'] = self.eligible
        return {
            'version': 1, 'type': 'frame', 'session': self.session,
            'reference': 0, 'seq': self.seq, 'captured_ms': app_ms,
            'focused': self.focus is True,
            'controllers': controllers,
            'head': {'quaternion': head_pose[1] if head_pose else [0.0, 0.0, 0.0, 1.0],
                     'tracked': bool(head_ok)},
            'safety': {'released': [], 'tracking_lost': not self.eligible,
                       'focus_lost': self.focus is not True},
            'buttons': {'calibrate': False, 'reset': False},
        }
