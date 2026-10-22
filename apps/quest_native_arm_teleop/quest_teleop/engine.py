from __future__ import annotations

import math
import time
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation

from .math3d import (XR_TO_WORLD, heading_basis, pose_matrix, relative_target, rotation_angle,
                     serialized_pose, smooth_target)
from .kinematics import IKError
from .one_euro import OneEuroPose
from .protocol import Frame
from .wrist import WristEnvelope

SIDES = ('left', 'right')


class ControllerDiscontinuity(IKError):
    """A tracked controller pose changed too far to command safely."""


class TeleopEngine:
    """Single-owner virtual model and optional robot output."""
    KEYBOARD_TIMEOUT_S = 1.
    TRACKING_RECOVERY_STABLE_S = .6
    TRACKING_RECOVERY_MIN_FRAMES = 8
    TRACKING_RECOVERY_MAX_FRAME_GAP_S = .15

    def __init__(self, config, kinematics, clock=time.perf_counter):
        self.config, self.kine, self.clock = config, kinematics, clock
        self.control_sides = tuple(config.get('hardware', {}).get('control_sides', SIDES))
        self.source = config.get('initial_source', 'simulation')
        self.scale = config['position_scale']
        self.enabled = False
        self.keyboard_mode = False
        self.keyboard_follow = False
        self.ui_follow = None  # None uses physical triggers; bool overrides both.
        self.trigger_release_required = {s: False for s in SIDES}
        self.trigger_release_after_seq = -1
        self.keyboard_deadline = None
        self.reason = '等待输入；按 A 校准相对位姿，然后按住扳机键'
        self.calibration_revision = 0
        self.reset_revision = 0
        self.frame: Frame | None = None
        self.clock_offset = None
        self.extra_delay_s = 0.
        self.consume_delay_s = 0.
        self.arrivals = deque(maxlen=120)
        self.last_tick = self.clock()
        self.basis = Rotation.from_euler('z', config['operator_yaw_deg'], degrees=True).as_matrix() @ XR_TO_WORLD
        self.calibrated_heading_deg = None
        self.offsets = {s: pose_matrix(config['arms'][s]['controller_to_wrist_m'],
                                       config['arms'][s]['controller_to_wrist_xyzw']) for s in SIDES}
        self.q = {}
        self.actual = {}
        self.target = {}
        self.filtered = {}
        self.anchors = {}
        self.last_controller = {}
        filter_settings = config.get('input_filter', {'type': 'none'})
        self.pose_filters = ({s: OneEuroPose(filter_settings) for s in SIDES}
                             if filter_settings.get('type') == 'one_euro' else {})
        self.active = {s: False for s in SIDES}
        self.ik_feedback = {s: {'limited': False, 'iterations': 0} for s in SIDES}
        self.reset_plan = None
        self.reset_progress = 1.
        self.hardware_output = None
        self.hardware_fault = None
        self.hardware_feedback = None
        self.measured_q = {}
        self.tracking_error_since = None
        self.tracking_loss_since = None
        self.tracking_loss_active = False
        self.tracking_recovery_pending = False
        self.tracking_recovery_valid_since = None
        self.tracking_recovery_valid_frames = 0
        self.tracking_recovery_last_valid_s = None
        self.tracking_recovery_last_controller = None
        self.tracking_recovery_resume = False
        self._set_joints({s: np.array(config['arms'][s]['initial_joints_deg'], dtype=float) for s in SIDES})
        self.wrist = (WristEnvelope(self.q, config['hardware'])
                      if config.get('hardware', {}).get('motion_scope') == 'wrist' else None)
        self.reason = '初始姿态已就绪；松开扳机后按 A 校准'

    def _set_joints(self, joints):
        poses = {s: self.kine.fk(s, joints[s]) for s in SIDES}
        for side in SIDES:
            self.q[side] = joints[side].copy()
            self.actual[side] = poses[side]
            self.target[side] = poses[side].copy()
            self.filtered[side] = poses[side].copy()

    def _reset_pose_filters(self, sides=SIDES):
        for side in sides:
            if side in self.pose_filters:
                self.pose_filters[side].reset()

    def attach_hardware(self, output, joints):
        """Attach a connected output, retaining its verified startup hold goal."""
        values = {side: np.asarray(joints[side], dtype=float) for side in SIDES}
        if any(value.shape != (7,) or not np.isfinite(value).all()
               or not self.kine.valid_joints(side, value)
               for side, value in values.items()):
            raise ValueError('实机关节反馈与当前运动学模型不匹配')
        if self.config['hardware'].get('motion_scope') == 'full':
            for side in self.control_sides:
                self._check_full_arm_singularity(side, values[side], ValueError)
        hold_targets = getattr(output, 'hold_targets', None)
        commands = ({s: np.asarray(hold_targets[s], dtype=float) for s in SIDES}
                    if hold_targets is not None else values)
        if any(q.shape != (7,) or not np.isfinite(q).all() or not self.kine.valid_joints(s, q)
               for s, q in commands.items()):
            raise ValueError('实机关节保持目标与当前运动学模型不匹配')
        if self.config['hardware'].get('motion_scope') == 'full':
            for side in self.control_sides:
                self._check_full_arm_singularity(side, commands[side], ValueError)
        self.hardware_output = output
        self.source = 'quest'
        self.frame = None
        self.measured_q = {s: values[s].copy() for s in SIDES}
        if self.wrist is not None:
            self.wrist = WristEnvelope(values, self.config['hardware'])
        self._set_joints(commands)
        self.reason = '天机已连接并待机；松开扳机后按 A 校准'

    def _check_full_arm_singularity(self, side, joints, error_type=IKError):
        """Reject startup or live commands that enter a rank-deficient pose."""
        threshold = self.config['hardware'].get('minimum_jacobian_singular_value', 0.)
        if threshold <= 0:
            return
        actual = self._minimum_singular_value(side, joints)
        if actual < threshold:
            raise error_type(f'{side}: 当前姿态接近运动学奇异位'
                             f'（最小奇异值 {actual:.6f}）；已暂停整臂跟随')

    def _minimum_singular_value(self, side, joints):
        singular = np.linalg.svd(self.kine.jacobian(side, joints), compute_uv=False)
        if singular.shape != (6,) or not np.isfinite(singular).all():
            raise IKError(f'{side}: 运动学雅可比无效；已暂停整臂跟随')
        return float(singular[-1])

    def _limit_live_singularity(self, side, start, proposed):
        """Keep live joint commands inside a small buffer around the singular gate.

        Near the threshold, a rejected IK step should behave like a workspace
        boundary: hold or take the feasible fraction. Hard startup checks and
        truly rank-deficient feedback remain separate failure paths.
        """
        threshold = self.config['hardware'].get('minimum_jacobian_singular_value', 0.)
        if threshold <= 0:
            return proposed, False
        buffer = threshold + .005
        current = self._minimum_singular_value(side, start)
        candidate = self._minimum_singular_value(side, proposed)
        if current < threshold * .5:
            raise IKError(f'{side}: 实际姿态已深度进入奇异区'
                          f'（最小奇异值 {current:.6f}）；已暂停整臂跟随')
        if current < buffer:
            # A small amount of settling may put feedback just below the
            # buffer. Permit only steps that increase the singular value.
            return (proposed, False) if candidate > current + 1e-6 else (start.copy(), True)
        if candidate >= buffer:
            return proposed, False
        low, high = 0., 1.
        for _ in range(12):
            fraction = (low + high) / 2.
            q = start + fraction * (proposed - start)
            if self._minimum_singular_value(side, q) >= buffer:
                low = fraction
            else:
                high = fraction
        return start + low * (proposed - start), True

    @property
    def hardware_enabled(self):
        return self.hardware_output is not None

    def _hold_hardware(self):
        if self.hardware_output is None or self.hardware_fault:
            return
        try:
            held = self.hardware_output.hold(self.q)
            self._sync_measured(SIDES, command_joints=held)
        except Exception as exc:
            self._hardware_failed(exc)

    def _hardware_failed(self, exc):
        self.hardware_fault = self.hardware_fault or str(exc)
        self.hardware_output.stop(self.hardware_fault)
        self.pause(f'实机故障已锁存：{self.hardware_fault}；排查后重启服务')

    def _sync_measured(self, sides, command_joints=None):
        joints = self.hardware_output.read_joints()
        for side in SIDES:
            if not self.kine.valid_joints(side, joints[side]):
                raise RuntimeError(f'{side}: 实机反馈超出当前模型关节约束')
            self.measured_q[side] = joints[side].copy()
        for side in sides:
            command = joints[side] if command_joints is None else np.asarray(command_joints[side], dtype=float)
            if command.shape != (7,) or not np.isfinite(command).all() or not self.kine.valid_joints(side, command):
                raise RuntimeError(f'{side}: 实机保持目标超出当前模型关节约束')
            self.q[side] = (self.wrist.hold_target(side, command) if self.wrist is not None
                            else command.copy())
            self.actual[side] = self.kine.fk(side, self.q[side])
            self.target[side] = self.actual[side].copy()
            self.filtered[side] = self.actual[side].copy()

    def _poll_hardware(self):
        if self.hardware_output is None:
            return True
        if self.hardware_fault:
            return False
        try:
            was_enabled = self.hardware_output.enabled
            self.hardware_feedback = self.hardware_output.poll()
            self.measured_q = {s: q.copy() for s, q in self.hardware_feedback['joints'].items()}
            if self.wrist is not None:
                self.wrist.check_feedback(self.measured_q)
            if any(not self.kine.valid_joints(s, q) for s, q in self.measured_q.items()):
                raise RuntimeError('实机反馈超出当前模型关节约束')
            if not was_enabled:
                # During nonblocking servo setup the arms can settle. Re-anchor
                # only once ready, using the latest input and measured joints.
                command_joints = ({s: self.wrist.hold_target(s, q) for s, q in self.measured_q.items()}
                                  if self.wrist is not None else self.measured_q)
                self._set_joints(command_joints)
                self.anchors.clear()
                self.last_controller.clear()
                self._reset_pose_filters()
                self.active = {s: False for s in SIDES}
            error = max(float(np.max(np.abs(self.q[s] - self.measured_q[s]))) for s in SIDES)
            settings = self.config['hardware']
            if self.hardware_output.enabled and error > settings['max_tracking_error_deg']:
                if self.tracking_error_since is None:
                    self.tracking_error_since = self.clock()
                if self.clock() - self.tracking_error_since >= settings['tracking_error_timeout_s']:
                    raise RuntimeError(f'关节跟随误差持续过大（{error:.1f}°）')
            else:
                self.tracking_error_since = None
            return True
        except Exception as exc:
            self._hardware_failed(exc)
            return False

    def _hardware_ready(self):
        if self.hardware_output is None:
            return True
        if self.hardware_fault:
            raise RuntimeError(self.hardware_fault)
        if self.hardware_output.enabled:
            return True
        self.hardware_output.enable()
        self.reason = '实机正在使能；保持新鲜手柄输入，松开扳机可取消'
        return False

    def pause(self, reason='已暂停；按 A 重新校准，再开始遥操'):
        self.enabled = False
        self.keyboard_follow = False
        self.ui_follow = None
        self.trigger_release_required = {s: False for s in SIDES}
        self.reset_plan = None
        self.reason = (f'实机故障已锁存：{self.hardware_fault}；排查后重启服务'
                       if self.hardware_fault else reason)
        self.anchors.clear()
        self.last_controller.clear()
        self._reset_pose_filters()
        self.calibrated_heading_deg = None
        self.tracking_loss_since = None
        self.tracking_loss_active = False
        self.tracking_recovery_pending = False
        self.tracking_recovery_valid_since = None
        self.tracking_recovery_valid_frames = 0
        self.tracking_recovery_last_valid_s = None
        self.tracking_recovery_last_controller = None
        self.tracking_recovery_resume = False
        for side in SIDES:
            self.active[side] = False
            self.ik_feedback[side] = {'limited': False, 'iterations': 0}
        # A stale Quest frame must never leave the physical controller waiting
        # for a moving target. Keep impedance active and send one hold target;
        # shutdown() separately transitions the robot to standby.
        self._hold_hardware()

    def attach_keyboard(self):
        self.set_source('quest')
        self.pause('键盘已连接；按 A 校准，空格开始 / 停止跟随')
        self.keyboard_mode = True
        self.keyboard_deadline = self.clock() + self.KEYBOARD_TIMEOUT_S

    def detach_keyboard(self, reason='键盘已退出；双臂已暂停'):
        self.pause(reason)
        self.keyboard_mode = False
        self.keyboard_deadline = None

    def keyboard_heartbeat(self):
        if not self.keyboard_mode:
            raise ValueError('键盘控制未连接；请重新运行 teleop.ps1')
        if self.clock() >= self.keyboard_deadline:
            self.detach_keyboard('键盘心跳超时；双臂已暂停，请重新运行 teleop.ps1')
            raise ValueError(self.reason)
        self.keyboard_deadline = self.clock() + self.KEYBOARD_TIMEOUT_S

    def trigger(self, side):
        if self.hardware_output is not None and side not in self.control_sides:
            return 0.
        if self.keyboard_mode:
            return float(self.keyboard_follow)
        if self.ui_follow is not None:
            return float(self.ui_follow)
        return self.frame.controllers[side].grip if self.frame else 0.

    @property
    def triggers_overridden(self):
        return self.keyboard_mode or self.ui_follow is not None

    def release_clutch(self, sides, *, hardware_reanchor=True):
        sides = tuple(sides)
        releasing = any(self.active[s] for s in sides)
        if hardware_reanchor and self.hardware_output is not None and releasing and not self.hardware_fault:
            try:
                self._sync_measured(sides)
                self.hardware_output.send_joint_targets(self.q)
            except Exception as exc:
                self._hardware_failed(exc)
                return
        for side in sides:
            self.active[side] = False
            self.anchors.pop(side, None)
            self.last_controller.pop(side, None)
            self._reset_pose_filters((side,))
            self.target[side] = self.actual[side].copy()
            self.filtered[side] = self.actual[side].copy()
            self.ik_feedback[side] = {'limited': False, 'iterations': 0}

    def toggle_keyboard_follow(self):
        self.keyboard_heartbeat()
        if self.tracking_recovery_pending:
            self.pause('已取消追踪恢复；按 A 校准后再开始')
            return
        self.set_follow(not self.keyboard_follow)

    def use_controller_triggers(self):
        """Return simulation checkboxes to independent trigger control."""
        if self.ui_follow is not None:
            self.release_clutch(SIDES)
            self.ui_follow = None

    def set_follow(self, following):
        """Shared, idempotent desktop/keyboard clutch; preserves calibration."""
        if type(following) is not bool:
            raise ValueError('跟随状态必须为 true 或 false')
        if not following:
            self.tracking_recovery_pending = False
            self.tracking_recovery_valid_since = None
            self.tracking_recovery_valid_frames = 0
            self.tracking_recovery_last_valid_s = None
            self.tracking_recovery_last_controller = None
            self.tracking_recovery_resume = False
            if self.reset_plan is not None:
                self.pause('已中止恢复；保持当前位置，请按 A 重新校准')
                return
            if self.keyboard_mode:
                self.keyboard_follow = False
            else:
                self.ui_follow = False
            self._hold_hardware()
            if self.hardware_fault:
                return
            self.release_clutch(SIDES, hardware_reanchor=False)
            self.reason = '已暂停跟随；保持当前位置，可再次开始遥操'
            return
        if self.keyboard_mode and self.clock() >= self.keyboard_deadline:
            self.detach_keyboard('键盘心跳超时；双臂已暂停，请重新运行 teleop.ps1')
            raise ValueError(self.reason)
        if not self.enabled or self.reset_plan is not None:
            raise ValueError('请先按 A 校准；B 恢复完成后也需要重新校准')
        if (not self.frame or self.clock() - self.frame.received_s > self.config['input_timeout_s'] or
                self.extra_delay_s > self.config['frame_age_limit_s'] or not self.frame.focused or
                not all(self.frame.controllers[s].tracked for s in self.control_sides) or
                self.frame.safety.get('tracking_lost') or self.frame.safety.get('focus_lost')):
            self.pause('没有新鲜、有效的双手柄追踪；请恢复后按 A 校准')
            raise ValueError(self.reason)
        if self.hardware_fault:
            raise ValueError(f'实机故障已锁存：{self.hardware_fault}；请重启服务')
        # Starting from the keyboard is a fresh clutch engagement. Never replay
        # the operator's displacement between startup calibration and Space.
        self.release_clutch(SIDES)
        if self.keyboard_mode:
            self.keyboard_follow = True
        else:
            self.ui_follow = True
        self.reason = '遥操已开始；点击暂停或按空格停止跟随' if self.keyboard_mode else '遥操已开始；点击暂停停止跟随'

    def reset(self):
        """Start a synchronized joint-space quintic return, without a pose jump."""
        if self.reset_plan is not None:
            return
        if self.hardware_output is not None:
            self.pause('实机已暂停；本机型回零轨迹未经现场验证，请用厂家软件恢复姿态')
            raise ValueError(self.reason)
        self.pause('正在插值恢复初始姿态')
        self.reset_revision += 1
        start = {s: self.q[s].copy() for s in SIDES}
        goal = {s: np.array(self.config['arms'][s]['initial_joints_deg'], dtype=float) for s in SIDES}
        distance = max(float(np.max(np.abs(goal[s] - start[s]))) for s in SIDES)
        self._set_joints(start)
        if distance < 1e-9:
            self.reset_progress = 1.
            self.reason = '已在初始姿态；松开扳机后按 A 校准'
            return
        # Quintic smoothstep has peak slope 1.875 and peak curvature 10/sqrt(3).
        speed = min(self.config['max_reset_speed_deg_s'], self.config['max_joint_speed_deg_s'])
        duration = max(self.config['reset_duration_s'], 1.875 * distance / speed,
                       math.sqrt((10. / math.sqrt(3.)) * distance / self.config['max_reset_accel_deg_s2']))
        self.reset_progress = 0.
        self.reset_plan = {'start': start, 'goal': goal, 'duration': duration, 'elapsed': 0.}
        self.last_tick = self.clock()

    def _step_reset(self, dt):
        plan = self.reset_plan
        # Advance by bounded control steps, so a stalled event loop cannot jump home.
        plan['elapsed'] = min(plan['duration'], plan['elapsed'] + dt)
        u = plan['elapsed'] / plan['duration']
        blend = u ** 3 * (10. - 15. * u + 6. * u ** 2)
        self._set_joints({s: plan['start'][s] + blend * (plan['goal'][s] - plan['start'][s]) for s in SIDES})
        if self.hardware_output is not None:
            self.hardware_output.send_joint_targets(self.q)
        self.reset_progress = u
        self.reason = f'正在插值恢复初始姿态 · {u:.0%}'
        if u >= 1.:
            self._set_joints(plan['goal'])
            self.reset_plan = None
            self.reason = '已恢复初始姿态；松开扳机后按 A 重新校准'

    def set_source(self, source):
        if source not in ('simulation', 'quest'):
            raise ValueError('未知输入源')
        if self.hardware_output is not None and source != 'quest':
            raise ValueError('实机仅接受 Quest 输入；模拟手柄和演示不能驱动实机')
        if self.keyboard_mode and source != 'quest':
            raise ValueError('键盘遥操使用 Quest USB；请先退出 teleop.ps1')
        if source == self.source:
            return
        self.pause('输入源已切换；等待新数据')
        self.source = source
        self.frame = None
        self.clock_offset = None
        self.arrivals.clear()

    def set_scale(self, scale):
        if type(scale) not in (int, float) or not math.isfinite(scale) or not 0.05 <= scale <= 1:
            raise ValueError('移动比例须为 0.05–1.00')
        if abs(self.scale - scale) > 1e-6:
            self.pause('移动比例已更改；请按 A 重新校准')
            self.scale = float(scale)

    def receive(self, frame: Frame, source: str):
        if source != self.source:
            return False
        previous = self.frame
        if previous and frame.session == previous.session:
            if frame.seq <= previous.seq:
                return False  # Do not refresh arrival time for duplicates or reordered frames.
            if frame.captured_s <= previous.captured_s:
                self.pause('采样时钟未前进；重新连接采集端')
                return False
            if frame.reference != previous.reference:
                self.pause('追踪空间已重置；请按 A 重新校准')
        else:
            if previous:
                self.pause('采集会话已变化；请按 A 重新校准')
            self.clock_offset = None
            self.arrivals.clear()

        # Minimum observed offset estimates excess queue delay, NOT absolute USB latency.
        offset = frame.received_s - frame.captured_s
        self.clock_offset = offset if self.clock_offset is None else min(offset, self.clock_offset)
        self.consume_delay_s = max(0., self.clock() - frame.received_s)
        self.extra_delay_s = max(0., offset - self.clock_offset) + self.consume_delay_s
        if self.extra_delay_s > self.config['frame_age_limit_s']:
            self.pause('输入帧积压；等待新鲜数据后按 A 重新校准')
            return False
        self.frame = frame
        self.arrivals.append(frame.received_s)
        # Rate limiting and mailbox coalescing may omit an intermediate sample.
        # Preserve its release/loss edge before using the latest position.
        if not self.triggers_overridden:
            self.release_clutch(frame.safety.get('released', []))
        tracking_lost = (frame.safety.get('tracking_lost', False) or
                         not all(frame.controllers[s].tracked for s in self.control_sides))
        # Queue coalescing may carry an older loss edge together with a newer
        # valid pose. During the grace window, the newest pose decides recovery.
        if (self.hardware_output is not None and self.config.get('tracking_loss_grace_s', 0.) > 0 and
                (not self.tracking_recovery_pending or not self.tracking_recovery_resume) and
                all(frame.controllers[s].tracked for s in self.control_sides)):
            tracking_lost = False
            # Calibrate/set_follow inspect this same frame later.
            frame.safety['tracking_lost'] = False
        if not frame.focused or frame.safety.get('focus_lost', False):
            self.pause('Quest 会话不可见 / 已失焦')
        elif self.tracking_recovery_pending:
            if tracking_lost:
                self.tracking_recovery_valid_since = None
                self.tracking_recovery_valid_frames = 0
                self.tracking_recovery_last_valid_s = None
                self.tracking_recovery_last_controller = None
            else:
                sample_time = self.clock()
                controllers = {
                    side: pose_matrix(frame.controllers[side].position,
                                      frame.controllers[side].quaternion) @ self.offsets[side]
                    for side in self.control_sides
                }
                previous_controllers = self.tracking_recovery_last_controller
                discontinuity = (previous_controllers is not None and any(
                    np.linalg.norm(controllers[side][:3, 3] - previous_controllers[side][:3, 3]) >
                    self.config['controller_jump_m'] or
                    rotation_angle(controllers[side][:3, :3], previous_controllers[side][:3, :3]) >
                    self.config['controller_jump_rad']
                    for side in self.control_sides))
                if (self.tracking_recovery_last_valid_s is None or
                        discontinuity or
                        (self.tracking_recovery_resume and
                         sample_time - self.tracking_recovery_last_valid_s > self.TRACKING_RECOVERY_MAX_FRAME_GAP_S)):
                    self.tracking_recovery_valid_since = sample_time
                    self.tracking_recovery_valid_frames = 1
                else:
                    self.tracking_recovery_valid_frames += 1
                self.tracking_recovery_last_valid_s = sample_time
                self.tracking_recovery_last_controller = controllers
                stable_s = self.TRACKING_RECOVERY_STABLE_S if self.tracking_recovery_resume else .2
                min_frames = self.TRACKING_RECOVERY_MIN_FRAMES if self.tracking_recovery_resume else 2
                if (sample_time - self.tracking_recovery_valid_since >= stable_s and
                        self.tracking_recovery_valid_frames >= min_frames):
                    # Both actions use this fresh frame and measured joints. The
                    # clutch is then rebuilt at the next control tick, so hand
                    # motion during the outage cannot become a target jump.
                    resume = self.tracking_recovery_resume
                    try:
                        self.calibrate()
                        if resume:
                            self.set_follow(True)
                        else:
                            self.reason = '追踪已稳定恢复并重建映射；按空格继续'
                    except ValueError as exc:
                        self.pause(f'自动重建映射失败：{exc}；请按 A 校准')
        elif self.reset_plan is None and tracking_lost:
            grace = float(self.config.get('tracking_loss_grace_s', 0.)) if self.hardware_output is not None else 0.
            now = self.clock()
            if grace <= 0. or not self.enabled:
                self.pause('手柄失去位置追踪；双臂已暂停')
            else:
                if self.tracking_loss_since is None:
                    self.tracking_loss_since = now
                    self._hold_hardware()
                    if self.hardware_fault:
                        return True
                    self.active = {s: False for s in SIDES}
                self.tracking_loss_active = True
                if now - self.tracking_loss_since > grace:
                    self._expire_tracking_loss()
                else:
                    self.reason = f'手柄短暂失去追踪；保持当前位置（缓冲 {grace * 1000:.0f} ms）'
        elif self.tracking_loss_active:
            # Re-anchor on the first valid frame so displacement accumulated
            # while tracking was missing can never become a command jump.
            self.tracking_loss_since = None
            self.tracking_loss_active = False
            self.anchors.clear()
            self.last_controller.clear()
            self._reset_pose_filters()
            self.active = {s: False for s in SIDES}
            for side in SIDES:
                self.target[side] = self.actual[side].copy()
                self.filtered[side] = self.actual[side].copy()
            self.reason = '手柄追踪已恢复；已重新锚定'
        return True

    def _expire_tracking_loss(self):
        if (self.tracking_loss_active and self.tracking_loss_since is not None and
                self.clock() - self.tracking_loss_since > self.config.get('tracking_loss_grace_s', 0.)):
            recover = (self.enabled and self.keyboard_mode and not self.hardware_fault and
                       self.config.get('keyboard_auto_recover_tracking', False))
            resume = bool(recover and self.keyboard_follow and
                          self.config.get('keyboard_auto_resume_tracking', False))
            self.pause('手柄持续失去位置追踪；双臂已暂停，请重新校准')
            if recover and not self.hardware_fault:
                self.tracking_recovery_pending = True
                self.tracking_recovery_resume = resume
                self.reason = ('持续丢踪；等待稳定追踪后自动重锚并继续跟随' if resume else
                               '持续丢踪；等待稳定追踪后重建映射，再按空格继续')

    def _recover_controller_discontinuity(self, reason):
        recover = (self.enabled and self.keyboard_mode and self.keyboard_follow and
                   not self.hardware_fault and
                   self.config.get('keyboard_auto_recover_tracking', False) and
                   self.config.get('keyboard_auto_resume_tracking', False))
        self.pause(reason)
        if recover and not self.hardware_fault:
            self.tracking_recovery_pending = True
            self.tracking_recovery_resume = True
            self.reason = f'{reason}；双臂与灵巧手保持，追踪稳定后自动重新对齐并继续'

    def arm(self):
        """Compatibility alias for the original desktop command."""
        self.calibrate()

    def calibrate(self, *, web_control=False):
        """Freeze Quest viewer heading and pair controller/wrist and robot poses.

        Calibration never moves the robot. The first clutch uses this sample;
        later clutch releases/re-engagements retain their no-jump behavior.
        """
        if self.reset_plan is not None:
            raise ValueError('正在恢复初始姿态；完成后再按 A 校准')
        if self.hardware_fault:
            raise ValueError(f'实机故障已锁存：{self.hardware_fault}；请重启服务')
        if self.hardware_output is not None and any(self.active.values()):
            raise ValueError('实机跟随中；先松开扳机或暂停，再按 A 校准')
        if not self.frame or self.clock() - self.frame.received_s > self.config['input_timeout_s']:
            raise ValueError('没有新鲜的手柄数据')
        if self.extra_delay_s > self.config['frame_age_limit_s']:
            raise ValueError('输入仍有积压；请等待新鲜帧')
        if (not self.frame.focused or not all(self.frame.controllers[s].tracked for s in self.control_sides) or
                self.frame.safety.get('tracking_lost', False) or self.frame.safety.get('focus_lost', False)):
            raise ValueError('请等待两个手柄追踪正常')
        if self.triggers_overridden and any(self.trigger(s) > 0.35 for s in SIDES):
            raise ValueError('请先暂停跟随，再按 A 校准')
        # XR local-floor yaw need not match the operator's forward direction.
        # Derive heading at A only, not from mounting angles or live head turns.
        if self.source == 'quest' and self.config.get('heading_source', 'head') == 'head':
            if self.frame.head is None:
                raise ValueError('缺少头显朝向；请刷新 Quest 采集页并重新进入 VR 后按 A')
            if not self.frame.head.tracked:
                raise ValueError('头显朝向追踪无效；请平视操作正前方后按 A')
            basis, heading = heading_basis(self.frame.head.quaternion, self.config['operator_yaw_deg'])
        else:
            basis = Rotation.from_euler('z', self.config['operator_yaw_deg'], degrees=True).as_matrix() @ XR_TO_WORLD
            heading = float(self.config['operator_yaw_deg']) if self.source == 'quest' else None
        # Calibration does not move an arm. An already-high physical trigger
        # may be a browser/runtime anomaly; accept A, but require a later release
        # before that controller can start following. Web/keyboard Start remains
        # an explicit alternative and builds a fresh clutch anchor.
        release_required = {s: not web_control and not self.triggers_overridden and self.trigger(s) > .35 for s in SIDES}
        if self.hardware_output is not None:
            try:
                self._sync_measured(SIDES, command_joints=getattr(self.hardware_output, 'hold_targets', None))
            except Exception as exc:
                self._hardware_failed(exc)
                raise ValueError(self.reason) from exc
        anchors = {
            side: (pose_matrix(self.frame.controllers[side].position,
                               self.frame.controllers[side].quaternion) @ self.offsets[side],
                   self.actual[side].copy())
            for side in self.control_sides if not release_required[side]
        }
        self.basis, self.calibrated_heading_deg, self.anchors = basis, heading, anchors
        self.trigger_release_required = release_required
        self.trigger_release_after_seq = self.frame.seq
        if web_control:
            self.ui_follow = False
        self.last_controller.clear()
        self._reset_pose_filters()
        self.active = {s: False for s in SIDES}
        for side in SIDES:
            self.target[side] = self.actual[side].copy()
            self.filtered[side] = self.actual[side].copy()
        self.enabled = True
        self.tracking_loss_since = None
        self.tracking_loss_active = False
        self.tracking_recovery_pending = False
        self.tracking_recovery_valid_since = None
        self.tracking_recovery_valid_frames = 0
        self.tracking_recovery_last_valid_s = None
        self.tracking_recovery_last_controller = None
        self.tracking_recovery_resume = False
        self.calibration_revision += 1
        self.reason = ('朝向与相对位姿已校准' if self.source == 'quest' else '相对位姿已校准') + (
            ('；按空格开始 / 停止键盘遥操' if self.keyboard_mode else
             '；按住扳机键跟随，B 暂停双臂') if self.hardware_output is not None else
            '；按住扳机键跟随，B 恢复初始姿态')

    def step(self):
        now = self.clock()
        if not self._poll_hardware():
            return
        if self.keyboard_mode and now >= self.keyboard_deadline:
            self.detach_keyboard('键盘心跳超时；双臂已暂停，请重新运行 teleop.ps1')
        dt = min(max(now - self.last_tick, 0.001), 0.05)
        self.last_tick = now
        if self.frame and now - self.frame.received_s > self.config['input_timeout_s']:
            if self.enabled or self.reset_plan is not None or self.tracking_recovery_pending:
                self.pause('数据超时 / USB 断连；双臂已暂停')
            return
        self._expire_tracking_loss()
        if self.reset_plan is not None:
            try:
                self._step_reset(dt)
            except (IKError, ValueError, RuntimeError) as exc:
                if isinstance(exc, RuntimeError):
                    self.hardware_fault = str(exc)
                self.pause(str(exc))
            return
        if not self.enabled or not self.frame:
            return
        if self.tracking_loss_active:
            return
        candidates = {}
        try:
            # Physical triggers and web/keyboard follow share the same servo
            # gate. Never stall this loop while waiting for robot mode feedback.
            if self.hardware_output is not None:
                wants_follow = any(self.trigger(s) > .65 and
                                   (self.triggers_overridden or not self.trigger_release_required[s])
                                   for s in self.control_sides)
                if wants_follow and not self._hardware_ready():
                    return
                if not wants_follow and self.hardware_output.enabling:
                    self.hardware_output.cancel_enable()
            for side in SIDES:
                c = self.frame.controllers[side]
                gripping = (side in self.control_sides and
                            self.trigger(side) > (0.35 if self.active[side] else 0.65))
                if not self.triggers_overridden and self.trigger_release_required[side]:
                    if self.frame.seq > self.trigger_release_after_seq and (
                            c.grip <= .35 or side in self.frame.safety.get('released', [])):
                        self.trigger_release_required[side] = False
                    else:
                        gripping = False
                if not gripping:
                    # Preserve A's anchor until the first grip. Only an actual
                    # clutch release discards it for a no-jump re-engagement.
                    if self.active[side]:
                        self.release_clutch((side,))
                    self.active[side] = False
                    self.last_controller.pop(side, None)
                    self._reset_pose_filters((side,))
                    self.target[side] = self.actual[side].copy()
                    self.ik_feedback[side] = {'limited': False, 'iterations': 0}
                    continue
                controller = pose_matrix(c.position, c.quaternion) @ self.offsets[side]
                if not self.active[side]:
                    if self.hardware_output is not None:
                        self._sync_measured((side,))
                        self.anchors.pop(side, None)
                    if side not in self.anchors:
                        self.anchors[side] = (controller.copy(), self.actual[side].copy())
                    self.filtered[side] = self.actual[side].copy()
                    self.active[side] = True
                previous = self.last_controller.get(side)
                if previous is not None:
                    if np.linalg.norm(controller[:3, 3] - previous[:3, 3]) > self.config['controller_jump_m']:
                        raise ControllerDiscontinuity('手柄位置跳变')
                    if rotation_angle(controller[:3, :3], previous[:3, :3]) > self.config['controller_jump_rad']:
                        raise ControllerDiscontinuity('手柄姿态跳变')
                self.last_controller[side] = controller.copy()
                controller_anchor, robot_anchor = self.anchors[side]
                desired = relative_target(controller, controller_anchor, robot_anchor, self.basis, self.scale)
                if self.wrist is not None:
                    desired[:3, 3] = self.actual[side][:3, 3]
                base = self.kine.arms[side]['base']
                radius = self.config['workspace_radius_m']
                if np.linalg.norm(self.actual[side][:3, 3] - base[:3, 3]) > radius + 1e-6:
                    raise IKError(f'{side}: 实际末端已超出预演工作区；已暂停跟随')
                displacement = desired[:3, 3] - base[:3, 3]
                distance = float(np.linalg.norm(displacement))
                workspace_limited = distance > radius
                if workspace_limited:
                    desired[:3, 3] = base[:3, 3] + displacement * (max(0., radius - .005) / distance)
                    # Slide the position anchor along the boundary. Reversing
                    # the hand then moves inward immediately, without a dead
                    # zone from accumulated motion beyond the workspace.
                    controller_anchor[:3, 3] = (controller[:3, 3] -
                        self.basis.T @ (desired[:3, 3] - robot_anchor[:3, 3]) / self.scale)
                self.target[side] = desired
                # Anchor the next command at achieved FK, so joint saturation cannot
                # accumulate an unreachable Cartesian target ahead of the arm.
                adaptive = self.pose_filters[side](desired, dt) if side in self.pose_filters else desired
                filtered = smooth_target(self.actual[side], adaptive, dt,
                                         self.config['filter_time_s'], self.config['max_target_speed_m_s'],
                                         self.config['max_target_rotation_rad_s'])
                max_step = min(self.config['max_joint_step_deg'], self.config['max_joint_speed_deg_s'] * dt)
                if self.wrist is not None:
                    max_step = min(max_step, self.config['hardware']['wrist_max_speed_deg_s'] * dt)
                    result = self.wrist.track(self.kine, side, filtered, self.q[side], max_step)
                else:
                    result = self.kine.track(side, filtered, self.q[side], max_step)
                if workspace_limited:
                    result.limited = True
                q = result.joints_deg
                if (self.hardware_output is not None and
                        self.config['hardware'].get('motion_scope') == 'full'):
                    q, singular_limited = self._limit_live_singularity(side, self.q[side], q)
                    if singular_limited:
                        result.joints_deg = q
                        result.limited = True
                delta = q - self.q[side]
                largest = float(np.max(np.abs(delta)))
                if largest > self.config['max_joint_step_deg']:
                    raise IKError('IK 关节构型跳变；需要重新对齐')
                if largest > self.config['max_joint_speed_deg_s'] * dt:
                    q = self.q[side] + delta * self.config['max_joint_speed_deg_s'] * dt / largest
                candidates[side] = (q, result)
            # Commit both arms only after every requested solve passes.
            for side, (q, result) in candidates.items():
                self.q[side] = q
                self.actual[side] = self.kine.fk(side, q)
                self.filtered[side] = self.actual[side].copy()
                self.ik_feedback[side] = {'limited': result.limited, 'iterations': result.iterations,
                    'position_error_m': float(np.linalg.norm(self.target[side][:3, 3] - self.actual[side][:3, 3])),
                    'rotation_error_rad': rotation_angle(self.target[side][:3, :3], self.actual[side][:3, :3])}
            if self.hardware_output is not None and candidates:
                self.hardware_output.send_joint_targets(self.q)
            limited = [label for side, label in (('left', '左臂'), ('right', '右臂')) if self.ik_feedback[side]['limited']]
            waiting = not self.triggers_overridden and any(self.trigger_release_required.values())
            self.reason = ('、'.join(limited) + '限速 / 目标受限；可放慢或移回可达区域' if limited else
                           ('键盘跟随中；空格暂停' if self.keyboard_mode else
                            '扳机跟随中；B 暂停双臂' if self.hardware_output is not None else '扳机跟随中；B 恢复初始姿态') if any(self.active.values()) else
                           '已校准；手柄扳机需先完全松开，再按下接管' if waiting else
                           '相对位姿已校准；按空格开始' if self.keyboard_mode else
                           '相对位姿已校准；按住扳机键跟随')
        except ControllerDiscontinuity as exc:
            self._recover_controller_discontinuity(str(exc))
        except (IKError, ValueError, RuntimeError) as exc:
            if self.hardware_output is not None and isinstance(exc, RuntimeError):
                self._hardware_failed(exc)
            else:
                self.pause(str(exc))

    def snapshot(self):
        age = self.clock() - self.frame.received_s if self.frame else None
        fresh = age is not None and age < self.config['input_timeout_s']
        hz = 0.
        if fresh and len(self.arrivals) > 1:
            duration = self.arrivals[-1] - self.arrivals[0]
            hz = (len(self.arrivals) - 1) / duration if duration > 0 else 0.
        return {
            'type': 'state', 'virtual_only': not self.hardware_enabled,
            'hardware': self.hardware_enabled,
            'motion_scope': 'wrist' if self.wrist is not None else 'full',
            'control_sides': list(self.control_sides),
            'wrist_limits': ({'excursion_deg': self.wrist.excursion,
                              'max_speed_deg_s': self.config['hardware']['wrist_max_speed_deg_s'],
                              'locked_joints': [1, 2, 3, 4, 5]} if self.wrist is not None else None),
            'hardware_identity': getattr(self.hardware_output, 'identity', None),
            'hardware_connected': bool(self.hardware_output and self.hardware_output.connected),
            'hardware_enabled': bool(self.hardware_output and self.hardware_output.enabled),
            'hardware_fault': self.hardware_fault,
            'hardware_enabling': bool(self.hardware_output and self.hardware_output.enabling),
            'hardware_feedback': ({k: v for k, v in self.hardware_feedback.items() if k != 'joints'}
                                  if self.hardware_feedback is not None else None),
            'model': self.config['model'],
            'visual_model': self.config.get('visual_model'),
            'model_label': self.config['model_label'], 'source': self.source,
            'enabled': self.enabled, 'following': any(self.active.values()), 'reason': self.reason,
            'keyboard_mode': self.keyboard_mode, 'keyboard_follow': self.keyboard_follow,
            'ui_follow': self.ui_follow,
            'trigger_source': 'keyboard' if self.keyboard_mode else 'web' if self.ui_follow is not None else 'controller',
            'calibration_revision': self.calibration_revision, 'reset_revision': self.reset_revision,
            'calibrated_heading_deg': self.calibrated_heading_deg,
            'heading_source': self.config.get('heading_source', 'head'),
            'tracking_loss_grace_s': float(self.config.get('tracking_loss_grace_s', 0.)),
            'tracking_loss_active': bool(self.tracking_loss_active),
            'tracking_recovery_pending': bool(self.tracking_recovery_pending),
            'input_filter': self.config.get('input_filter', {}).get('type', 'none'),
            'hardware_send_clear_retries': int(getattr(self.hardware_output, 'send_clear_retries', 0)),
            'buttons': dict(self.frame.buttons) if fresh and self.frame else {
                'calibrate': False, 'reset': False},
            'input_to_world_rotation': self.basis.tolist(),
            'resetting': self.reset_plan is not None, 'reset_progress': self.reset_progress,
            'scale': self.scale, 'fresh': fresh, 'input_hz': round(hz, 1),
            'frame_age_ms': round(age * 1000, 1) if age is not None else None,
            'input_session': self.frame.session if self.frame else None,
            'input_reference': self.frame.reference if self.frame else None,
            'input_focus_lost': bool(self.frame and self.frame.safety.get('focus_lost')),
            'queue_delay_ms': round(self.extra_delay_s * 1000, 1),
            'receive_to_consume_ms': round(self.consume_delay_s * 1000, 1),
            'seq': self.frame.seq if self.frame else None,
            'arms': {s: {
                'joints_deg': self.q[s].tolist(), 'active': self.active[s],
                'measured_joints_deg': self.measured_q[s].tolist() if s in self.measured_q else None,
                'measured_pose': serialized_pose(self.kine.fk(s, self.measured_q[s])) if s in self.measured_q else None,
                'measured_chain': [serialized_pose(t) for t in self.kine.chain(s, self.measured_q[s])] if s in self.measured_q else None,
                'tracking_error_deg': float(np.max(np.abs(self.q[s] - self.measured_q[s]))) if s in self.measured_q else None,
                'ik': self.ik_feedback[s],
                'tracked': bool(fresh and self.frame.focused and self.frame.controllers[s].tracked),
                'grip': self.trigger(s),
                'raw_trigger': self.frame.controllers[s].grip if fresh and self.frame.focused and self.frame.controllers[s].tracked else None,
                'trigger_release_required': self.trigger_release_required[s],
                'actual': serialized_pose(self.actual[s]), 'target': serialized_pose(self.target[s]),
                'chain': [serialized_pose(t) for t in self.kine.chain(s, self.q[s])],
            } for s in SIDES},
        }
