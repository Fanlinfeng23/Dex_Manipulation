"""Tianji joint impedance output using the vendor Marvin SDK.

The Vive opt-in loads impedance parameters before servo-on, following Wuji's
chest driver. Waiting advances through poll(), outside the live input loop.
"""
from __future__ import annotations

import importlib.util
import ipaddress
import logging
import time
from pathlib import Path

import numpy as np

from .config import SDK_ROOT
from .vendor_checks import bind_robot_abi, version_info, controller_model_info
from .wrist import WristEnvelope

SIDES = ('left', 'right')


class HardwareError(RuntimeError):
    """Latched robot connection, feedback or command failure."""


class TianjiHardwareOutput:
    def __init__(self, config, logger=None, clock=time.monotonic):
        self.config = config
        self.settings = config.get('hardware', {})
        self.control_sides = tuple(self.settings.get('control_sides', SIDES))
        self.inactive_sides = tuple(side for side in SIDES if side not in self.control_sides)
        self.logger = logger or logging.getLogger(__name__)
        self.clock = clock
        self.robot = None
        self.connected = self.enabled = False
        self.phase = None
        self.fault = None
        self.feedback = None
        self.identity = None
        self._motion_started = False
        self._standby_confirmed = None
        self._tool_configured = False
        self._tool_ready_at = None
        self._wrist = None
        self._last_targets = None
        self._last_target_time = None
        # Preserve the verified startup goal while the operator prepares the
        # Quest input.  A pause during motion replaces this once with the
        # current measured pose; repeated pauses must not ratchet it lower.
        self._hold_targets = None
        self._startup_joints = None
        self._serials = [None, None]
        self._updated = [None, None]
        self._changed = [False, False]
        # Marvin's SDK sends asynchronously.  OnClearSet() deliberately
        # returns false while the previous packet is still in the SDK's
        # 1 kHz sender (m_SendTag == 50/100).  A single false is therefore a
        # transient transport-busy condition, not a robot fault.
        self._send_clear_timeout_s = float(self.settings.get('send_clear_timeout_s', .04))
        if not np.isfinite(self._send_clear_timeout_s) or self._send_clear_timeout_s <= 0:
            raise ValueError('hardware.send_clear_timeout_s 必须为正有限数值')
        self.send_clear_retries = 0
        wrapper = SDK_ROOT / 'SDK_PYTHON/fx_robot.py'
        if not wrapper.is_file():
            raise FileNotFoundError(f'缺少天机控制 SDK: {wrapper}')
        self._robot_module = self._load_module('_quest_tianji_robot', wrapper)
        self._dcss_type = self._robot_module.DCSS
        try:
            self.robot = self._robot_module.Marvin_Robot()
            bind_robot_abi(self.robot, self._dcss_type)
        except OSError as exc:
            raise HardwareError('无法加载 libMarvinSDK.so；检查架构与厂家 SDK 文件') from exc

    @staticmethod
    def _load_module(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f'无法加载 SDK 模块: {path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _check(result, action):
        if not result:
            raise HardwareError(f'天机命令失败: {action}')

    def _clear_set(self):
        """Wait briefly for the SDK's asynchronous send buffer to become free.

        The vendor API documents ``clear_set``/``send_cmd`` as a pair, but its
        implementation returns false when the background sender has not yet
        consumed the previous packet.  Retrying here preserves the official
        packet sequence while preventing a harmless sub-millisecond race from
        stopping an otherwise healthy teleoperation session.
        """
        deadline = self.clock() + self._send_clear_timeout_s
        attempts = 0
        while True:
            attempts += 1
            if self.robot.clear_set():
                self.send_clear_retries += max(0, attempts - 1)
                return
            if self.clock() >= deadline:
                self.send_clear_retries += max(0, attempts - 1)
                raise HardwareError(
                    f'天机命令失败: 清空发送缓冲超时（SDK 发送线程忙，重试 {attempts} 次）')
            # 1 ms matches the SDK sender cadence and avoids a busy spin.
            time.sleep(min(.001, max(0., deadline - self.clock())))

    def _subscribe(self):
        try:
            return self.robot.subscribe(self._dcss_type())
        except Exception as exc:
            raise HardwareError(f'读取天机状态失败: {exc}') from exc

    @property
    def enabling(self):
        return self.phase is not None

    def read_feedback(self, *, initializing=False):
        now = self.clock()
        data = self._subscribe()
        values, states, serials = {}, [], []
        try:
            for i, side in enumerate(SIDES):
                row, status = data['outputs'][i], data['states'][i]
                q = np.asarray(row['fb_joint_pos'], dtype=float)
                temperatures = np.asarray(row['fb_joint_them'], dtype=float)
                serial, state, error = int(row['frame_serial']), int(status['cur_state']), int(status['err_code'])
                if q.shape != (7,) or not np.isfinite(q).all():
                    raise ValueError(f'{side}: 非有限或非 7 轴关节反馈')
                if (temperatures.shape != (7,) or not np.isfinite(temperatures).all() or
                        not np.any(np.abs(temperatures) > 1e-6) or
                        not np.all((-20 < temperatures) & (temperatures < 150))):
                    raise HardwareError(
                        f'{side}: 轴温度/编码器遥测为占位值；检查机械臂 DC48V 驱动电源和 EtherCAT')
                # Wuji's driver observes state=100–103 while switching into
                # impedance.  err_code=6 accompanies this transient on some
                # controllers; it is not a latched fault until the switch
                # times out.  All other errors remain fatal.
                switching = (self.phase == 'state_after_params' and
                             state in (100, 101, 102, 103) and error == 6)
                if error and not switching:
                    detail = '急停' if error == 13 else '控制柜故障'
                    raise HardwareError(f'{side}: {detail}，错误码 {error}；请在现场 / 厂家软件排查')
                if serial != 0 and serial != self._serials[i]:
                    self._changed[i] |= self._serials[i] not in (None, 0)
                    self._serials[i], self._updated[i] = serial, now
                if not initializing and (serial == 0 or self._updated[i] is None or not self._changed[i] or
                        now - self._updated[i] > self.settings.get('feedback_timeout_s', .2)):
                    raise HardwareError(f'{side}: 实机反馈帧未更新 / 超时')
                expected = self.servo_state if side in self.control_sides else 0
                if self.enabled and state != expected:
                    raise HardwareError(f'{side}: 实机退出控制状态（state={state}，期望 {expected}）')
                values[side] = q
                states.append(state)
                serials.append(serial)
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
            raise HardwareError(f'天机反馈格式无效: {exc}') from exc
        if self._wrist is not None:
            self._wrist.check_feedback(values)
        if self._startup_joints is not None:
            tolerance = self.settings.get('inactive_joint_tolerance_deg', .3)
            for side in self.inactive_sides:
                drift = float(np.max(np.abs(values[side] - self._startup_joints[side])))
                if drift > tolerance:
                    raise HardwareError(f'{side}: 非控制臂偏移 {drift:.2f}°，超过 {tolerance:.2f}°')
        self.feedback = {'joints': values, 'states': states, 'serials': serials,
                         'age_ms': [None if t is None else (now - t) * 1000 for t in self._updated]}
        return self.feedback

    @property
    def servo_state(self):
        return 1 if self.settings.get('servo_mode') == 'position' else 3

    def held_joints(self, joints):
        return ({s: self._wrist.hold_target(s, q) for s, q in joints.items()}
                if self._wrist is not None else {s: q.copy() for s, q in joints.items()})

    @property
    def hold_targets(self):
        return (None if self._hold_targets is None else
                {s: q.copy() for s, q in self._hold_targets.items()})

    def connect(self):
        ip = self.settings.get('robot_ip', '192.168.1.190')
        self._standby_confirmed = None
        self._tool_configured = False
        self._tool_ready_at = None
        try:
            ipaddress.IPv4Address(ip)
            self._check(self.robot.connect(ip), f'连接 {ip}')
            self.connected = True
            self.identity = version_info(self.robot)
            self.identity.update(controller_model_info(self.robot, self.config))
            # A successful connect only creates the SDK socket. Prove that BOTH
            # feedback counters advance before issuing any mode command.
            deadline = self.clock() + 3.
            while self.clock() < deadline:
                self.read_feedback(initializing=True)
                if all(self._changed):
                    break
                time.sleep(.02)
            feedback = self.read_feedback()
            if feedback['states'] != [0, 0]:
                raise HardwareError('连接时双臂必须在待机状态；请先在厂家软件停止并下伺服')
            self._startup_joints = {s: q.copy() for s, q in feedback['joints'].items()}
            if self.settings.get('motion_scope') == 'wrist':
                self._wrist = WristEnvelope(feedback['joints'], self.settings)
            return feedback['joints']
        except Exception:
            self.close()
            raise

    def read_joints(self):
        return self.read_feedback()['joints']

    def _configure_tool(self, feedback):
        """Send the configured tool payload once per connection while both arms are idle."""
        if self._tool_configured:
            return
        kine = self.settings.get('tool_kine')
        dyn = self.settings.get('tool_dyn')
        if kine is None and dyn is None:
            return
        if (not self.connected or self.enabled or self.enabling or
                feedback['states'] != [0, 0]):
            raise HardwareError('Wuji Hand 工具参数只能在双臂待机、使能前设置')
        if (not isinstance(kine, list) or len(kine) != 6 or
                not isinstance(dyn, list) or len(dyn) != 10 or
                not np.isfinite(kine).all() or not np.isfinite(dyn).all()):
            raise HardwareError('Wuji Hand 工具参数格式无效')
        self._clear_set()
        for arm in ('A', 'B'):
            self._check(self.robot.set_tool(arm, kine, dyn),
                        f'{arm} Wuji Hand 工具参数')
        self._check(self.robot.send_cmd(), '发送 Wuji Hand 工具参数')
        self._tool_configured = True
        # The vendor sequence waits 300 ms for the asynchronous tool packet
        # before switching servo mode.  enable()/poll() observe this deadline
        # without blocking the live input loop.
        self._tool_ready_at = self.clock() + .3

    def move_to_startup_pose(self):
        """Run Wuji's synchronized quintic dual-arm startup trajectory.

        This matches Wuji's state=3 joint-impedance parameters, Wuji Hand tool
        data, 3 s/100 Hz quintic timing, simultaneous A/B packets, settling,
        and 5 degree endpoint verification.
        """
        if not self.settings.get('startup_move_enabled', False):
            return self.read_joints()
        if not self.connected or self.enabled or self.enabling:
            raise HardwareError('官方初始姿态只能从已连接的双臂待机状态开始')
        feedback = self.read_feedback()
        start = feedback['joints']
        goal = {side: np.asarray(self.settings['startup_joints_deg'][side], dtype=float)
                for side in SIDES}
        if any(q.shape != (7,) or not np.isfinite(q).all() for q in goal.values()):
            raise HardwareError('官方初始姿态关节目标无效')
        duration = float(self.settings['startup_duration_s'])
        dt = float(self.settings['startup_dt_s'])
        steps = max(1, int(round(duration / dt)))
        try:
            self._configure_tool(feedback)
            if self._tool_ready_at is not None:
                time.sleep(max(0., self._tool_ready_at - self.clock()))

            # Seed measured joints in the mode-switch packet so an old SDK
            # target can never be inherited.
            self._clear_set()
            self._write_targets(start, holding=True)
            for arm in ('A', 'B'):
                self._check(self.robot.set_vel_acc(
                    arm, self.settings['vel_ratio'], self.settings['acc_ratio']),
                    f'{arm} 官方速度比例')
                self._check(self.robot.set_state(arm, 3), f'{arm} 进入关节阻抗')
            self._check(self.robot.send_cmd(), '使能双臂关节阻抗')
            self._motion_started = True
            deadline = self.clock() + 2.
            while self.clock() < deadline:
                feedback = self.read_feedback()
                if feedback['states'] == [3, 3]:
                    break
                time.sleep(.02)
            else:
                raise HardwareError('双臂未在 2 秒内进入关节阻抗模式')
            self.enabled = True

            self._clear_set()
            for arm in ('A', 'B'):
                self._check(self.robot.set_joint_kd_params(
                    arm, self.settings['joint_k'], self.settings['joint_d']),
                    f'{arm} 官方关节阻抗 K/D')
            self._check(self.robot.send_cmd(), '发送官方关节阻抗 K/D')
            time.sleep(.5)
            self._clear_set()
            for arm in ('A', 'B'):
                self._check(self.robot.set_impedance_type(arm, 1), f'{arm} 关节阻抗类型')
            self._check(self.robot.send_cmd(), '发送关节阻抗类型')
            time.sleep(.5)

            started = self.clock()
            for index in range(steps + 1):
                t = index / steps
                blend = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
                target = {side: start[side] + blend * (goal[side] - start[side])
                          for side in SIDES}
                self._clear_set()
                self._write_targets(target, holding=True)
                self._check(self.robot.send_cmd(), '发送双臂初始姿态轨迹')
                self.read_feedback()
                wait = started + index * duration / steps - self.clock()
                if wait > 0:
                    time.sleep(wait)

            settle_deadline = self.clock() + float(self.settings['startup_settle_s'])
            while self.clock() < settle_deadline:
                self.read_feedback()
                time.sleep(min(.02, max(0., settle_deadline - self.clock())))
            feedback = self.read_feedback()
            errors = {side: float(np.max(np.abs(feedback['joints'][side] - goal[side])))
                      for side in SIDES}
            tolerance = float(self.settings['startup_tolerance_deg'])
            if max(errors.values()) >= tolerance:
                raise HardwareError(f'官方初始姿态未到位：'
                                    f'左臂 {errors["left"]:.2f}°，右臂 {errors["right"]:.2f}°')

            self._hold_targets = {s: q.copy() for s, q in goal.items()}
            return feedback['joints']
        except BaseException as exc:
            self.stop(str(exc))
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise HardwareError(f'双臂自动初始姿态失败: {exc}') from exc

    def _set_standby(self, wait=False):
        self._clear_set()
        for arm, side in zip(('A', 'B'), SIDES):
            if side not in self.control_sides:
                continue
            self._check(self.robot.set_state(arm, 0), f'{arm} 进入待机')
        self._check(self.robot.send_cmd(), '发送待机命令')
        if wait:
            deadline = self.clock() + 3.
            while self.clock() < deadline:
                if self.read_feedback()['states'] == [0, 0]:
                    return
                time.sleep(.02)
            raise HardwareError('天机未在 3 秒内进入待机')

    def _write_targets(self, joints, *, holding=False):
        # Validate BOTH sides before putting anything in the outbound SDK buffer.
        values = {s: np.asarray(joints[s], dtype=float) for s in SIDES}
        if any(q.shape != (7,) or not np.isfinite(q).all() for q in values.values()):
            raise HardwareError('双臂关节目标必须是有限 7 轴角度')
        if self._wrist is not None:
            self._wrist.check_commands(values)
            if self._last_targets is not None and not holding:
                dt = min(.05, max(0., self.clock() - self._last_target_time))
                maximum = self.settings['wrist_max_speed_deg_s'] * dt
                for side in SIDES:
                    delta = values[side] - self._last_targets[side]
                    size = float(np.max(np.abs(delta)))
                    if size > maximum:
                        values[side] = self._last_targets[side] + delta * maximum / size
        elif (self.settings.get('servo_mode') == 'position' and
              self._last_targets is not None and not holding):
            dt = min(.05, max(0., self.clock() - self._last_target_time))
            maximum = self.config['max_joint_speed_deg_s'] * dt
            for side in self.control_sides:
                delta = values[side] - self._last_targets[side]
                size = float(np.max(np.abs(delta)))
                if size > maximum:
                    values[side] = self._last_targets[side] + delta * maximum / size
        if self._startup_joints is not None and 'joint_excursion_deg' in self.settings:
            excursion = self.settings['joint_excursion_deg']
            for side in self.control_sides:
                if np.max(np.abs(values[side] - self._startup_joints[side])) > excursion + 1e-9:
                    raise HardwareError(f'{side}: 指令超出启动角 ±{excursion}° 安全包络')
        for arm, side in zip(('A', 'B'), SIDES):
            if side not in self.control_sides:
                continue
            self._check(self.robot.set_joint_cmd_pose(arm, values[side].tolist()), f'{side} 关节目标')
        self._last_targets = {s: q.copy() for s, q in values.items()}
        self._last_target_time = self.clock()

    def enable(self):
        """Begin nonblocking servo preparation; poll() completes the transition."""
        if self.fault:
            raise HardwareError(self.fault)
        if not self.connected:
            raise HardwareError('天机尚未连接')
        if self.enabled or self.enabling:
            return self.enabled
        vel, acc = self.settings.get('vel_ratio', 20), self.settings.get('acc_ratio', 20)
        if any(type(v) is not int or not 1 <= v <= 100 for v in (vel, acc)):
            raise HardwareError('速度 / 加速度比例须为 1–100 的整数')
        try:
            feedback = self.read_feedback()
            self._configure_tool(feedback)
            if self._tool_ready_at is not None and self.clock() < self._tool_ready_at:
                self.phase, self.deadline = 'tool_parameters', self._tool_ready_at
                return False
            return self._begin_enable(feedback, vel, acc)
        except Exception as exc:
            self.stop(str(exc))
            raise HardwareError(f'天机使能失败: {exc}') from exc

    def _begin_enable(self, feedback, vel, acc):
        if feedback['states'] != [0, 0]:
            raise HardwareError('使能前双臂必须在待机状态')
        if self.settings.get('impedance_params_before_enable', False):
            # Match Wuji's set_impedance_mode(): load gains and mode while
            # both arms are still in standby, then allow the SDK packet to
            # settle before requesting state=3.  No robot target is sent yet.
            self._clear_set()
            for arm in ('A', 'B'):
                self._check(self.robot.set_joint_kd_params(
                    arm, self.settings['joint_k'], self.settings['joint_d']),
                    f'{arm} 关节阻抗参数')
                self._check(self.robot.set_impedance_type(arm, 1),
                            f'{arm} 关节阻抗类型')
            self._check(self.robot.send_cmd(), '发送待机状态关节阻抗参数与类型')
            self.phase, self.deadline = 'impedance_parameters', self.clock() + .5
            return False
        self._clear_set()
        # Seed the current measured pose in the same packet as servo-on;
        # never inherit an old client's joint target.
        self._write_targets(self.held_joints(feedback['joints']), holding=True)
        for arm, side in zip(('A', 'B'), SIDES):
            if side not in self.control_sides:
                continue
            self._check(self.robot.set_vel_acc(arm, vel, acc), f'{arm} 速度比例')
        self._motion_started = True
        if self.servo_state == 1:
            self._check(self.robot.send_cmd(), '发送位置模式限速与当前目标')
            self.phase, self.deadline = 'position_parameters', self.clock() + .1
            return False
        for arm, side in zip(('A', 'B'), SIDES):
            if side not in self.control_sides:
                continue
            self._check(self.robot.set_state(arm, 3), f'{arm} 进入阻抗状态')
        self._check(self.robot.send_cmd(), '发送使能命令')
        self.phase, self.deadline = 'state', self.clock() + 3.
        self._phase_serials = feedback['serials'].copy()
        return False

    def poll(self):
        if self.fault:
            raise HardwareError(self.fault)
        try:
            feedback = self.read_feedback()
            now = self.clock()
            if self.phase == 'tool_parameters' and now >= self.deadline:
                vel, acc = self.settings.get('vel_ratio', 20), self.settings.get('acc_ratio', 20)
                self._begin_enable(feedback, vel, acc)
            elif self.phase == 'impedance_parameters' and now >= self.deadline:
                if feedback['states'] != [0, 0]:
                    raise HardwareError('准备关节阻抗时双臂不在待机状态')
                self._clear_set()
                # The input pose is sampled again after the 0.5 s settling
                # interval, so the enable packet cannot inherit a stale goal.
                self._write_targets(self.held_joints(feedback['joints']), holding=True)
                vel, acc = self.settings['vel_ratio'], self.settings['acc_ratio']
                for arm in ('A', 'B'):
                    self._check(self.robot.set_state(arm, 3), f'{arm} 进入阻抗状态')
                    self._check(self.robot.set_vel_acc(arm, vel, acc), f'{arm} 速度比例')
                self._motion_started = True
                self._check(self.robot.send_cmd(), '发送关节阻抗使能与当前目标')
                self.phase, self.deadline = 'state_after_params', now + 5.
                self._phase_serials = feedback['serials'].copy()
            elif self.phase == 'state_after_params':
                if now > self.deadline:
                    raise HardwareError('天机未在 5 秒内进入关节阻抗状态')
                if feedback['states'] == [3, 3] and all(
                        a != b for a, b in zip(feedback['serials'], self._phase_serials)):
                    self.enabled, self.phase = True, None
            elif self.phase == 'position_parameters' and now >= self.deadline:
                if feedback['states'] != [0, 0]:
                    raise HardwareError('准备位置模式时双臂不在待机状态')
                self._clear_set()
                self._write_targets(self.held_joints(feedback['joints']), holding=True)
                for arm, side in zip(('A', 'B'), SIDES):
                    if side not in self.control_sides:
                        continue
                    self._check(self.robot.set_state(arm, 1), f'{arm} 进入位置跟随')
                self._check(self.robot.send_cmd(), '发送位置模式使能')
                self.phase, self.deadline = 'position_wait', now + 3.
                self._position_ready_after = now + 1.
                self._phase_serials = feedback['serials'].copy()
            elif self.phase == 'position_wait':
                if now > self.deadline:
                    raise HardwareError('天机未在 3 秒内进入位置模式')
                expected = [self.servo_state if side in self.control_sides else 0 for side in SIDES]
                if now >= self._position_ready_after and feedback['states'] == expected and all(
                        a != b for a, b in zip(feedback['serials'], self._phase_serials)):
                    self.enabled, self.phase = True, None
            elif self.phase == 'state':
                if now > self.deadline:
                    raise HardwareError('天机未在 3 秒内进入阻抗状态')
                expected = [self.servo_state if side in self.control_sides else 0 for side in SIDES]
                if feedback['states'] == expected and all(a != b for a, b in
                        zip(feedback['serials'], self._phase_serials)):
                    self._clear_set()
                    for arm, side in zip(('A', 'B'), SIDES):
                        if side not in self.control_sides:
                            continue
                        self._check(self.robot.set_joint_kd_params(arm,
                            self.settings.get('joint_k', [2, 2, 2, 1.6, 1, 1, 1]),
                            self.settings.get('joint_d', [.3, .3, .3, .2, .2, .2, .2])), f'{arm} 关节阻抗参数')
                    self._check(self.robot.send_cmd(), '发送阻抗参数')
                    self.phase, self.deadline = 'gains', now + .5
            elif self.phase in ('gains', 'settle'):
                expected = [self.servo_state if side in self.control_sides else 0 for side in SIDES]
                if feedback['states'] != expected:
                    raise HardwareError('设置阻抗期间伺服状态异常')
                if now >= self.deadline:
                    if self.phase == 'gains':
                        self._clear_set()
                        # At mode switch, use the latest measured pose again.
                        self._write_targets(feedback['joints'])
                        for arm, side in zip(('A', 'B'), SIDES):
                            if side not in self.control_sides:
                                continue
                            self._check(self.robot.set_impedance_type(arm, 1), f'{arm} 关节阻抗类型')
                        self._check(self.robot.send_cmd(), '发送阻抗类型')
                        self.phase, self.deadline = 'settle', now + .5
                    else:
                        self.enabled, self.phase = True, None
            return feedback
        except Exception as exc:
            self.stop(str(exc))
            raise HardwareError(str(exc)) from exc

    def send_joint_targets(self, joints):
        if self.fault or not self.enabled:
            raise HardwareError(self.fault or '伺服尚未就绪，拒绝跟随指令')
        try:
            self.read_feedback()
            self._clear_set()
            self._write_targets(joints)
            self._check(self.robot.send_cmd(), '发送双臂关节目标')
            # The next pause captures the achieved pose once. Never keep
            # driving toward an old teleoperation target after clutch loss.
            self._hold_targets = None
        except Exception as exc:
            self.stop(str(exc))
            raise HardwareError(f'发送关节目标失败: {exc}') from exc

    def hold(self, _joints=None):
        if self.enabling:
            self.cancel_enable()
        elif self.enabled:
            try:
                # Read feedback on every hold for telemetry/fault validation.
                # Only the first pause after motion selects a new target.
                measured = self.read_joints()
                joints = (self.held_joints(measured) if self._hold_targets is None else
                          {s: q.copy() for s, q in self._hold_targets.items()})
                self._clear_set()
                self._write_targets(joints, holding=True)
                self._check(self.robot.send_cmd(), '保持锁存的关节目标')
                self._hold_targets = {s: q.copy() for s, q in self._last_targets.items()}
                return self.hold_targets
            except Exception as exc:
                self.stop(str(exc))
                raise HardwareError(str(exc)) from exc

    def cancel_enable(self):
        if self.enabling:
            self.phase = None
            self.enabled = False
            try:
                self._set_standby()
            except Exception as exc:
                self.stop(str(exc))
                raise

    def stop(self, reason):
        """Latch faults and make one best-effort standby request (not an E-stop)."""
        already_faulted = self.fault is not None
        self.fault = self.fault or reason
        self.enabled, self.phase = False, None
        self._hold_targets = None
        if self.connected and self._motion_started and not already_faulted:
            try:
                self._set_standby()
            except Exception as exc:
                self.logger.error('故障后待机请求失败；需要现场急停: %s', exc)

    def close(self):
        """Release the SDK connection and report whether both arms reached standby.

        A best-effort standby request is not proof of state=0. Callers that own
        real hardware must inspect this result before reporting a clean stop.
        """
        if not self.connected:
            return self._standby_confirmed
        self.enabled, self.phase = False, None
        self._hold_targets = None
        standby_confirmed = False
        try:
            if self._motion_started:
                try:
                    self._set_standby(wait=True)
                    standby_confirmed = True
                except Exception as exc:
                    self.logger.warning('待机确认失败: %s', exc)
            else:
                # The connection gate established state=0. Verify it is still
                # true without issuing a mode command we do not need.
                try:
                    standby_confirmed = self.read_feedback()['states'] == [0, 0]
                except Exception as exc:
                    self.logger.warning('待机复核失败: %s', exc)
        finally:
            if self.connected:
                try:
                    self.robot.release_robot()
                except Exception as exc:
                    self.logger.warning('释放连接失败: %s', exc)
            self.connected = False
            self._tool_configured = False
            self._tool_ready_at = None
            self._standby_confirmed = bool(standby_confirmed)
        return self._standby_confirmed
