"""Cross-platform terminal launcher and keyboard controls for Quest input."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import select
import sys
import socket
import time
import uuid

if sys.platform != 'win32':
    import termios
    import tty
else:
    termios = tty = None

from aiohttp import ClientSession, ClientTimeout, ClientConnectorError, WSMsgType, web

from .config import APP_ROOT, configure_runtime, load_config
from .server import create_app


KEY_ACTIONS = {0x41: 'calibrate', 0x42: 'reset', 0x20: 'keyboard_toggle',
               0x51: 'quit', 0x1B: 'quit'}


class KeyEdges:
    """Use key-up events, so Windows auto-repeat never toggles Space twice."""
    def __init__(self):
        self.held = set()

    def update(self, virtual_key, down):
        if not down:
            self.held.discard(virtual_key)
            return None
        if virtual_key in self.held:
            return None
        self.held.add(virtual_key)
        return KEY_ACTIONS.get(virtual_key)


class WindowsConsole:
    def __init__(self):
        if sys.platform != 'win32' or not sys.stdin.isatty():
            raise RuntimeError('请在交互式 PowerShell / Windows Terminal 中运行 teleop.ps1')

        class KeyEvent(ctypes.Structure):
            _fields_ = [('down', wintypes.BOOL), ('repeat', wintypes.WORD),
                        ('key', wintypes.WORD), ('scan', wintypes.WORD),
                        ('char', wintypes.WCHAR), ('modifiers', wintypes.DWORD)]

        class EventData(ctypes.Union):
            _fields_ = [('key', KeyEvent), ('focus', wintypes.BOOL), ('padding', ctypes.c_byte * 16)]

        class InputRecord(ctypes.Structure):
            _fields_ = [('kind', wintypes.WORD), ('event', EventData)]

        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.GetStdHandle.argtypes = [wintypes.DWORD]
        self.api.GetStdHandle.restype = wintypes.HANDLE
        self.api.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.api.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.api.GetNumberOfConsoleInputEvents.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.api.ReadConsoleInputW.argtypes = [wintypes.HANDLE, ctypes.POINTER(InputRecord),
                                             wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        self.handle = self.api.GetStdHandle(-10 & 0xffffffff)
        self.mode = wintypes.DWORD()
        if not self.api.GetConsoleMode(self.handle, ctypes.byref(self.mode)):
            raise RuntimeError('当前终端不支持控制台按键事件；请使用独立 PowerShell 窗口')
        self.records = (InputRecord * 64)()
        self.edges = KeyEdges()

    def __enter__(self):
        # Disable Quick Edit, line buffering and echo; retain Ctrl+C handling.
        if not self.api.SetConsoleMode(self.handle, (self.mode.value | 0x80) & ~(0x40 | 0x2 | 0x4)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            self.poll()  # Discard input typed before the controls became ready.
        except BaseException:
            self.api.SetConsoleMode(self.handle, self.mode.value)
            raise
        self.edges.held.clear()
        return self

    def __exit__(self, *_):
        self.api.SetConsoleMode(self.handle, self.mode.value)

    def poll(self):
        count, read = wintypes.DWORD(), wintypes.DWORD()
        if not self.api.GetNumberOfConsoleInputEvents(self.handle, ctypes.byref(count)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not count.value:
            return []
        if not self.api.ReadConsoleInputW(self.handle, self.records, min(count.value, 64), ctypes.byref(read)):
            raise ctypes.WinError(ctypes.get_last_error())
        actions = []
        for record in self.records[:read.value]:
            if record.kind == 1:
                action = self.edges.update(record.event.key.key, bool(record.event.key.down))
                if action:
                    actions.append(action)
            elif record.kind == 16 and not record.event.focus:
                self.edges.held.clear()
        return actions


class UnixConsole:
    """Read one-key commands from a Linux TTY without requiring Enter."""
    REPEAT_GUARD_S = .25

    def __init__(self):
        if sys.platform == 'win32' or not sys.stdin.isatty():
            raise RuntimeError('请在交互式 Ubuntu 终端中运行 teleop.sh')
        self.fd = sys.stdin.fileno()
        self.original = None
        self.last_key = None
        self.last_key_at = float('-inf')

    def __enter__(self):
        self.original = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        current = termios.tcgetattr(self.fd)
        current[3] &= ~termios.ECHO
        termios.tcsetattr(self.fd, termios.TCSANOW, current)
        return self

    def __exit__(self, *_):
        if self.original is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.original)
            self.original = None

    def poll(self):
        if not select.select([self.fd], [], [], 0)[0]:
            return []
        raw = os.read(self.fd, 64).decode('utf-8', errors='ignore')
        actions = []
        now = time.monotonic()
        key_actions = {'a': 'calibrate', 'A': 'calibrate', 'b': 'reset', 'B': 'reset',
                       ' ': 'keyboard_toggle',
                       'q': 'quit', 'Q': 'quit', '\x1b': 'quit'}
        for key in raw:
            action = key_actions.get(key)
            if action is None:
                continue
            # Linux terminal input has no key-up event. Ignore the terminal's
            # auto-repeat while still allowing a new press after a release.
            if key == self.last_key and now - self.last_key_at < self.REPEAT_GUARD_S:
                continue
            self.last_key, self.last_key_at = key, now
            actions.append(action)
        return actions


def quest_tracking_ready(state, *, native_arm_mode=False):
    """A fresh pair of poses; native mode has only numeric, not optical, validity."""
    arms = state.get('arms') or {}
    sides = state.get('control_sides') or ('left', 'right')
    age = state.get('frame_age_ms')
    hardware_ready = (state.get('hardware_connected') and state.get('hardware_enabled')
                      if not native_arm_mode or state.get('hardware') else True)
    native_ready = (state.get('input_backend') == 'xrobotoolkit' and state.get('native_eligible')
                    if native_arm_mode else True)
    return bool(state.get('source') == 'quest' and state.get('keyboard_mode') and
                state.get('fresh') and hardware_ready and native_ready and
                not state.get('hardware_fault') and
                not state.get('tracking_loss_active') and
                not state.get('tracking_recovery_pending') and
                not state.get('input_focus_lost') and
                isinstance(age, (int, float)) and 0 <= age < 150 and
                all(arms.get(side, {}).get('numeric_pose_gate' if native_arm_mode else 'tracked')
                    for side in sides))


class StableQuestTracking:
    """Count distinct healthy frames within one unchanged XR reference space."""
    STABLE_S = .6
    MIN_FRAMES = 8
    MAX_FRAME_GAP_S = .15

    def __init__(self, stable_s=STABLE_S, min_frames=MIN_FRAMES, *, native_arm_mode=False):
        self.stable_s = stable_s
        self.min_frames = min_frames
        self.native_arm_mode = native_arm_mode
        self.reset()

    def reset(self):
        self.context = None
        self.first_s = self.last_s = None
        self.last_seq = None
        self.frames = 0

    def observe(self, state, now):
        session, reference, seq = (state.get('input_session'), state.get('input_reference'),
                                   state.get('seq'))
        if (not quest_tracking_ready(state, native_arm_mode=self.native_arm_mode) or not isinstance(session, str) or
                type(reference) is not int or type(seq) is not int):
            self.reset()
            return False
        context = (session, reference)
        if (self.context != context or self.last_seq is None or seq < self.last_seq or
                now - self.last_s > self.MAX_FRAME_GAP_S):
            self.reset()
            self.context, self.first_s, self.last_s, self.last_seq, self.frames = context, now, now, seq, 1
        elif seq > self.last_seq:
            self.last_s, self.last_seq = now, seq
            self.frames += 1
        return self.frames >= self.min_frames and now - self.first_s >= self.stable_s


def joint_gate_ready(state, *, native_arm_mode=False):
    """Only follow the glove while both measured arm inputs can be followed."""
    arms = state.get('arms') or {}
    sides = state.get('control_sides') or ('left', 'right')
    if native_arm_mode:
        # The official APK does not report optical validity. Use the bridge's
        # numeric gate, freshness and focus, while retaining all arm guards.
        input_ready = bool(state.get('hardware') is True and
                           quest_tracking_ready(state, native_arm_mode=True))
    else:
        input_ready = bool(state.get('fresh') and
                           all(arms.get(side, {}).get('tracked') for side in sides))
    return bool(state.get('keyboard_mode') and state.get('keyboard_follow') and
                state.get('enabled') and state.get('following') and input_ready and
                state.get('hardware_connected') and state.get('hardware_enabled') and
                not state.get('hardware_fault') and
                not state.get('tracking_loss_active') and
                not state.get('tracking_recovery_pending'))


class KeyboardClient:
    def __init__(self, ws, report=print, hardware=False, auto_calibrate=False,
                 auto_follow_on_ready=False, joint_gate=None, native_arm_mode=False,
                 clock=time.monotonic):
        self.ws, self.report = ws, report
        self.hardware = hardware
        self.auto_calibrate = auto_calibrate
        self.auto_follow_on_ready = auto_follow_on_ready
        self.joint_gate = Path(joint_gate) if joint_gate else None
        self.native_arm_mode = native_arm_mode
        self.require_space_start = bool(hardware and native_arm_mode)
        self.initial_prepared = False
        self.first_started = False
        self.clock = clock
        self.command_timeout = 8. if hardware else .8
        self.pending = {}
        self.last_status = None
        self.latest_state = None
        self.latest_state_at = None
        self.state_updates = asyncio.Queue(maxsize=64)
        self.receiver = asyncio.create_task(self.receive())

    def set_joint_gate(self, enabled):
        if self.joint_gate is None:
            return
        if enabled:
            self.joint_gate.touch(exist_ok=True)
        else:
            self.joint_gate.unlink(missing_ok=True)

    async def receive(self):
        try:
            async for message in self.ws:
                if message.type != WSMsgType.TEXT:
                    continue
                data = message.json()
                if data.get('type') == 'command_ack':
                    future = self.pending.get(data.get('id'))
                    if future is not None and not future.done():
                        future.set_result(data)
                elif data.get('type') == 'error':
                    self.report(data.get('message', '服务错误'))
                elif data.get('type') == 'state':
                    self.latest_state = data
                    self.latest_state_at = time.monotonic()
                    if self.state_updates.full():
                        self.state_updates.get_nowait()
                    self.state_updates.put_nowait(data)
                    gate_open = joint_gate_ready(data, native_arm_mode=self.native_arm_mode)
                    if self.require_space_start and not self.first_started:
                        gate_open = False
                    native_joint_gate = self.native_arm_mode and self.joint_gate is not None
                    if native_joint_gate:
                        # The hand-side watchdog watches mtime, so refresh only
                        # while every current broadcast still permits following.
                        self.set_joint_gate(gate_open)
                    native_status = ((data.get('native_eligible'), data.get('native_reason'),
                                      data.get('input_focus_lost'),
                                      tuple((data.get('arms') or {}).get(side, {}).get('numeric_pose_gate')
                                            for side in ('left', 'right')))
                                     if self.native_arm_mode else ())
                    status = tuple(data.get(k) for k in ('enabled', 'resetting', 'fresh', 'keyboard_mode',
                                                         'keyboard_follow', 'following', 'tracking_loss_active',
                                                         'tracking_recovery_pending')) + (gate_open,) + native_status
                    if status != self.last_status:
                        self.last_status = status
                        if not native_joint_gate:
                            self.set_joint_gate(gate_open)
                        if data.get('tracking_loss_active') or data.get('tracking_recovery_pending'):
                            self.report(data['reason'])
                        elif native_joint_gate and data.get('keyboard_mode') and data.get('enabled'):
                            self.report('原生双臂与灵巧手联合跟随中；空格暂停' if gate_open else
                                        '等待原生数值门禁与双臂跟随；灵巧手保持' if data.get('keyboard_follow') else
                                        '已校准；按空格开始原生联合跟随')
                        elif self.native_arm_mode and data.get('keyboard_mode') and data.get('enabled'):
                            self.report('原生双臂跟随中；空格暂停' if data.get('keyboard_follow') and data.get('following') else
                                        '原生位姿暂不可用；双臂保持，等待数值门禁恢复' if data.get('keyboard_follow') else
                                        '已校准；按空格开始双臂跟随')
                        elif data.get('keyboard_mode') and data.get('enabled'):
                            self.report('双臂与灵巧手跟随中；空格暂停' if gate_open else
                                        '等待双臂与手柄就绪；灵巧手保持，准备就绪后自动跟随'
                                        if data.get('keyboard_follow') else
                                        '已校准；按空格开始跟随')
                        else:
                            self.report(data['reason'])
        finally:
            self.set_joint_gate(False)
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(ConnectionError('遥操连接已断开'))

    async def command(self, action):
        if self.receiver.done() or self.ws.closed:
            raise ConnectionError('遥操连接已断开')
        command_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[command_id] = future
        try:
            await self.ws.send_json({'type': 'command', 'id': command_id, 'action': action})
            result = await asyncio.wait_for(future, self.command_timeout)
            if not result.get('ok'):
                raise ValueError(result.get('error', '操作未执行'))
            return result
        finally:
            self.pending.pop(command_id, None)

    async def heartbeat(self):
        while True:
            await asyncio.sleep(.2)
            await self.command('keyboard_heartbeat')

    async def calibrate_when_ready(self, timeout_s=180.):
        """Calibrate once when a fresh Quest frame arrives; never continuously."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while deadline is None or time.monotonic() < deadline:
            if self.require_space_start and (self.latest_state or {}).get('hardware_fault'):
                raise RuntimeError(f"实机故障已锁存：{self.latest_state['hardware_fault']}")
            try:
                await self.command('calibrate')
                if self.require_space_start:
                    self.initial_prepared = True
                self.report('原生 Quest 位姿已自动校准；按空格开始双臂遥操' if self.native_arm_mode else
                            'Quest 输入已自动校准；按空格开始遥操')
                return
            except ValueError as exc:
                if self.require_space_start and ('实机故障' in str(exc) or
                                                 (self.latest_state or {}).get('hardware_fault')):
                    raise
                await asyncio.sleep(.1)
        raise RuntimeError(f'等待 Quest 新鲜双手柄位姿超过 {timeout_s:g} 秒；请检查 XRoboToolkit 原生采集状态'
                           if self.native_arm_mode else
                           f'等待 Quest 新鲜双手柄追踪超过 {timeout_s:g} 秒；请检查电脑侧 Quest 采集与双手柄真实 6DoF 状态')

    async def next_state(self, deadline, heartbeat):
        while time.monotonic() < deadline:
            if heartbeat.done():
                await heartbeat
            if self.receiver.done():
                await self.receiver
                raise ConnectionError('Quest 状态连接已断开')
            try:
                return await asyncio.wait_for(self.state_updates.get(),
                                              min(.2, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                pass
        raise RuntimeError('等待 Quest 稳定追踪或校准确认超时')

    async def auto_follow_once(self, heartbeat, timeout_s=180.):
        # Ignore state buffered before keyboard ownership was attached.
        while not self.state_updates.empty():
            self.state_updates.get_nowait()
        # The neck-mounted camera sees a narrower useful hand workspace.
        # Require a longer uninterrupted pair of physical 6DoF poses before
        # the first joint follow; later tracking recovery has its own gate.
        window = (StableQuestTracking(stable_s=2.0, min_frames=30,
                                      native_arm_mode=self.native_arm_mode)
                  if self.joint_gate or self.native_arm_mode else StableQuestTracking())
        deadline = time.monotonic() + timeout_s
        while True:
            state = await self.next_state(deadline, heartbeat)
            if window.observe(state, self.clock()):
                break
        context = (state['input_session'], state['input_reference'])
        revision = state['calibration_revision']
        await self.command('calibrate')
        # Confirm that calibration belongs to the same XR tracking space. A
        # recenter or hard pause between readiness and A must not auto-start.
        confirm_deadline = time.monotonic() + 2.
        while True:
            confirmed = await self.next_state(confirm_deadline, heartbeat)
            if (confirmed.get('input_session'), confirmed.get('input_reference')) != context:
                raise RuntimeError('Quest 追踪空间在自动校准期间变化；已取消自动开始')
            if not quest_tracking_ready(confirmed, native_arm_mode=self.native_arm_mode):
                raise RuntimeError('Quest 追踪在自动校准期间中断；已取消自动开始')
            if confirmed.get('calibration_revision', 0) > revision:
                if not confirmed.get('enabled'):
                    raise RuntimeError('自动校准后机械臂已暂停；已取消自动开始')
                break
        result = await self.command('keyboard_toggle')
        if not result.get('keyboard_follow'):
            raise RuntimeError('自动开始未进入跟随状态')
        self.report(('原生 Quest 双手柄数值位姿稳定，已自动校准并开始双臂与灵巧手联合跟随；空格可暂停 / 恢复'
                     if self.joint_gate else
                     '原生 Quest 双手柄数值位姿稳定，已自动校准并开始双臂跟随；空格可暂停 / 恢复')
                    if self.native_arm_mode else
                    'Quest 双手柄稳定追踪，已自动校准并开始联合跟随；空格可暂停 / 恢复')

    async def manual_calibrate(self):
        """A is also the resume key in joint mode; each command checks fresh input."""
        self.set_joint_gate(False)
        first_native_start = self.require_space_start and not self.first_started
        if first_native_start:
            self.initial_prepared = False
        if self.native_arm_mode:
            # A may be pressed while the keyboard clutch is engaged. Stop and
            # hold from measured joints before acquiring a new pose anchor.
            # A failed calibration leaves both arms paused.
            await self.command('pause')
        await self.command('calibrate')
        if first_native_start:
            self.initial_prepared = True
            self.report('A 已校准原生双臂位姿；保持当前位置，按空格首次开始跟随')
            return
        if self.joint_gate is None and not self.native_arm_mode:
            self.report('A 校准完成；按空格开始遥操')
            return
        result = await self.command('keyboard_toggle')
        if not result.get('keyboard_follow'):
            raise ValueError('校准后未进入跟随；请确认双手柄有效，再按空格继续')
        self.report(('A 已重新校准并自动继续原生双臂与灵巧手联合跟随' if self.joint_gate else
                     'A 已重新校准并自动继续原生双臂跟随') if self.native_arm_mode else
                    'A 已重新校准并自动继续联合跟随')

    def first_space_ready(self):
        """A rejected first Space must never become a deferred follow request."""
        state = self.latest_state
        return bool(self.initial_prepared and state is not None and
                    self.latest_state_at is not None and
                    0 <= time.monotonic() - self.latest_state_at <= .3 and
                    state.get('hardware') is True and state.get('enabled') and
                    not state.get('resetting') and not state.get('keyboard_follow') and
                    quest_tracking_ready(state, native_arm_mode=True))

    async def run(self, console):
        heartbeat = None
        initial_calibration_task = None
        try:
            await self.command('keyboard_attach')
            self.set_joint_gate(False)
            heartbeat = asyncio.create_task(self.heartbeat())
            if self.require_space_start:
                if self.auto_calibrate or self.auto_follow_on_ready:
                    # Keep Q and Space responsive while waiting for native input.
                    # Even an accidental auto-follow flag only prepares the pose.
                    initial_calibration_task = asyncio.create_task(
                        self.calibrate_when_ready(timeout_s=None))
            elif self.auto_follow_on_ready:
                await self.auto_follow_once(heartbeat)
            elif self.auto_calibrate:
                await self.calibrate_when_ready()
            self.report('键盘已就绪（无需回车）：' +
                        ('A 校准，首次仍须空格开始' if self.require_space_start else
                         'A 校准并继续' if self.joint_gate or self.native_arm_mode else 'A 校准') + ' | ' +
                        ('B 暂停双臂' if self.hardware or self.native_arm_mode else 'B 回初始') + ' | 空格 开始/停止 | Q / Esc 退出')
            self.report('请将键盘焦点放在本终端；原生采集由 XRoboToolkit Quest 客户端和 PC Service 提供。中断后等待位姿恢复；首次开始前按 A 仅校准，空格开始。'
                        if self.native_arm_mode else
                        '请将键盘焦点放在本终端；Quest 采集由电脑启动。若会话结束，可在另一终端运行 ./tianji_quest.sh capture，再按 A' +
                        (' 自动继续。' if self.joint_gate else '、空格。'))
            while True:
                if heartbeat.done():
                    await heartbeat
                if self.receiver.done():
                    await self.receiver
                    raise ConnectionError('遥操服务已断开')
                actions = console.poll()
                if 'quit' in actions:
                    break
                if initial_calibration_task is not None and initial_calibration_task.done():
                    await initial_calibration_task
                    initial_calibration_task = None
                # B wins if several keys arrive together, as on the controller.
                if 'reset' in actions:
                    actions = ['reset']
                for action in actions:
                    if (self.hardware or self.native_arm_mode) and action == 'reset':
                        action = 'pause'
                    if action in ('keyboard_toggle', 'pause'):
                        # ACK is not proof that a valid optical pose still exists.
                        # Only a later healthy state may open the glove gate.
                        self.set_joint_gate(False)
                    try:
                        if action == 'calibrate':
                            if initial_calibration_task is not None:
                                initial_calibration_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await initial_calibration_task
                                initial_calibration_task = None
                            try:
                                await self.manual_calibrate()
                            except ValueError:
                                if self.require_space_start and not self.first_started and self.auto_calibrate:
                                    initial_calibration_task = asyncio.create_task(
                                        self.calibrate_when_ready(timeout_s=None))
                                raise
                            continue
                        if (action == 'keyboard_toggle' and self.require_space_start and
                                not self.first_started and not self.first_space_ready()):
                            self.report('原生输入或校准尚未就绪；保持当前位置。就绪后请再次按空格开始')
                            continue
                        result = await self.command(action)
                        if action == 'keyboard_toggle':
                            if self.require_space_start and result.get('keyboard_follow'):
                                self.first_started = True
                            self.report('遥操已开始：键盘控制左右双臂' if result['keyboard_follow'] else
                                        '遥操已停止：保持当前位置，再按空格继续')
                        elif action == 'reset':
                            self.report('B 已执行；恢复完成后按 A 重新校准')
                    except ValueError as exc:
                        self.report(str(exc))
                await asyncio.sleep(.01)
        finally:
            if initial_calibration_task is not None:
                initial_calibration_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await initial_calibration_task
            if heartbeat:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await heartbeat
            with contextlib.suppress(Exception):
                await self.command('keyboard_detach')
            self.set_joint_gate(False)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.ws.close(), 1.)
            self.receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.receiver


async def run_usb(port, adb, serial):
    if sys.platform == 'win32':
        command = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                   '-File', str(APP_ROOT / 'usb.ps1'), '-Port', str(port), '-Adb', adb]
    else:
        command = [str(APP_ROOT / 'usb.sh'), '--port', str(port), '--adb', adb]
    if serial:
        command += ['-Serial' if sys.platform == 'win32' else '--serial', serial]
    process = await asyncio.create_subprocess_exec(*command, cwd=str(APP_ROOT))
    try:
        if await process.wait():
            raise RuntimeError('USB 配置失败；请根据上方提示处理后重新运行 teleop.ps1')
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()


async def run(args):
    console = WindowsConsole() if sys.platform == 'win32' else UnixConsole()
    native_arm_mode = bool(getattr(args, 'native_arm_mode', False))
    full_arm = bool(getattr(args, 'full_arm', False))
    auto_calibrate = bool(getattr(args, 'auto_calibrate', False))
    auto_follow_on_ready = bool(getattr(args, 'auto_follow_on_ready', False))
    joint_gate = getattr(args, 'joint_gate', '')
    profile = args.config or (APP_ROOT / 'configs/hardware_dual_arm.json' if full_arm else None)
    config = configure_runtime(load_config(profile, hardware=args.hardware), hardware=args.hardware,
                               robot_ip=args.robot_ip, vel_ratio=args.vel_ratio,
                               acc_ratio=args.acc_ratio)
    expected_scope = 'full' if full_arm else 'wrist'
    if args.hardware and config['hardware']['motion_scope'] != expected_scope:
        raise RuntimeError(('键盘入口的实机整臂控制需要 --full-arm 与整臂配置' if full_arm else
                            '键盘入口默认仅限手腕；整臂请使用 --full-arm'))
    url = f'http://127.0.0.1:{args.port}'
    runner = None
    try:
        async with ClientSession() as session:
            try:
                async with session.get(f'{url}/api/health', timeout=ClientTimeout(total=2)) as response:
                    health = await response.json()
            except (ClientConnectorError, asyncio.TimeoutError):
                if native_arm_mode:
                    raise RuntimeError(f'原生双臂服务未在 {url} 就绪；请通过原生启动器启动')
                # Windows may time out before reporting connection refusal.
                # Only start when the port is free; an unresponsive existing
                # service must never be replaced or silently reused.
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    if sys.platform == 'win32':
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    try:
                        probe.bind(('127.0.0.1', args.port))
                    except OSError as exc:
                        raise RuntimeError(f'端口 {args.port} 已占用但服务无响应；请关闭旧服务或使用其他 -Port') from exc
                if not (APP_ROOT / 'node_modules/three/build/three.module.js').is_file():
                    raise RuntimeError('Three.js 未安装；请先运行 setup.sh')
                runner = web.AppRunner(create_app(config))
                await runner.setup()
                await web.TCPSite(runner, '127.0.0.1', args.port).start()
                print(f'预演服务已启动：{url}/', flush=True)
            else:
                if not health.get('ok'):
                    raise RuntimeError(f'端口 {args.port} 上的服务异常；请检查或使用其他端口')
                if bool(health.get('hardware')) != args.hardware:
                    mode = '实机' if args.hardware else '虚拟'
                    raise RuntimeError(f'端口 {args.port} 上已有不同模式的服务；请先清理或使用其他端口（需要{mode}模式）')
                if args.hardware and health.get('motion_scope') != expected_scope:
                    raise RuntimeError(f'现有实机服务不是所需的 {expected_scope} 配置，拒绝复用')
                if health.get('keyboard_control') != 1:
                    cleanup = 'cleanup.ps1' if sys.platform == 'win32' else 'cleanup.sh'
                    raise RuntimeError(f'已有服务仍为旧版本；请先运行同目录 {cleanup}，再重新运行键盘脚本')
                if native_arm_mode and health.get('input_backend') != 'xrobotoolkit':
                    raise RuntimeError('端口上的服务不是 XRoboToolkit 原生双臂服务，拒绝复用')
                print(f'复用已有预演服务：{url}/', flush=True)
            if native_arm_mode:
                from .native_server import read_token
                token = read_token(args.token_file)
                headers = {'X-Teleop-Token': token}
                print(f'原生双臂状态页：{url}/', flush=True)
            else:
                await run_usb(args.port, args.adb, args.serial)
                headers = None
                print(f'电脑预演：{url}/\nQuest 浏览器：{url}/quest', flush=True)
            async with session.ws_connect(f'{url}/ws/keyboard', headers=headers) as ws:
                welcome = await ws.receive_json(timeout=2)
                if welcome.get('type') != 'welcome' or not welcome.get('control'):
                    raise RuntimeError(welcome.get('message', '键盘连接被拒绝'))
                with console:
                    await KeyboardClient(ws, hardware=args.hardware,
                                         auto_calibrate=auto_calibrate,
                                         auto_follow_on_ready=auto_follow_on_ready,
                                         joint_gate=joint_gate,
                                         native_arm_mode=native_arm_mode).run(console)
    finally:
        if runner is not None:
            await runner.cleanup()
        print('键盘遥操已退出。', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Quest USB 遥操：A / B / 空格 / Q / Esc')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--adb', default='adb')
    parser.add_argument('--serial', default='')
    parser.add_argument('--config', default=None)
    parser.add_argument('--hardware', action='store_true',
                        help='启用天机实机输出；默认仅虚拟预演')
    parser.add_argument('--full-arm', action='store_true',
                        help='允许使用双臂整臂配置；仅在现场已确认路径安全时使用')
    parser.add_argument('--auto-calibrate', action='store_true',
                        help='首次连接后等待新鲜 Quest 帧并自动校准一次')
    parser.add_argument('--auto-follow-on-ready', action='store_true',
                        help='联合模式首次稳定追踪后自动校准并开始跟随一次')
    parser.add_argument('--joint-gate', default='',
                        help='联合遥操的灵巧手跟随门控文件（内部参数）')
    parser.add_argument('--native-arm-mode', action='store_true',
                        help='独立 XRoboToolkit 原生双臂模式，不启动 USB/WebXR 采集')
    parser.add_argument('--token-file', default='', help='原生服务键盘授权 token 文件')
    parser.add_argument('--robot-ip', default=None)
    parser.add_argument('--vel-ratio', type=int, default=None)
    parser.add_argument('--acc-ratio', type=int, default=None)
    args = parser.parse_args()
    if args.native_arm_mode and (not args.full_arm or not args.token_file or
                                 args.joint_gate and not args.hardware):
        parser.error('--native-arm-mode 需要 --full-arm --token-file；--joint-gate 还需要 --hardware')
    if args.auto_follow_on_ready and not (
            args.full_arm and (args.native_arm_mode or args.hardware and args.joint_gate)):
        parser.error('--auto-follow-on-ready 仅供原生双臂或实机联合模式使用')
    if not 1024 <= args.port <= 65535:
        parser.error('port 必须在 1024–65535 之间')
    for name, value in (('--vel-ratio', args.vel_ratio), ('--acc-ratio', args.acc_ratio)):
        if value is not None and not 1 <= value <= 100:
            parser.error(f'{name} 必须在 1–100 之间')
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f'启动 / 遥操失败：{str(exc) or type(exc).__name__}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
