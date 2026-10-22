"""Bounded IPC around the independent robot control process.

The spawned process owns the vendor *kinematics* library and, in opt-in
``hardware`` mode, the vendor robot SDK connection. A process failure is
latched; creating a new service is required before commands can be accepted
again.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import math
import multiprocessing as mp
import os
import time
from collections import deque
from dataclasses import asdict, replace

from .engine import SIDES, TeleopEngine
from .kinematics import Kinematics
from .hardware import TianjiHardwareOutput
from .timing import TimerResolution


def _safety(frame):
    """Retain safety edges even when newer poses overwrite unsent poses."""
    flags = getattr(frame, 'safety', {}) or {}
    return {
        'released': sorted(set(flags.get('released', ()))),
        'tracking_lost': bool(flags.get('tracking_lost') or
                              not all(frame.controllers[s].tracked for s in SIDES)),
        'focus_lost': bool(flags.get('focus_lost') or not frame.focused),
    }


def _merge_frames(previous, latest):
    a, b = _safety(previous), _safety(latest)
    result = replace(latest)
    result.safety = {
        'released': sorted(set(a['released']) | set(b['released'])),
        'tracking_lost': a['tracking_lost'] or b['tracking_lost'],
        'focus_lost': a['focus_lost'] or b['focus_lost'],
    }
    return result


def _frame_result(frame, accepted, reason):
    return {'accepted': bool(accepted), 'reason': reason,
            'seq': frame.seq, 'session': frame.session}


def _receive(engine, frame, source):
    if source != engine.source:
        return _frame_result(frame, False, 'inactive_source')
    old = engine.frame
    if old and old.session == frame.session and frame.seq <= old.seq:
        return _frame_result(frame, False, 'out_of_order')
    accepted = engine.receive(frame, source)
    return _frame_result(frame, accepted, 'accepted' if accepted else engine.reason)


def _command(engine, item):
    data, source, frame = item['data'], item.get('source'), item.get('frame')
    action = data.get('action')
    calibration_input = None
    try:
        keyboard_actions = ('keyboard_attach', 'keyboard_detach', 'keyboard_heartbeat', 'keyboard_toggle')
        if action in keyboard_actions and source != 'keyboard':
            raise ValueError('此操作仅供键盘遥操连接使用')
        if source == 'keyboard':
            if action not in (*keyboard_actions, 'calibrate', 'reset', 'pause') or frame is not None:
                raise ValueError('键盘仅支持 A / B、空格和退出，不提供手柄位姿')
            if action not in ('keyboard_attach', 'keyboard_detach', 'pause'):
                engine.keyboard_heartbeat()
        if source == 'quest':
            if action not in ('arm', 'calibrate', 'reset', 'pause', 'input_pause'):
                raise ValueError('Quest 采集端仅支持校准、恢复初始姿态和暂停')
            if action not in ('pause', 'input_pause') and engine.source != 'quest':
                raise ValueError('请先在电脑选择 Quest USB 输入')
        if action == 'input_pause':
            reasons = {'ack_timeout': '输入确认延迟超过 150 ms', 'sample_expired': '位姿采样已过期'}
            cause = data.get('value')
            if not isinstance(cause, str) or cause not in reasons or frame is not None:
                raise ValueError('输入暂停原因无效')
            # An idle/unselected input's delayed ACK must not stop the selected input.
            applied = engine.source == ('quest' if source == 'quest' else 'simulation')
            if applied:
                engine.pause(reasons[cause] + '；采集连接保持，继续遥操需重新按 A')
            return {'ok': True, 'input_paused': applied}
        modern_pose_command = action in ('arm', 'calibrate', 'reset') and ('id' in data or 'captured_ms' in data)
        if modern_pose_command and source == 'quest' and frame is None:
            raise ValueError('操作缺少同帧手柄采样；请刷新采集页面')
        if modern_pose_command and source in (None, 'keyboard') and frame is None:
            # The desktop can command the currently selected Quest stream, but
            # its performance.now() belongs to a different browser clock.
            frame = engine.frame
        if modern_pose_command and (source == 'quest' or
                (frame is not None and data.get('session') == frame.session)):
            if frame is None:
                raise ValueError('操作缺少同帧手柄采样；请刷新采集页面')
            captured = data.get('captured_ms')
            if type(captured) not in (int, float) or not math.isfinite(captured) or captured < 0:
                raise ValueError('操作采样时间无效')
            if data.get('session') != frame.session:
                raise ValueError('操作与采样帧不属于同一会话')
            if engine.clock_offset is None:
                raise ValueError('尚无采集时钟基准；等待新鲜输入帧后重试')
            event_age = engine.clock() - captured / 1000. - engine.clock_offset
            if event_age > engine.config['frame_age_limit_s'] or event_age < -.05:
                raise ValueError('操作采样时间已过期或超前；请重新操作')
        # A context sample must not revive an old pose or refresh its arrival
        # timestamp. A sample already consumed with the same seq is legal.
        if frame is not None and action in ('arm', 'calibrate', 'reset'):
            if time.perf_counter() - frame.received_s > engine.config['frame_age_limit_s']:
                raise ValueError('操作采样帧已过期；等待新鲜数据后重试')
            if source == 'quest' and data.get('session') is not None and data['session'] != frame.session:
                raise ValueError('操作与采样帧不属于同一会话')
            old = engine.frame
            same = bool(old and old.session == frame.session and old.seq == frame.seq)
            if same:
                if (old.reference != frame.reference or old.captured_s != frame.captured_s or
                        old.controllers != frame.controllers or old.focused != frame.focused or
                        old.head != frame.head):
                    raise ValueError('操作采样帧与已接收的同序号帧不一致')
                if time.perf_counter() - old.received_s > engine.config['frame_age_limit_s']:
                    raise ValueError('操作采样帧已过期；等待新鲜数据后重试')
                merged = _merge_frames(old, frame)
                merged.received_s = old.received_s
                # In the real-hardware profile a valid newest frame may arrive
                # within the configured tracking-loss grace window. Do not let
                # an older coalesced loss edge reject that fresh frame.
                if (engine.hardware_output is not None and
                        all(merged.controllers[s].tracked for s in engine.control_sides)):
                    merged.safety['tracking_lost'] = False
                engine.frame = merged
                if not engine.triggers_overridden:
                    engine.release_clutch(merged.safety['released'])
                if merged.safety['focus_lost']:
                    engine.pause('Quest 会话不可见 / 已失焦')
                    raise ValueError('操作采样包含失焦；请恢复追踪后重新操作')
                if merged.safety['tracking_lost'] and action != 'reset':
                    engine.pause('手柄失去位置追踪；双臂已暂停')
                    raise ValueError('操作采样包含追踪丢失；请重新操作')
            else:
                if old and old.session == frame.session and frame.seq < old.seq:
                    raise ValueError('操作采样帧已被更新帧替代；请重新操作')
                result = _receive(engine, frame, source or engine.source)
                if not result['accepted']:
                    raise ValueError(result['reason'])
            if not frame.focused or getattr(frame, 'safety', {}).get('focus_lost'):
                engine.pause('Quest 会话不可见 / 已失焦')
                raise ValueError('操作采样包含失焦；请恢复追踪后重新操作')
        stopping = action in ('pause', 'keyboard_detach') or (action == 'follow' and data.get('value') is False)
        if not stopping and time.perf_counter() - item['queued_s'] > engine.config['input_timeout_s']:
            raise ValueError('操作在电脑队列中已过期；请重新操作')
        if action in ('arm', 'calibrate'):
            clutch = data.get('value')
            if clutch is not None and clutch != 'controller':
                raise ValueError('校准控制源无效')
            web_calibration = (source is None and engine.source == 'quest' and
                               not engine.keyboard_mode and clutch != 'controller')
            if clutch == 'controller':
                engine.use_controller_triggers()
            if source == 'quest' and not engine.keyboard_mode:
                # A on the physical controller explicitly returns clutch
                # ownership from desktop Start/Stop to the controller triggers.
                engine.use_controller_triggers()
            current = engine.frame
            calibration_input = {
                'session': current.session if current else None,
                'seq': current.seq if current else None,
                'frame_age_ms': (engine.clock() - current.received_s) * 1000 if current else None,
                'trigger_source': 'keyboard' if engine.keyboard_mode else 'web' if engine.ui_follow is not None else 'controller',
                'raw_triggers': {s: current.controllers[s].grip if current else None for s in SIDES},
                'effective_triggers': {s: engine.trigger(s) for s in SIDES},
                'requested_control': 'web' if web_calibration else 'unchanged',
            }
            engine.calibrate(web_control=web_calibration)
        elif action == 'pause':
            engine.pause()
        elif action == 'reset':
            engine.reset()
        elif action == 'source':
            engine.set_source(data.get('value'))
        elif action == 'scale':
            engine.set_scale(data.get('value'))
        elif action == 'follow':
            engine.set_follow(data.get('value'))
        elif action == 'follow_manual':
            engine.use_controller_triggers()
        elif action == 'keyboard_attach':
            engine.attach_keyboard()
        elif action == 'keyboard_detach':
            engine.detach_keyboard()
        elif action == 'keyboard_heartbeat':
            pass  # Renewed above, before accepting any keyboard operation.
        elif action == 'keyboard_toggle':
            engine.toggle_keyboard_follow()
        else:
            raise ValueError('未知操作')
        return {'ok': True, 'keyboard_follow': engine.keyboard_follow,
                **({'calibration_input': calibration_input} if calibration_input is not None else {})}
    except (ValueError, TypeError, KeyError) as exc:
        return {'ok': False, 'error': str(exc),
                **({'calibration_input': calibration_input} if calibration_input is not None else {})}


def _worker(connection, config):
    """Fixed-rate control with no catch-up loop and no unsolicited IPC writes."""
    engine = None
    hardware = None
    timer_resolution = TimerResolution()
    try:
        timer_resolution.start()
        # perf_counter is monotonic and high resolution on both Windows and
        # Linux. Parent receipt times use this same clock.
        engine = TeleopEngine(config, Kinematics(config), clock=time.perf_counter)
        if config.get('hardware', {}).get('enabled', False):
            hardware = TianjiHardwareOutput(config)
            joints = hardware.connect()
            if config['hardware'].get('startup_move_enabled', False):
                targets = config['hardware']['startup_joints_deg']
                if any(not engine.kine.valid_joints(side, targets[side]) for side in ('left', 'right')):
                    raise ValueError('官方双臂初始姿态超出当前 Lite 模型关节限位')
                joints = hardware.move_to_startup_pose()
            engine.attach_hardware(hardware, joints)
        tick_interval = 1. / config['control_hz']
        snapshot_interval = 1. / config['broadcast_hz']
        diagnostics = {'worker_pid': os.getpid(), 'worker_fault': False,
                       'timer_period_ms': timer_resolution.period_ms,
                       'control_tick_ms': 0., 'control_period_ms': 0.,
                       'control_tick_max_ms': 0., 'control_overruns': 0,
                       'receive_to_consume_ms': 0.}

        def snapshot():
            state = engine.snapshot()
            state['diagnostics'] = {**diagnostics, 'snapshot_s': time.perf_counter()}
            return {'state': state, 'frame': asdict(engine.frame) if engine.frame else None}

        cached = snapshot()
        visual = engine.kine.visual
        connection.send({'ready': True, **cached,
                         'visual_data': visual.data if visual else
                             {'kind': 'schematic', 'model': config['model']},
                         'meshes': {k: str(v) for k, v in visual.meshes.items()} if visual else {}})
        next_tick = previous_tick = time.perf_counter()
        next_snapshot = next_tick + snapshot_interval
        pending_reply = None
        command_dirty = False
        while True:
            now = time.perf_counter()
            if pending_reply is None and connection.poll(max(0., next_tick - now)):
                packet = connection.recv()
                if packet.get('stop'):
                    pause_error = None
                    try:
                        engine.pause('服务关闭')
                    except Exception as exc:
                        # Even a failed hold must not skip the verified
                        # standby request below.
                        pause_error = f'{type(exc).__name__}: {exc}'
                    # The parent must not mistake an early IPC ACK for a
                    # completed SDK standby transition. close() waits for
                    # fresh feedback to show state=0 on both arms.
                    standby_confirmed = None
                    if hardware is not None:
                        standby_confirmed = hardware.close()
                        hardware = None
                    connection.send({'stopped': True,
                                     'standby_confirmed': standby_confirmed,
                                     'pause_error': pause_error})
                    break
                results = []
                for item in packet.get('items', ()):
                    if item['kind'] == 'frame':
                        if item.get('transition') and item['source'] == engine.source:
                            engine.pause('追踪会话或空间已变化；请按 A 重新校准')
                        diagnostics['receive_to_consume_ms'] = round(
                            max(0., time.perf_counter() - item['frame'].received_s) * 1000, 3)
                        result = _receive(engine, item['frame'], item['source'])
                    elif item['kind'] == 'command':
                        result = _command(engine, item)
                        command_dirty = True
                    elif item['kind'] == 'pause':
                        engine.pause(item['reason'])
                        result = {'ok': True}
                        command_dirty = True
                    else:
                        raise ValueError('Invalid control IPC message')
                    results.append(result)
                pending_reply = results
            now = time.perf_counter()
            if now >= next_tick:
                diagnostics['control_period_ms'] = round((now - previous_tick) * 1000, 3)
                previous_tick = now
                start = time.perf_counter()
                engine.step()
                elapsed = time.perf_counter() - start
                diagnostics['control_tick_ms'] = round(elapsed * 1000, 3)
                diagnostics['control_tick_max_ms'] = max(diagnostics['control_tick_max_ms'],
                                                           diagnostics['control_tick_ms'])
                if elapsed > tick_interval:
                    diagnostics['control_overruns'] += 1
                # Reschedule from this tick, never replay missed control ticks.
                next_tick = max(now + tick_interval, time.perf_counter())
            now = time.perf_counter()
            if now >= next_snapshot or command_dirty:
                cached = snapshot()
                next_snapshot = now + snapshot_interval
                command_dirty = False
            if pending_reply is not None:
                # Pose ACKs must not inherit the slower UI snapshot rate.
                # Cached input and state always belong to the same snapshot.
                connection.send({'results': pending_reply, **cached})
                pending_reply = None
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    except BaseException as exc:
        with contextlib.suppress(Exception):
            connection.send({'fatal': f'{type(exc).__name__}: {exc}'})
    finally:
        if engine is not None:
            with contextlib.suppress(Exception):
                engine.pause('控制进程已退出')
        if hardware is not None:
            with contextlib.suppress(Exception):
                hardware.close()
        connection.close()
        timer_resolution.close()


class ControlProcess:
    """Async parent facade. Public methods are called from one asyncio loop.

    At most one Pipe request is in flight. Adjacent frames may be replaced,
    commands never are. There are no unbounded multiprocessing queues.
    """
    QUEUE_LIMIT = 64
    # Hardware startup can include Wuji's 3 s quintic move, 3 s settling,
    # controller mode polling and SDK initialization before the worker is ready.
    START_TIMEOUT_S = 40.
    REPLY_TIMEOUT_S = 1.
    HARDWARE_COMMAND_TIMEOUT_S = 8.
    STOP_REPLY_TIMEOUT_S = 8.

    def __init__(self, config):
        self.config = config
        poll_interval = config.get('ipc_poll_interval_s', .004)
        if (type(poll_interval) not in (int, float) or not math.isfinite(poll_interval)
                or not .0005 <= poll_interval <= .02):
            raise ValueError('ipc_poll_interval_s must be 0.0005–0.02 seconds')
        self.ipc_poll_interval_s = float(poll_interval)
        self.timer_resolution = TimerResolution()
        self.state = None
        self.frame = None
        self.visual_data = None
        self.meshes = {}
        self.fault = None
        self._queue = deque()
        self._process = None
        self._connection = None
        self._pump_task = None
        self._closing = False
        self._stopping = False
        self._stop_requested = False
        self._coalesced = 0
        self._inflight = []
        self._observed = {}
        self._stop_reply = None

    @property
    def healthy(self):
        return self.state is not None and self.fault is None and not self._closing and not self._stopping

    async def start(self):
        if self._process is not None or self.fault:
            raise RuntimeError('控制进程只能启动一次；故障后请重启服务')
        context = mp.get_context('spawn')
        self._connection, child = context.Pipe(duplex=True)
        self._process = context.Process(target=_worker, args=(child, self.config),
                                        name='quest-control', daemon=True)
        self._process.start()
        child.close()
        try:
            self.timer_resolution.start()
            reply = await asyncio.wait_for(asyncio.to_thread(self._connection.recv), self.START_TIMEOUT_S)
            if not reply.get('ready'):
                raise RuntimeError(reply.get('fatal', '控制进程初始化失败'))
            self.visual_data, self.meshes = reply['visual_data'], reply['meshes']
            self._accept_state(reply)
            self._pump_task = asyncio.create_task(self._pump(), name='control-ipc')
        except BaseException as exc:
            self._set_fault('控制进程初始化失败；请重启服务')
            try:
                await self.close()
            except Exception as shutdown_exc:
                raise RuntimeError(f'控制进程启动失败：{exc}；{shutdown_exc}') from exc
            raise

    def _accept_state(self, reply):
        self.state, self.frame = reply['state'], reply['frame']
        self.state.setdefault('diagnostics', {}).update(
            ipc_pending=len(self._queue), frames_coalesced=self._coalesced)
        if self.fault:
            self._mark_fault_state()

    def _mark_fault_state(self):
        if self.state is None:
            return
        self.state = copy.deepcopy(self.state)
        self.state.update(enabled=False, following=False, resetting=False,
                          fresh=False, reason=self.fault, calibrated_heading_deg=None,
                          keyboard_mode=False, keyboard_follow=False, ui_follow=None)
        self.state.setdefault('diagnostics', {})['worker_fault'] = True
        for arm in self.state.get('arms', {}).values():
            arm.update(active=False, tracked=False)

    def _set_fault(self, reason):
        self.fault = self.fault or reason
        self._mark_fault_state()
        for item in list(self._queue) + self._inflight:
            self._resolve(item, self._rejected(item, self.fault))
        self._queue.clear()

    @staticmethod
    def _resolve(item, result):
        future = item.get('_future')
        if future is not None and not future.done():
            future.set_result(result)

    @staticmethod
    def _rejected(item, reason):
        if item['kind'] == 'frame':
            return _frame_result(item['frame'], False, reason)
        return {'ok': False, 'error': reason}

    def _enqueue(self, item):
        future = asyncio.get_running_loop().create_future()
        item['_future'] = future
        item.setdefault('queued_s', time.perf_counter())
        if self.fault or self._closing or self._stopping or self._process is None:
            future.set_result(self._rejected(item, self.fault or '控制进程未运行'))
            return future
        if item['kind'] == 'frame' and self._queue:
            previous = self._queue[-1]
            if previous['kind'] == 'frame' and previous['source'] == item['source']:
                a, b = previous['frame'], item['frame']
                if a.session == b.session and b.seq <= a.seq:
                    future.set_result(_frame_result(b, False, 'out_of_order'))
                    return future
                item['frame'] = _merge_frames(a, b)
                item['transition'] = (previous.get('transition', False) or
                                      a.session != b.session or a.reference != b.reference)
                self._resolve(previous, _frame_result(a, False, 'superseded'))
                self._queue[-1] = item
                self._coalesced += 1
                return future
        if len(self._queue) >= self.QUEUE_LIMIT:
            reason = '控制命令队列溢出；双臂已暂停，请重启服务'
            self._set_fault(reason)
            self._queue.append({'kind': 'pause', 'reason': reason, 'queued_s': time.perf_counter()})
            future.set_result(self._rejected(item, reason))
            return future
        self._queue.append(item)
        return future

    def offer_frame(self, frame, source):
        return self._enqueue({'kind': 'frame', 'frame': self._observe(frame, source), 'source': source})

    def _observe(self, frame, source):
        previous = self._observed.get(source)
        result = replace(frame)
        result.safety = _safety(frame)
        if previous is None or previous.session != frame.session or frame.seq > previous.seq:
            if previous is not None and previous.session == frame.session:
                result.safety['released'] = sorted(set(result.safety['released']) | {
                    side for side in SIDES if previous.controllers[side].grip > .35
                    and frame.controllers[side].grip <= .35})
            self._observed[source] = frame
        return result

    async def command(self, data, source=None, frame=None):
        if frame is not None:
            input_source = source or (self.state or {}).get('source', 'simulation')
            frame = self._observe(frame, input_source)
        return await self._enqueue({'kind': 'command', 'data': dict(data), 'source': source, 'frame': frame})

    def pause(self, reason):
        # A system pause is a queue barrier, just like an explicit button.
        self._enqueue({'kind': 'pause', 'reason': reason})

    def _exchange(self, items):
        wire = [{k: v for k, v in item.items() if not k.startswith('_')} for item in items]
        self._connection.send({'items': wire})
        return self._connection.recv()

    def _exchange_stop(self):
        self._connection.send({'stop': True})
        return self._connection.recv()

    def _reply_timeout(self, items):
        """Allow the vendor state machine time to complete rare hardware commands."""
        if self.config.get('hardware', {}).get('enabled', False):
            for item in items:
                data = item.get('data', {})
                if item.get('kind') == 'command' and data.get('action') in (
                        'follow', 'reset', 'keyboard_toggle'):
                    return self.HARDWARE_COMMAND_TIMEOUT_S
        return self.REPLY_TIMEOUT_S

    async def _pump(self):
        try:
            while not self._closing:
                if self._stop_requested and not self._queue:
                    reply = await asyncio.wait_for(asyncio.to_thread(self._exchange_stop),
                                                   self.STOP_REPLY_TIMEOUT_S)
                    if reply.get('stopped'):
                        self._stop_reply = reply
                        return
                    raise RuntimeError('控制进程未确认停止')
                self._inflight = list(self._queue)
                self._queue.clear()
                reply = await asyncio.wait_for(asyncio.to_thread(self._exchange, self._inflight),
                                               self._reply_timeout(self._inflight))
                if reply.get('fatal'):
                    raise RuntimeError(reply['fatal'])
                if reply.get('stopped'):
                    return
                self._accept_state(reply)
                for item, result in zip(self._inflight, reply['results'], strict=True):
                    self._resolve(item, result)
                self._inflight = []
                # Polling is bounded even while idle; ACKs are independent of
                # the slower snapshot rate so a 60 Hz sender stays at 60 Hz.
                await asyncio.sleep(self.ipc_poll_interval_s)
        except asyncio.CancelledError:
            raise
        except (Exception, asyncio.TimeoutError) as exc:
            outcome = ('实机停止未确认，请使用现场急停' if self.config.get('hardware', {}).get('enabled')
                       else '双臂已暂停，请重启服务')
            self._set_fault(f'控制进程失去响应；{outcome} ({type(exc).__name__})')
            # Only this service's own control worker may be stopped here.
            if self._process is not None and self._process.is_alive():
                self._process.terminate()

    async def close(self):
        """Stop the worker; live hardware must ACK a verified standby state."""
        if self._closing:
            return
        self._stopping = True
        self._stop_requested = True
        for item in list(self._queue) + self._inflight:
            self._resolve(item, self._rejected(item, '服务关闭'))
        self._queue.clear()

        # Give the worker a chance to execute its hardware shutdown path. A
        # direct terminate() here is acceptable for the virtual preview but can
        # leave a connected Tianji controller in impedance mode.
        stopped_cleanly = False
        if self._process is not None:
            if self._process.is_alive():
                if self._pump_task and not self._pump_task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(self._pump_task),
                                               self.STOP_REPLY_TIMEOUT_S + 2.)
                        stopped_cleanly = self._stop_reply is not None
                    except (Exception, asyncio.TimeoutError):
                        pass
                else:
                    try:
                        reply = await asyncio.wait_for(asyncio.to_thread(self._exchange_stop),
                                                       self.STOP_REPLY_TIMEOUT_S)
                        self._stop_reply = reply if reply.get('stopped') else None
                        stopped_cleanly = bool(reply.get('stopped'))
                    except (Exception, asyncio.TimeoutError):
                        pass
            if self._process.is_alive() and not stopped_cleanly:
                self._process.terminate()
            await asyncio.to_thread(self._process.join, 2.)
            if self._process.is_alive():
                self._process.kill()
                await asyncio.to_thread(self._process.join, 2.)
        if self._pump_task:
            if not self._pump_task.done():
                self._pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump_task
        self._closing = True
        self.timer_resolution.close()
        if self._connection is not None:
            self._connection.close()
        if self.config.get('hardware', {}).get('enabled', False):
            if not stopped_cleanly or not self._stop_reply or self._stop_reply.get('standby_confirmed') is not True:
                reason = '实机未确认双臂待机；请核对控制柜状态，必要时使用现场急停'
                self._set_fault(reason)
                raise RuntimeError(reason)
        return self._stop_reply
