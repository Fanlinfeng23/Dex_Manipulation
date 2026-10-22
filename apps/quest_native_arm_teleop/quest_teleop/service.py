"""Network/visualization service; robot calculations live in a separate process."""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import time
from collections import deque, OrderedDict
from pathlib import Path

from aiohttp import web, WSMsgType

from .control_process import ControlProcess
from .protocol import parse_frame, ProtocolError


class Peer:
    """Reliable bounded replies plus one replaceable visualization snapshot."""
    def __init__(self, ws):
        self.ws = ws
        self.replies = deque()
        self.latest = None
        self.ready = asyncio.Event()
        self.task = asyncio.create_task(self.write())

    def reply(self, data):
        if self.ws.closed:
            return
        if len(self.replies) >= 64:
            asyncio.create_task(self.ws.close())
            return
        self.replies.append(json.dumps(data, ensure_ascii=False, allow_nan=False))
        self.ready.set()

    def publish(self, message):
        self.latest = message
        self.ready.set()

    async def write(self):
        try:
            while not self.ws.closed:
                await self.ready.wait()
                if self.replies:
                    message = self.replies.popleft()
                else:
                    message, self.latest = self.latest, None
                if not self.replies and self.latest is None:
                    self.ready.clear()
                if message is not None:
                    # Motion freshness is enforced in the independent worker.
                    # Allow a short socket stall to recover without tearing down
                    # capture; queues remain bounded and snapshots replaceable.
                    await asyncio.wait_for(self.ws.send_str(message), timeout=1.)
        except (ConnectionError, RuntimeError, asyncio.TimeoutError):
            await self.ws.close()

    async def close(self):
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task
        await self.ws.close()


class PreviewServer:
    def __init__(self, config, *, native_token=None, native_status_file=None):
        self.config = config
        self.native_token = native_token
        self.native_status_file = Path(native_status_file) if native_status_file else None
        self.control = ControlProcess(config)
        self.uis = set()
        self.owner = None
        self.quest = None
        self.keyboard = None
        self.peers = {}
        self.task = None
        self.last_state = None
        self.history = deque(maxlen=3600)
        self.events = deque(maxlen=1024)
        self.last_input_key = None
        self.last_recorded = float('-inf')

    @property
    def native_backend(self):
        return self.native_token is not None

    def native_status(self):
        """Treat missing or stale bridge status as ineligible, never as healthy."""
        status = {}
        hand_gate = getattr(self, 'native_joint_gate', None)
        hand_gate_open = False
        if hand_gate is not None:
            try:
                hand_gate_open = 0 <= time.time() - hand_gate.stat().st_mtime <= .5
            except OSError:
                pass
        if self.native_status_file is not None:
            try:
                status = json.loads(self.native_status_file.read_text(encoding='utf-8'))
                if not isinstance(status, dict):
                    status = {}
            except (OSError, ValueError):
                status = {}
        stamp = status.get('wall_ns')
        age_s = time.time() - stamp / 1e9 if type(stamp) is int else float('inf')
        fresh = 0 <= age_s <= 2.5
        validity = status.get('tracking_validity') if fresh else None
        return {
            'native_joint': hand_gate is not None,
            'hand_follow_gate': hand_gate_open,
            'native_tracking_validity': validity if validity in ('reported', 'unreported') else 'unreported',
            'native_eligible': bool(fresh and status.get('eligible') and status.get('ws_connected')),
            'native_reason': status.get('reason', 'waiting_for_native_bridge') if fresh else 'native_status_stale',
            'native_status_age_ms': round(age_s * 1000) if fresh else None,
            'native_input_status': status if fresh else {},
        }

    def decorate_state(self, state):
        if not self.native_backend:
            return state
        enriched = {**state, 'input_backend': 'xrobotoolkit', **self.native_status()}
        enriched['arms'] = {
            side: {**arm, 'tracked': (bool(arm.get('tracked'))
                                    if enriched['native_tracking_validity'] == 'reported' else None),
                   'optical_tracking_validity': enriched['native_tracking_validity'],
                   'numeric_pose_gate': bool(arm.get('tracked'))}
            for side, arm in state['arms'].items()
        }
        return enriched

    def record_state(self, now, state, frame=None):
        if now - self.last_recorded < 1 / 30:
            return
        compact = {key: value for key, value in state.items() if key != 'arms'}
        compact['arms'] = {side: {key: value for key, value in arm.items() if key != 'chain'}
                           for side, arm in state['arms'].items()}
        input_key = (state['source'], frame['session'], frame['seq']) if frame else None
        self.history.append({'recorded_s': now, 'recorded_unix_ms': time.time() * 1000,
                             'input': frame, 'state': compact,
                             'new_input': input_key is not None and input_key != self.last_input_key,
                             'input_fresh': bool(state.get('fresh'))})
        self.last_input_key = input_key
        self.last_recorded = now
        self.prune_history(now)

    def prune_history(self, now):
        while self.history and self.history[0]['recorded_s'] < now - 120:
            self.history.popleft()
        while self.events and self.events[0]['recorded_s'] < now - 120:
            self.events.popleft()

    def record_event(self, kind, **details):
        now = time.perf_counter()
        self.events.append({'type': kind, 'recorded_s': now,
                            'recorded_unix_ms': time.time() * 1000, **details})
        self.prune_history(now)

    async def start(self, app):
        await self.control.start()
        self.task = asyncio.create_task(self.loop())

    async def stop(self, app):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        for peer in list(self.peers.values()):
            await peer.close()
        await self.control.close()

    async def loop(self):
        interval = 1 / self.config['broadcast_hz']
        while True:
            started = time.perf_counter()
            state = self.control.state
            if state:
                state = self.decorate_state(state)
                self.last_state = state
                self.record_state(started, state, self.control.frame)
                message = json.dumps(state, ensure_ascii=False, allow_nan=False)
                # Quest needs status and acknowledgments, not the desktop's link chains.
                compact = dict(state)
                compact['arms'] = {s: {k: v for k, v in arm.items() if k not in ('chain', 'measured_chain')}
                                   for s, arm in state['arms'].items()}
                quest_message = json.dumps(compact, ensure_ascii=False, allow_nan=False)
                for ws, peer in list(self.peers.items()):
                    peer.publish(quest_message if ws is self.quest else message)
            await asyncio.sleep(max(0., interval - (time.perf_counter() - started)))

    async def websocket(self, request):
        origin = request.headers.get('Origin')
        if origin and origin not in (f'http://{request.host}', f'https://{request.host}'):
            raise web.HTTPForbidden(text='Origin mismatch')
        role = request.match_info['role']
        if role not in ('ui', 'quest', 'keyboard'):
            raise web.HTTPNotFound()
        if self.native_backend and role in ('quest', 'keyboard'):
            if not hmac.compare_digest(request.headers.get('X-Teleop-Token', ''), self.native_token):
                raise web.HTTPForbidden(text='Native teleop token required')
        ws = web.WebSocketResponse(max_msg_size=16384, heartbeat=10)
        await ws.prepare(request)
        if role == 'keyboard' and self.keyboard is not None:
            await ws.send_json({'type': 'error', 'message': '已有键盘遥操连接；请先退出原脚本'})
            await ws.close()
            return ws
        if role == 'quest' and self.quest is not None:
            await ws.send_json({'type': 'error', 'message': '已有 Quest 采集连接，请先关闭旧页面'})
            await ws.close()
            return ws
        if role == 'quest':
            self.quest = ws
            self.record_event('quest_connected')
        elif role == 'keyboard':
            self.keyboard = ws
        else:
            self.uis.add(ws)
            if not self.native_backend and self.owner is None:
                self.owner = ws
        peer = self.peers[ws] = Peer(ws)
        peer.reply({'type': 'welcome', 'control': role in ('quest', 'keyboard') or self.owner is ws,
                    'pose_hz': 60, 'frame_ack': True, 'command_ack': True, 'follow_control': True,
                    'input_recovery': True})
        completed = OrderedDict()
        pending_gap = None
        last_frame_seq = -1

        def acknowledged(future):
            nonlocal pending_gap
            if ws.closed or future.cancelled():
                return
            try:
                result = future.result()
                if pending_gap and result.get('accepted') and result['seq'] > pending_gap['after_seq']:
                    self.record_event('input_resumed', source=role, id=pending_gap['id'],
                                      session=result['session'], seq=result['seq'])
                    pending_gap = None
                peer.reply({'type': 'frame_ack', **result})
            except Exception as exc:
                self.control.pause('控制进程不可用')
                peer.reply({'type': 'error', 'message': str(exc)})

        try:
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    if message.type == WSMsgType.ERROR:
                        break
                    continue
                if self.native_backend and role == 'ui':
                    peer.reply({'type': 'error', 'message': '原生双臂状态页只读；请使用授权键盘控制'})
                    continue
                command_id = None
                try:
                    data = json.loads(message.data)
                    if not isinstance(data, dict):
                        raise ProtocolError('消息必须为对象')
                    if self.native_backend and role == 'quest' and data.get('type') != 'frame':
                        raise ValueError('原生采集桥仅允许发送位姿帧')
                    if role == 'ui' and self.owner is not ws:
                        raise ValueError('此窗口为只读；请关闭原控制窗口后刷新')
                    if data.get('type') == 'frame':
                        if role == 'keyboard':
                            raise ValueError('键盘连接不接受位姿帧')
                        frame = parse_frame(data, time.perf_counter())
                        last_frame_seq = max(last_frame_seq, frame.seq)
                        self.control.offer_frame(frame, 'quest' if role == 'quest' else 'simulation').add_done_callback(acknowledged)
                    elif data.get('type') == 'command':
                        command_id = data.get('id')
                        if command_id is not None:
                            if not isinstance(command_id, str) or not 1 <= len(command_id) <= 100:
                                command_id = None
                                raise ProtocolError('command id 错误')
                            if command_id in completed:
                                peer.reply(completed[command_id])
                                continue
                        frame = parse_frame(data['frame'], time.perf_counter()) if data.get('frame') is not None else None
                        result = await self.control.command(data, source=role if role in ('quest', 'keyboard') else None, frame=frame)
                        if data.get('action') in ('arm', 'calibrate'):
                            self.record_event('calibration', source=role, id=command_id,
                                              ok=result['ok'], error=result.get('error'),
                                              input=result.get('calibration_input'))
                        if data.get('action') == 'input_pause' and result.get('ok'):
                            pending_gap = {'id': command_id, 'after_seq': last_frame_seq}
                            self.record_event('input_pause', source=role, id=command_id,
                                              cause=data['value'], input_paused=result['input_paused'],
                                              after_seq=last_frame_seq)
                        if command_id is not None:
                            reply = {'type': 'command_ack', 'id': command_id, **result}
                            completed[command_id] = reply
                            if len(completed) > 128:
                                completed.popitem(last=False)
                            peer.reply(reply)
                        elif not result.get('ok'):
                            peer.reply({'type': 'error', 'message': result.get('error', '命令失败')})
                    else:
                        raise ProtocolError('未知消息类型')
                except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                    if isinstance(exc, (ProtocolError, json.JSONDecodeError)):
                        self.control.pause('输入协议错误；双臂已暂停')
                    if command_id is not None:
                        reply = {'type': 'command_ack', 'id': command_id, 'ok': False, 'error': str(exc)}
                        completed[command_id] = reply
                        if len(completed) > 128:
                            completed.popitem(last=False)
                        peer.reply(reply)
                    else:
                        peer.reply({'type': 'error', 'message': str(exc)})
        finally:
            if role == 'keyboard' and self.keyboard is ws:
                detach = asyncio.create_task(self.control.command({'action': 'keyboard_detach'}, source='keyboard'))
                try:
                    await asyncio.shield(detach)
                except asyncio.CancelledError:
                    # WebSocket close cancels its handler. Do not relinquish
                    # ownership until the worker has acknowledged clutch-off.
                    await detach
                    raise
                finally:
                    self.keyboard = None
            if role == 'quest' and self.quest is ws:
                self.quest = None
                self.record_event('quest_disconnected', close_code=ws.close_code)
                if self.control.state and self.control.state['source'] == 'quest':
                    self.control.pause('Quest 连接已断开')
            self.uis.discard(ws)
            if self.owner is ws:
                self.owner = None
                self.control.pause('控制窗口已关闭')
            self.peers.pop(ws, None)
            await peer.close()
        return ws
