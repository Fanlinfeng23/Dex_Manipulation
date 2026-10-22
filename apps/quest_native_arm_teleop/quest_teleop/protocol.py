"""Protocol v1: JSON in WebSocket messages. Pose coordinates are native WebXR."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np


class ProtocolError(ValueError):
    pass


@dataclass
class Controller:
    position: list
    quaternion: list
    tracked: bool
    grip: float  # Protocol v1 field name; index-trigger analog value (xr-standard button 0).


@dataclass
class Head:
    quaternion: list
    tracked: bool


@dataclass
class Frame:
    session: str
    reference: int
    seq: int
    captured_s: float
    focused: bool
    controllers: dict[str, Controller]
    received_s: float
    head: Head | None = None  # v1 extension; older/simulation frames remain parseable.
    safety: dict = field(default_factory=dict)  # Latched edges from samples omitted by rate limiting.
    buttons: dict = field(default_factory=lambda: {'calibrate': False, 'reset': False})


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProtocolError(f'{label}: 非有限数值')
    return float(value)


def vector(value, count, label):
    if not isinstance(value, list) or len(value) != count:
        raise ProtocolError(f'{label}: 长度错误')
    return [finite_number(v, label) for v in value]


def parse_frame(data, received_s):
    if not isinstance(data, dict) or type(data.get('version')) is not int or data['version'] != 1 or data.get('type') != 'frame':
        raise ProtocolError('帧类型或协议版本错误')
    session = data.get('session')
    if not isinstance(session, str) or not 1 <= len(session) <= 80:
        raise ProtocolError('session 错误')
    for key in ('seq', 'reference'):
        if type(data.get(key)) is not int or not 0 <= data[key] < 2**53:
            raise ProtocolError(f'{key} 错误')
    if type(data.get('focused')) is not bool:
        raise ProtocolError('focused 错误')
    captured = finite_number(data.get('captured_ms'), 'captured_ms') / 1000.
    if captured < 0:
        raise ProtocolError('负采样时间')
    raw = data.get('controllers')
    if not isinstance(raw, dict):
        raise ProtocolError('缺少 controllers')
    controllers = {}
    for side in ('left', 'right'):
        c = raw.get(side)
        if not isinstance(c, dict) or type(c.get('tracked')) is not bool:
            raise ProtocolError(f'{side}: 状态错误')
        p = vector(c.get('position'), 3, 'position')
        q = vector(c.get('quaternion'), 4, 'quaternion')
        norm = np.linalg.norm(q)
        if max(abs(v) for v in p) > 30 or not 0.95 <= norm <= 1.05:
            raise ProtocolError(f'{side}: 非法位置或四元数')
        grip = finite_number(c.get('grip'), 'grip')
        if not 0 <= grip <= 1:
            raise ProtocolError('grip 越界')
        controllers[side] = Controller(p, (np.array(q) / norm).tolist(), c['tracked'], grip)
    head = None
    if data.get('head') is not None:
        raw_head = data['head']
        if not isinstance(raw_head, dict) or type(raw_head.get('tracked')) is not bool:
            raise ProtocolError('head: 状态错误')
        q = vector(raw_head.get('quaternion'), 4, 'head.quaternion')
        norm = np.linalg.norm(q)
        if not 0.95 <= norm <= 1.05:
            raise ProtocolError('head: 非法四元数')
        head = Head((np.array(q) / norm).tolist(), raw_head['tracked'])
    safety = data.get('safety', {})
    if not isinstance(safety, dict):
        raise ProtocolError('safety 错误')
    released = safety.get('released', [])
    if (not isinstance(released, list) or len(released) > 2 or
            any(side not in ('left', 'right') for side in released)):
        raise ProtocolError('safety.released 错误')
    for flag in ('tracking_lost', 'focus_lost'):
        if flag in safety and type(safety[flag]) is not bool:
            raise ProtocolError(f'safety.{flag} 错误')
    safety = {'released': list(dict.fromkeys(released)),
              'tracking_lost': safety.get('tracking_lost', False),
              'focus_lost': safety.get('focus_lost', False)}
    buttons = data.get('buttons', {})
    if not isinstance(buttons, dict) or any(
            key in buttons and type(buttons[key]) is not bool for key in ('calibrate', 'reset')):
        raise ProtocolError('buttons 错误')
    buttons = {key: buttons.get(key, False) for key in ('calibrate', 'reset')}
    return Frame(session, data['reference'], data['seq'], captured, data['focused'], controllers,
                 received_s, head, safety, buttons)
