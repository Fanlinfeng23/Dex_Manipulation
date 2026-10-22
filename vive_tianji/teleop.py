"""Two-wrist Vive Tracker → Tianji dual-arm teleoperation.

The default commands only read trackers or simulate the robot. Real hardware
requires an explicit ``live`` command, a read-only arm preflight, a typed
confirmation, keyboard calibration and a separate Space press.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from quest_teleop.config import load_config
from quest_teleop.control_process import ControlProcess
from quest_teleop.keyboard import UnixConsole
from quest_teleop.math3d import XR_TO_WORLD
from quest_teleop.protocol import Controller, Frame
from tools.diagnose_robot import diagnose
from tools.check_full_arm_pose import check as check_arm_pose

from .calibration import basis_from_forward_motion, measured_offset_from_mm, solve_pivot
from .tracker_input import BASE_STATION_SERIALS, TrackerPair


ROOT = Path(__file__).resolve().parents[1]
APP = Path(os.environ.get('TIANJI_QUEST_APP', str(ROOT / 'apps/quest_native_arm_teleop')))
BASE_CONFIG = APP / 'configs/hardware_dual_arm.json'
CALIBRATION = Path(os.environ.get('TIANJI_VIVE_CALIBRATION_FILE',
                                  ROOT / 'vive_tianji_calibration.json'))
SERIALS = {'left': 'LHR-8180F66F', 'right': 'LHR-00037B1A'}
SIDES = ('left', 'right')
SAMPLE_HZ = 120.0
MIN_BASE_STATION_BASELINE_M = .8
MAX_BASE_STATION_BASELINE_M = 5.
MAX_STATION_SHIFT_M = .010
MAX_STATION_ROTATION_RAD = .010


class TrackingGateError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


def _both_usable(sample, *, baseline_limits_m=None) -> bool:
    return bool(not sample.reference_changed and sample.base_station_count >= 2 and
                _baseline_usable(sample, baseline_limits_m=baseline_limits_m) and
                all(sample.poses[side] is not None for side in SIDES))


def _tracking_usable(sample) -> bool:
    """Live follow needs fresh valid poses for both wrists, not two base poses.

    ``TrackerPair`` exposes a wrist pose only when OpenVR reports a connected,
    valid Running_OK 6DoF pose in the same snapshot.  A base station reference
    pose is scene geometry, not proof that it illuminated either wrist on this
    frame.  Its transient absence must not be mistaken for lost wrist tracking.
    """
    return bool(not sample.reference_changed and
                all(sample.poses[side] is not None for side in SIDES))


def _baseline_usable(sample, *, baseline_limits_m=None) -> bool:
    baseline = sample.base_station_baseline_m
    lower, upper = baseline_limits_m or (MIN_BASE_STATION_BASELINE_M,
                                         MAX_BASE_STATION_BASELINE_M)
    return bool(baseline is not None and math.isfinite(baseline) and
                lower <= baseline <= upper)


def _baseline_limits(expected: float | None):
    """Bound an explicitly measured layout without changing the default gate."""
    if expected is None:
        return MIN_BASE_STATION_BASELINE_M, MAX_BASE_STATION_BASELINE_M
    if not math.isfinite(expected) or not .15 <= expected <= 5.:
        raise ValueError('--expected-baseline-m 必须在 0.15–5.0 m 之间')
    return expected - .020, expected + .020


def _check_live_chaperone(status: dict) -> None:
    """Headless tracking may lack room bounds; conflicting rooms may not drive arms."""
    state = status.get('state')
    driver = status.get('hmd_tracking_system')
    if state == 1:
        return
    if state == 201 and driver == 'null':
        print('SteamVR Chaperone=201，当前为无头显 null 驱动；仅在双腕/双站追踪、'
              '场地几何、本项目交互标定及机器人只读诊断门禁全部通过后允许受限实机测试。')
        return
    if state == 201:
        detail = '基站与追踪中心未完成房间标定'
    elif state == 202:
        detail = '两台基站的追踪空间冲突'
    else:
        detail = '房间标定状态异常'
    raise RuntimeError(
        f'SteamVR Chaperone={state}（{detail}），HMD 驱动={driver!r}；'
        '实机拒绝启动。先恢复双腕与双站连续 6DoF 和真实基站间距；'
        '如使用实体头显，在 SteamVR 运行 Room Setup，然后重新执行双腕标定。')


def _gather(pair: TrackerPair, seconds: float, *, side: str | None = None,
            baseline_limits_m=None, max_invalid_streak_frames: int = 2):
    """Collect bounded read-only samples at the intended control frequency."""
    count = max(1, round(seconds * SAMPLE_HZ))
    rows = []
    good = 0
    bad_streak = max_bad_streak = tail_good = 0
    max_sample_gap = 0.
    previous_time = None
    base_counts = Counter()
    baselines = []
    connected_counts = Counter()
    pose_counts = Counter()
    tracking_results = {side: Counter() for side in SIDES}
    start = time.perf_counter()
    for index in range(count):
        sample = pair.sample()
        base_counts[sample.base_station_count] += 1
        if (sample.base_station_baseline_m is not None and
                math.isfinite(sample.base_station_baseline_m)):
            baselines.append(sample.base_station_baseline_m)
        for name in SIDES:
            connected_counts[name] += bool(sample.states[name]['connected'])
            pose_counts[name] += sample.poses[name] is not None
            tracking_results[name][sample.states[name]['tracking_result']] += 1
        if previous_time is not None:
            max_sample_gap = max(max_sample_gap, sample.time_s - previous_time)
        previous_time = sample.time_s
        if sample.reference_changed:
            raise RuntimeError('SteamVR 站立参考系发生变化；请重新运行标定')
        usable = (sample.base_station_count >= 2 and
                  _baseline_usable(sample, baseline_limits_m=baseline_limits_m) and
                  (sample.poses[side] is not None if side else
                   _both_usable(sample, baseline_limits_m=baseline_limits_m)))
        if usable:
            good += 1
            rows.append(sample)
            bad_streak = 0
            tail_good += 1
        else:
            bad_streak += 1
            max_bad_streak = max(max_bad_streak, bad_streak)
            tail_good = 0
        delay = start + (index + 1) / SAMPLE_HZ - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
    ratio = good / count
    if (ratio < .98 or max_bad_streak > max_invalid_streak_frames or
            max_sample_gap > .05 or tail_good < min(count, round(.75 * SAMPLE_HZ))):
        which = side or '双腕'
        station = ', '.join(f'{number}台有效:{frames}帧' for number, frames in sorted(base_counts.items()))
        baseline_line = (f'两台指定基站间距 {baselines[-1]:.3f} m'
                         if baselines else '两台指定基站未同时提供有效位姿')
        devices = '；'.join(
            f'{name} {SERIALS[name]} 无线连接 {connected_counts[name]}/{count}、'
            f'6DoF {pose_counts[name]}/{count}、'
            f'追踪状态 {dict(tracking_results[name])}' for name in SIDES)
        lower, upper = baseline_limits_m or (MIN_BASE_STATION_BASELINE_M,
                                             MAX_BASE_STATION_BASELINE_M)
        diagnostics = {
            'samples': count, 'valid_samples': good,
            'base_station_valid_frames': {str(k): v for k, v in base_counts.items()},
            'base_station_baseline_m': {
                'latest': baselines[-1] if baselines else None,
                'min': min(baselines) if baselines else None,
                'max': max(baselines) if baselines else None,
                'required_min': lower,
                'required_max': upper,
            },
            'max_consecutive_invalid_frames': max_bad_streak,
            'allowed_consecutive_invalid_frames': max_invalid_streak_frames,
            'max_sample_gap_ms': round(max_sample_gap * 1000, 2),
            'trackers': {name: {
                'serial': SERIALS[name], 'connected_frames': connected_counts[name],
                'valid_6dof_frames': pose_counts[name],
                'tracking_results': {str(k): v for k, v in tracking_results[name].items()},
            } for name in SIDES},
        }
        raise TrackingGateError(
            f'{which} 有效定位 {good}/{count}，最长连续无效 {max_bad_streak} 帧，'
            f'最大采样间隔 {max_sample_gap * 1000:.1f} ms；'
            f'基站 {station}；{baseline_line}；{devices}。'
            f'需两台指定基站同帧有效、间距 {lower:.3f}–'
            f'{upper:.3f} m、≥98% 有效、最长连续无效不超过 '
            f'{max_invalid_streak_frames} 帧且最后 0.75 秒连续定位', diagnostics)
    return rows, ratio


def _station_geometry(sample, recorded: dict, *, require_both: bool = True) -> None:
    """Reject a relocated/re-solved Lighthouse layout before using a wrist fit.

    OpenVR reference poses describe scene geometry, not which station supplied
    optical hits on this particular frame.  Coverage is tested separately.
    """
    if not isinstance(recorded, dict) or set(recorded) != set(BASE_STATION_SERIALS):
        raise RuntimeError('标定文件缺少两台指定基站的几何指纹；请重新标定')
    if sample is None or (require_both and
                          set(sample.base_station_poses) != set(BASE_STATION_SERIALS)):
        raise RuntimeError('两台指定基站的有效位姿不完整；无法核对场地标定')
    for serial in set(sample.base_station_poses) & set(BASE_STATION_SERIALS):
        old = np.asarray(recorded[serial], dtype=float)
        new = sample.base_station_poses[serial]
        if old.shape != (4, 4) or not np.isfinite(old).all():
            raise RuntimeError(f'{serial}: 标定文件中的基站位姿无效')
        distance = float(np.linalg.norm(new[:3, 3] - old[:3, 3]))
        angle = float(np.linalg.norm(Rotation.from_matrix(
            old[:3, :3].T @ new[:3, :3]).as_rotvec()))
        if distance > MAX_STATION_SHIFT_M or angle > MAX_STATION_ROTATION_RAD:
            raise RuntimeError(
                f'{serial}: 基站场地几何相对标定改变 {distance * 1000:.1f} mm / '
                f'{math.degrees(angle):.2f}°；停止本次会话，重新运行 calibrate')


def _read_calibration(pair: TrackerPair, *, sample=None, hardware: bool = False,
                      expected_baseline_m: float | None = None):
    try:
        data = json.loads(CALIBRATION.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise RuntimeError(f'尚未标定；先运行 ./tianji_vive.sh calibrate') from exc
    if data.get('version') != 3 or data.get('serials') != SERIALS:
        raise RuntimeError('双腕标定版本或 Tracker 序列号不匹配；请重新标定')
    saved_baseline = data.get('expected_baseline_m')
    if ((saved_baseline is None) != (expected_baseline_m is None) or
            (saved_baseline is not None and
             (not isinstance(saved_baseline, (int, float)) or
              not math.isfinite(saved_baseline) or
              not math.isclose(saved_baseline, expected_baseline_m, abs_tol=.001)))):
        raise RuntimeError('标定文件与本次基站间距配置不一致；请用相同的 --expected-baseline-m 重新标定')
    if hardware and data.get('test_only'):
        raise RuntimeError('测试用虚拟标定不能用于实机；请佩戴双腕 Tracker 运行 ./tianji_vive.sh calibrate')
    if pair.reference_matrix is None:
        raise RuntimeError('SteamVR 站立参考系不可用')
    recorded = np.asarray(data.get('reference_matrix'), dtype=float)
    if recorded.shape != (4, 4) or not np.isfinite(recorded).all():
        raise RuntimeError('标定文件缺少有效的 SteamVR 参考系')
    delta = np.linalg.norm(recorded[:3, 3] - pair.reference_matrix[:3, 3])
    angle = np.linalg.norm(Rotation.from_matrix(
        recorded[:3, :3].T @ pair.reference_matrix[:3, :3]).as_rotvec())
    if delta > .002 or angle > .002:
        raise RuntimeError('SteamVR 站立参考系已改变；请重新运行 ./tianji_vive.sh calibrate')
    if not data.get('test_only'):
        _station_geometry(sample, data.get('base_station_poses_standing'))
    offsets = data.get('offset_tracker_m', {})
    sources = data.get('offset_calibration', {})
    if not isinstance(sources, dict):
        raise RuntimeError('腕部偏移方法记录无效')
    for side in SIDES:
        value = np.asarray(offsets.get(side), dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) > .25:
            raise RuntimeError(f'{side}: 标定偏移无效')
        source = sources.get(side)
        if source is not None and not isinstance(source, dict):
            raise RuntimeError(f'{side}: 腕部偏移方法记录无效')
        # Older version-3 files are pivot-only.  New measured entries must
        # explicitly bind the signed local vector to this tracker's serial.
        method = source.get('method') if isinstance(source, dict) else 'pivot'
        if method == 'measured':
            if (source.get('serial') != SERIALS[side] or
                    source.get('frame') != 'openvr_tracker_local_xyz' or
                    source.get('origin') != 'tracker_3_mounted_screw_datum' or
                    source.get('landmark') != 'anatomical_wrist_center'):
                raise RuntimeError(f'{side}: 手工测量偏移缺少对应 Tracker 的坐标和来源信息')
            try:
                measured = measured_offset_from_mm(source.get('offset_mm'))
            except ValueError as exc:
                raise RuntimeError(f'{side}: 手工测量偏移无效：{exc}') from exc
            if not np.allclose(value, measured, atol=1e-9, rtol=0):
                raise RuntimeError(f'{side}: 手工测量记录与实际偏移不一致')
        elif method == 'pivot':
            if source is not None and (
                    source.get('serial') != SERIALS[side] or
                    source.get('frame') != 'openvr_tracker_local_xyz' or
                    source.get('landmark') != 'anatomical_wrist_center'):
                raise RuntimeError(f'{side}: 枢轴标定对应的 Tracker 序列号或坐标信息不匹配')
            quality = data.get('pivot_quality', {}).get(side, {})
            if not (math.isfinite(float(quality.get('rms_m', float('nan')))) and
                    float(quality['rms_m']) <= .008):
                raise RuntimeError(f'{side}: 枢轴拟合质量无效')
            if hardware and not (
                    math.isfinite(float(quality.get('max_m', float('nan')))) and
                    float(quality['max_m']) <= .020 and
                    float(quality.get('rotation_span_deg', 0)) >= 20 and
                    float(quality.get('samples', 0)) >= 8):
                raise RuntimeError(f'{side}: 实机所需枢轴标定信息不完整')
        else:
            raise RuntimeError(f'{side}: 未知的腕部偏移标定方法 {method!r}')
    yaw = float(data.get('operator_yaw_deg'))
    if not math.isfinite(yaw):
        raise RuntimeError('标定朝向无效')
    if hardware:
        basis = np.asarray(data.get('standing_to_robot_rotation'), dtype=float)
        expected = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ XR_TO_WORLD
        if (basis.shape != (3, 3) or not np.isfinite(basis).all() or
                not np.allclose(basis, expected, atol=1e-5, rtol=0)):
            raise RuntimeError('标定朝向矩阵与偏航角不一致；请重新标定')
        motion = data.get('forward_motion_m', {})
        for side in SIDES:
            delta = np.asarray(motion.get(side), dtype=float)
            if delta.shape != (3,) or not np.isfinite(delta).all() or np.linalg.norm(delta[[0, 2]]) < .18:
                raise RuntimeError(f'{side}: 前向动作标定信息不完整')
        repeat = data.get('forward_repeatability', {})
        if (float(repeat.get('direction_difference_deg', float('inf'))) > 3.
                or float(repeat.get('return_max_m', float('inf'))) > .015):
            raise RuntimeError('前向动作重复性不符合实机标定要求')
    return data


def _uncalibrated_preview_calibration(pair: TrackerPair) -> dict:
    """In-memory relative-motion placeholder; it must never authorize hardware."""
    reference = pair.reference_matrix
    if reference is None or reference.shape != (4, 4) or not np.isfinite(reference).all():
        raise RuntimeError('SteamVR 站立参考系不可用；无法建立预演坐标')
    return {
        'version': 3, 'serials': SERIALS.copy(), 'test_only': True,
        'reference_matrix': reference.tolist(),
        'operator_yaw_deg': 0.,
        'standing_to_robot_rotation': XR_TO_WORLD.tolist(),
        'offset_tracker_m': {side: [0., 0., 0.] for side in SIDES},
        'pivot_quality': {side: {'rms_m': 0.} for side in SIDES},
        'note': '仅无实机相对运动预演；手腕偏移和操作员朝向未经标定',
    }


def _quality_line(sample) -> str:
    baseline = sample.base_station_baseline_m
    baseline_line = f'{baseline:.3f} m' if baseline is not None else '不可用'
    return (f"基站 {sample.base_station_count}/2，间距 {baseline_line}，"
            + '，'.join(f"{side} {SERIALS[side]}: "
                       f"{'6DoF OK' if sample.poses[side] is not None else '未定位'}"
                       for side in SIDES))


def _centers_from_stationary_samples(rows, offsets):
    centers = {}
    for side in SIDES:
        points = np.asarray([row.poses[side][:3, 3] +
                             row.poses[side][:3, :3] @ offsets[side] for row in rows])
        centers[side] = np.median(points, axis=0)
        jitter = np.sqrt(np.mean(np.sum((points - centers[side]) ** 2, axis=1)))
        segment = max(1, len(points) // 4)
        drift = np.linalg.norm(np.median(points[:segment], axis=0) -
                               np.median(points[-segment:], axis=0))
        if jitter > .008 or drift > .006:
            raise RuntimeError(f'{side}: 静止姿态 RMS {jitter * 1000:.1f} mm、'
                               f'首尾漂移 {drift * 1000:.1f} mm；请稳住双腕重标定')
        print(f'{side}: 静止姿态 RMS {jitter * 1000:.1f} mm，首尾漂移 {drift * 1000:.1f} mm')
    return centers


def _repeated_forward_basis(neutral, forward1, returned, forward2):
    """Average two independent long forward moves and check the return point."""
    return_max = max(float(np.linalg.norm(returned[side] - neutral[side]))
                     for side in SIDES)
    if return_max > .015:
        raise RuntimeError(f'双腕回到起点的最大偏差 {return_max * 1000:.1f} mm 超过 15 mm；请重新标定')
    moves = []
    bases = []
    for start, end in ((neutral, forward1), (returned, forward2)):
        deltas = {side: end[side] - start[side] for side in SIDES}
        for side in SIDES:
            horizontal = np.linalg.norm(deltas[side][[0, 2]])
            if horizontal < .18 or horizontal > .35:
                raise RuntimeError(f'{side}: 前向动作需在 18–35 cm 之间，实际 {horizontal * 100:.1f} cm')
        bases.append(basis_from_forward_motion(
            start['left'], start['right'], end['left'], end['right']))
        moves.append(deltas)
    disagreement = math.degrees(math.acos(float(np.clip(
        np.dot(bases[0][0], bases[1][0]), -1., 1.))))
    if disagreement > 3.:
        raise RuntimeError(f'两次前向标定方向相差 {disagreement:.1f}°，超过 3°；请重新标定')
    forward = bases[0][0] + bases[1][0]
    forward /= np.linalg.norm(forward)
    up = np.array([0., 1., 0.])
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    basis = np.stack((forward, left, up), axis=0)
    averaged_moves = {side: ((moves[0][side] + moves[1][side]) / 2.).tolist()
                      for side in SIDES}
    quality = {'direction_difference_deg': disagreement, 'return_max_m': return_max,
               'per_pass_m': [{side: float(np.linalg.norm(move[side][[0, 2]]))
                               for side in SIDES} for move in moves]}
    return basis, averaged_moves, quality


def _gather_startup(pair: TrackerPair, baseline_limits):
    """Repeat only a nearly valid startup sample before any arm connection."""
    for attempt in range(3):
        try:
            return _gather(pair, 3.0, baseline_limits_m=baseline_limits,
                           max_invalid_streak_frames=3)
        except TrackingGateError as exc:
            details = exc.diagnostics
            brief_loss = (details['valid_samples'] / details['samples'] >= .98 and
                          details['max_consecutive_invalid_frames'] <= 3)
            if not brief_loss or attempt == 2:
                raise
            print(f'双腕采样短时波动；自动重新采样 3 秒（第 {attempt + 2}/3 次）。')


def _save_failed_pivot(side: str, samples, error: ValueError) -> Path:
    """Keep read-only Tracker poses so a failed fit can be diagnosed later."""
    folder = ROOT / 'logs'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (f'vive-pivot-{side}-{datetime.now():%Y%m%d-%H%M%S}-'
                     f'{uuid.uuid4().hex[:8]}.json')
    data = {
        'side': side, 'serial': SERIALS[side], 'error': str(error),
        'sample_hz': SAMPLE_HZ, 'samples': len(samples),
        'tracker_poses_standing': [row.poses[side].tolist() for row in samples],
        'base_station_baseline_m': [row.base_station_baseline_m for row in samples],
    }
    path.write_text(json.dumps(data, ensure_ascii=False) + '\n', encoding='utf-8')
    return path


def _collect_pivot(pair: TrackerPair, side: str, label: str,
                   baseline_limits, initial_station_poses):
    """Allow a failed wrist sample to be repeated without discarding the other wrist."""
    attempt = 0
    while True:
        attempt += 1
        input(f'稳住{label}前臂和手腕关节位置，露出 Tracker 传感器；回车后立即缓慢沿两个方向转动{label}腕 8 秒'
              f'（第 {attempt} 次），勿平移手臂或松动绑带；按回车开始：')
        try:
            samples, _ = _gather(pair, 8.0, baseline_limits_m=baseline_limits,
                                  max_invalid_streak_frames=3)
        except TrackingGateError as exc:
            print(f'{label}腕采样中追踪暂时不稳定：{exc}')
            reason = exc
        else:
            _station_geometry(samples[-1], initial_station_poses)
            try:
                return solve_pivot([row.poses[side] for row in samples])
            except ValueError as exc:
                reason = exc
                try:
                    path = _save_failed_pivot(side, samples, exc)
                except OSError as save_error:
                    print(f'{label}腕失败采样未能保存：{save_error}')
                else:
                    print(f'{label}腕失败采样记录：{path}')
                print(f'{label}腕枢轴拟合未通过：{exc}。请检查连接件是否松动，'
                      '将前臂靠在稳定支撑上、手腕中心尽量留在原位，'
                      '在传感器无遮挡时缓慢改变两个转动方向。')
        while True:
            answer = input(f'按回车仅重采{label}腕（此前完成的步骤保留）；输入 q 回车退出：').strip().lower()
            if not answer:
                break
            if answer == 'q':
                raise RuntimeError(f'{label}腕标定未通过；本次未保存双腕标定') from reason
            print('请输入回车重采，或输入 q 回车退出。')


def calibrate(args):
    print('双腕标定只读取 SteamVR，不连接天机臂。')
    expected_baseline = getattr(args, 'expected_baseline_m', None)
    baseline_limits = _baseline_limits(expected_baseline)
    measured_offsets = {}
    for side in SIDES:
        millimetres = getattr(args, f'{side}_offset_mm', None)
        if millimetres is not None:
            measured_offsets[side] = measured_offset_from_mm(millimetres)
    with TrackerPair(SERIALS['left'], SERIALS['right']) as pair:
        rows, _ = _gather_startup(pair, baseline_limits)
        initial_station_poses = {
            serial: rows[-1].base_station_poses[serial].tolist()
            for serial in BASE_STATION_SERIALS
        }
        print('追踪门禁通过；' + _quality_line(rows[-1]))
        offsets, pivot_quality, offset_calibration = {}, {}, {}
        for side, label in (('left', '左'), ('right', '右')):
            if side in measured_offsets:
                measured = measured_offsets[side]
                offsets[side] = measured.tolist()
                offset_calibration[side] = {
                    'method': 'measured', 'serial': SERIALS[side],
                    'frame': 'openvr_tracker_local_xyz',
                    'origin': 'tracker_3_mounted_screw_datum',
                    'landmark': 'anatomical_wrist_center',
                    'offset_mm': (measured * 1000.0).tolist(),
                }
                print(f'{label}腕采用实测局部偏移 '
                      f'{np.round(measured * 1000., 1).tolist()} mm；'
                      '该数值由操作员提供，系统不能从追踪数据确认其绝对精度。')
            else:
                result = _collect_pivot(pair, side, label, baseline_limits,
                                        initial_station_poses)
                offsets[side] = result.offset_tracker_m.tolist()
                offset_calibration[side] = {
                    'method': 'pivot', 'serial': SERIALS[side],
                    'frame': 'openvr_tracker_local_xyz',
                    'landmark': 'anatomical_wrist_center',
                }
                pivot_quality[side] = {
                    'rms_m': result.rms_m, 'max_m': result.max_m,
                    'rotation_span_deg': result.rotation_span_deg,
                    'condition_number': result.condition_number, 'samples': result.samples,
                }
                print(f'{label}腕偏移 {np.round(result.offset_tracker_m * 1000, 1).tolist()} mm；'
                      f'拟合 RMS {result.rms_m * 1000:.1f} mm，'
                      f'最大 {result.max_m * 1000:.1f} mm，转动跨度 {result.rotation_span_deg:.1f}°')
        input('面对机器人，双腕自然放在身体两侧、同高且不交叉，间距至少 18 cm；静止后按回车采样 2 秒：')
        rows, _ = _gather(pair, 2.0, baseline_limits_m=baseline_limits,
                          max_invalid_streak_frames=3)
        _station_geometry(rows[-1], initial_station_poses)
        centers = _centers_from_stationary_samples(rows, offsets)
        input('双腕保持间距和朝向，沿机器人正前方平移 20–30 cm，不要抬高或旋腕；终点静止后按回车采样 1 秒：')
        rows, _ = _gather(pair, 1.0, baseline_limits_m=baseline_limits,
                          max_invalid_streak_frames=3)
        _station_geometry(rows[-1], initial_station_poses)
        moved1 = _centers_from_stationary_samples(rows, offsets)
        input('双腕返回刚才起点并静止；按回车采样 1 秒：')
        rows, _ = _gather(pair, 1.0, baseline_limits_m=baseline_limits,
                          max_invalid_streak_frames=3)
        _station_geometry(rows[-1], initial_station_poses)
        returned = _centers_from_stationary_samples(rows, offsets)
        input('再次沿同一正前方平移 20–30 cm，到终点静止后按回车采样 1 秒：')
        rows, _ = _gather(pair, 1.0, baseline_limits_m=baseline_limits,
                          max_invalid_streak_frames=3)
        _station_geometry(rows[-1], initial_station_poses)
        moved2 = _centers_from_stationary_samples(rows, offsets)
        basis, average_motion, forward_quality = _repeated_forward_basis(
            centers, moved1, returned, moved2)
        yaw_matrix = basis @ XR_TO_WORLD.T
        if (not np.allclose(yaw_matrix[2], [0, 0, 1], atol=1e-6) or
                not np.allclose(yaw_matrix[:, 2], [0, 0, 1], atol=1e-6)):
            raise RuntimeError('双腕中位姿态无法建立水平朝向')
        yaw = math.degrees(math.atan2(yaw_matrix[1, 0], yaw_matrix[0, 0]))
        data = {
            'version': 3, 'created_at': datetime.now().astimezone().isoformat(),
            'serials': SERIALS, 'offset_tracker_m': offsets,
            'expected_baseline_m': expected_baseline,
            'pivot_quality': pivot_quality,
            'offset_calibration': offset_calibration,
            'operator_yaw_deg': yaw,
            'reference_matrix': pair.reference_matrix.tolist(),
            'base_station_poses_standing': initial_station_poses,
            'standing_to_robot_rotation': basis.tolist(),
            'forward_motion_m': average_motion,
            'forward_repeatability': forward_quality,
            'note': '相对遥操坐标标定；不是机器人绝对手眼标定',
        }
        temporary = CALIBRATION.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(CALIBRATION)
        print(f'双腕标定已保存：{CALIBRATION}；映射偏航 {yaw:.1f}°，'
              f'两次方向相差 {forward_quality["direction_difference_deg"]:.2f}°')


def _config(calibration, *, hardware: bool, side: str, scale: float,
            full_range: bool = False):
    if full_range and (not hardware or side != 'both'):
        raise ValueError('正常幅度仅用于双臂实机模式')
    config = load_config(BASE_CONFIG, hardware=True)
    config['initial_source'] = 'quest'
    config['heading_source'] = 'fixed'
    config['operator_yaw_deg'] = float(calibration['operator_yaw_deg'])
    config['position_scale'] = float(scale)
    config['control_hz'] = int(SAMPLE_HZ)
    config['broadcast_hz'] = 20
    # The shared control IPC defaults to 4 ms for Quest.  At 120 Hz that idle
    # wait alone consumes nearly half a Tracker period, so use a bounded 1 ms
    # poll interval for this high-rate OpenVR input.
    config['ipc_poll_interval_s'] = .001
    config['input_timeout_s'] = .18
    config['frame_age_limit_s'] = .12
    config['tracking_loss_grace_s'] = .08
    # A short optical loss first holds the robot.  If it outlasts the grace
    # period, the shared controller requires a stable fresh stream and anchors
    # both wrists to measured arm poses before resuming.  An operator pause,
    # reference reset, or hardware fault still requires explicit intervention.
    config['keyboard_auto_recover_tracking'] = True
    config['keyboard_auto_resume_tracking'] = True
    config['filter_time_s'] = .001
    dual_arm = side == 'both'
    # Keep the original conservative single-arm profile for staged testing.
    # Dual-arm tracking uses the established Tianji impedance profile and
    # software speed bounds from hardware_dual_arm.json.
    if dual_arm:
        # Wuji's OpenVR TF path publishes each valid pose directly.  Our
        # downstream Cartesian and joint speed bounds still smooth commands.
        config['input_filter'] = {'type': 'none'}
    else:
        config['input_filter'] = {'type': 'one_euro', 'min_cutoff_hz': 2., 'beta': .5,
                                  'derivative_cutoff_hz': 1., 'position_beta_scale': 20.,
                                  'rotation_beta_scale': 4.}
        config['max_target_speed_m_s'] = .04
        config['max_target_rotation_rad_s'] = .15
        config['max_joint_speed_deg_s'] = 2.
        config['max_joint_step_deg'] = .12
    config['workspace_radius_m'] = .95
    config['controller_jump_m'] = .045
    config['controller_jump_rad'] = .20
    config['hardware']['enabled'] = bool(hardware)
    config['hardware']['robot_ip'] = os.environ.get(
        'TIANJI_ROBOT_IP', config['hardware']['robot_ip'])
    config['hardware']['startup_move_enabled'] = False
    config['hardware']['servo_mode'] = 'impedance' if dual_arm else 'position'
    config['hardware']['impedance_params_before_enable'] = dual_arm
    config['hardware']['vel_ratio'] = 60 if dual_arm else 1
    config['hardware']['acc_ratio'] = 60 if dual_arm else 1
    if dual_arm:
        # The official chest controller has no Jacobian sigma-min cutoff.
        # DLS regularization, joint/coupling limits and speed caps stay active.
        config['hardware']['minimum_jacobian_singular_value'] = 0.
    if full_range:
        # Full-range still obeys the model joint/coupling limits and the
        # Cartesian/joint software speed bounds.
        config['hardware'].pop('joint_excursion_deg', None)
    else:
        config['hardware']['joint_excursion_deg'] = 3.
    config['hardware']['control_sides'] = ['left', 'right'] if side == 'both' else [side]
    if not dual_arm:
        config['hardware']['max_tracking_error_deg'] = 3.
        config['hardware']['tracking_error_timeout_s'] = .35
    for name in SIDES:
        config['arms'][name]['controller_to_wrist_m'] = calibration['offset_tracker_m'][name]
        config['arms'][name]['controller_to_wrist_xyzw'] = [0, 0, 0, 1]
    return config


def _robot_preflight(config, session_dir: Path):
    settings = config['hardware']
    report = diagnose(settings['robot_ip'], seconds=2.)
    path = session_dir / 'tianji-readonly.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if not report.get('ready_for_wrist_test'):
        raise RuntimeError(f'天机只读诊断未通过：{report.get("message") or report.get("error") or path}')
    config_file = session_dir / 'config.json'
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    gate = check_arm_pose(path, config_file)
    (session_dir / 'tianji-pose-gate.json').write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if not gate.get('ready_for_full_arm'):
        raise RuntimeError(f'天机整臂姿态门禁未通过：{gate.get("message")}')
    print('天机只读诊断与关节/腕部耦合限位通过；程序不会执行自动回初始位。')


def _frame(sample, seq: int, session: str, last_good: dict, *, baseline_limits_m=None):
    controllers = {}
    both_valid = _tracking_usable(sample)
    for side in SIDES:
        matrix = sample.poses[side]
        if matrix is not None:
            last_good[side] = matrix.copy()
        fallback = last_good.get(side)
        position = fallback[:3, 3].tolist() if fallback is not None else [0., 0., 0.]
        quaternion = (Rotation.from_matrix(fallback[:3, :3]).as_quat().tolist()
                      if fallback is not None else [0., 0., 0., 1.])
        controllers[side] = Controller(position, quaternion, bool(both_valid), 0.)
    return Frame(session=session, reference=0, seq=seq,
                 captured_s=sample.time_s, received_s=time.perf_counter(),
                 focused=True, controllers=controllers,
                 safety={'released': [], 'tracking_lost': not both_valid, 'focus_lost': False})


def _session_dir(kind: str):
    path = ROOT / 'logs' / f'vive-{kind}-{datetime.now().strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:6]}'
    path.mkdir(parents=True, exist_ok=False)
    return path


def _joint_gate_ready(state: dict, *, valid: bool, healthy: bool) -> bool:
    """Open the bilateral hand gate only while both physical arms follow."""
    arms = state.get('arms') or {}
    age = state.get('frame_age_ms')
    return bool(valid and healthy and state.get('hardware') is True and
                state.get('hardware_connected') and state.get('hardware_enabled') and
                state.get('source') == 'quest' and state.get('enabled') and
                state.get('keyboard_mode') and state.get('keyboard_follow') and
                state.get('following') and state.get('fresh') and
                not state.get('hardware_fault') and
                not state.get('tracking_loss_active') and
                not state.get('tracking_recovery_pending') and
                isinstance(age, (int, float)) and math.isfinite(age) and
                0 <= age < 120 and
                all(arms.get(side, {}).get('active') and arms.get(side, {}).get('tracked')
                    for side in SIDES))


def _set_joint_gate(path: Path | None, enabled: bool) -> None:
    if path is None:
        return
    if enabled:
        path.touch(exist_ok=True)
    else:
        path.unlink(missing_ok=True)


async def _heartbeat(control: ControlProcess, stop: asyncio.Event):
    while not stop.is_set():
        result = await control.command({'action': 'keyboard_heartbeat'}, source='keyboard')
        if not result.get('ok'):
            raise RuntimeError(f'键盘心跳失败：{result.get("error")}')
        try:
            await asyncio.wait_for(stop.wait(), .25)
        except asyncio.TimeoutError:
            pass


async def _run(args):
    expected_baseline = getattr(args, 'expected_baseline_m', None)
    baseline_limits = _baseline_limits(expected_baseline)
    uncalibrated_preview = bool(getattr(args, 'uncalibrated_preview', False))
    if uncalibrated_preview and args.mode != 'preview':
        raise ValueError('虚拟标定仅允许无实机 preview')
    raw_joint_gate = getattr(args, 'joint_gate', None)
    raw_joint_ready = getattr(args, 'joint_ready_file', None)
    if bool(raw_joint_gate) != bool(raw_joint_ready):
        raise ValueError('联合遥操须同时指定 --joint-gate 与 --joint-ready-file')
    joint_gate = Path(raw_joint_gate).expanduser() if raw_joint_gate else None
    joint_ready = Path(raw_joint_ready).expanduser() if raw_joint_ready else None
    if joint_gate is not None and (args.mode != 'live' or args.side != 'both'):
        raise ValueError('Wuji 双手门控只允许双臂实机 live 模式')
    if args.mode == 'live':
        from .joint_launcher import _quest_live
        if _quest_live():
            raise RuntimeError('Quest 实机会话仍在运行；请先退出 Quest，再启动 Vive 双臂')
    _set_joint_gate(joint_gate, False)
    session_dir = _session_dir('preview' if args.mode == 'preview' else 'live')
    summary = {'mode': args.mode, 'side': args.side, 'session_dir': str(session_dir),
               'started_at': datetime.now().astimezone().isoformat(),
               'samples': 0, 'valid_samples': 0, 'invalid': Counter(), 'commands': 0,
               'base_station_baseline_limits_m': list(baseline_limits),
               'test_only': uncalibrated_preview,
               'full_range': bool(getattr(args, 'full_range', False)),
               'expected_baseline_m': expected_baseline}
    control = None
    stop = asyncio.Event()
    heartbeat_task = None
    try:
        with TrackerPair(SERIALS['left'], SERIALS['right']) as pair:
            if joint_ready is not None and not joint_ready.is_file():
                raise RuntimeError('左右 Wuji Glove/Hand2 未就绪；双臂实机未连接')
            try:
                chaperone = pair.chaperone_status()
            except Exception as exc:
                chaperone = {'state': None, 'hmd_tracking_system': None,
                             'error': str(exc)}
            summary['chaperone'] = chaperone
            if chaperone.get('error'):
                print(f"SteamVR Chaperone 状态读取失败：{chaperone['error']}")
            else:
                print(f"SteamVR Chaperone={chaperone['state']}，"
                      f"HMD 驱动={chaperone['hmd_tracking_system']}")
            print('正在验证双腕和双基站连续定位 3 秒；此时不连接机器人。')
            # The final 0.75 s must remain continuous; the live loop separately
            # pauses when tracking is lost.
            samples, quality = _gather_startup(pair, baseline_limits)
            summary['base_station_baseline_m'] = samples[-1].base_station_baseline_m
            if args.mode == 'live':
                _check_live_chaperone(chaperone)
            calibration = (_uncalibrated_preview_calibration(pair) if uncalibrated_preview
                           else _read_calibration(pair, sample=samples[-1],
                                                  hardware=args.mode == 'live',
                                                  expected_baseline_m=expected_baseline))
            print(f'输入就绪 {quality * 100:.1f}%：' + _quality_line(samples[-1]))
            if uncalibrated_preview:
                print('仅预演：使用内存中的零手腕偏移/零偏航虚拟标定；'
                      '只检验相对运动与丢踪暂停，不评估真实末端精度。')
            config = _config(calibration, hardware=args.mode == 'live', side=args.side,
                             scale=args.scale, full_range=bool(getattr(args, 'full_range', False)))
            if args.mode == 'preview' and config['hardware']['enabled']:
                raise RuntimeError('预演配置异常：实机输出被启用')
            (session_dir / 'config.json').write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            if args.mode == 'live':
                _robot_preflight(config, session_dir)
                token = 'LIVE-RIGHT' if args.side == 'right' else 'LIVE-BOTH'
                print(f'即将建立天机 SDK 连接；仅在按 A 校准、再按空格后才会使能并移动受控臂。')
                if input(f'确认现场急停可用且运动范围净空，输入 {token} 后回车：').strip() != token:
                    raise RuntimeError('现场确认未完成；实机未启动')
            control = ControlProcess(config)
            await control.start()
            attached = await control.command({'action': 'keyboard_attach'}, source='keyboard')
            if not attached.get('ok'):
                raise RuntimeError(f'键盘控制未连接：{attached.get("error")}')
            heartbeat_task = asyncio.create_task(_heartbeat(control, stop), name='vive-keyboard-heartbeat')
            print('A：以当前实测臂姿态和双腕位姿对齐；空格：开始/暂停；B：暂停；Q：退出。')
            if args.mode == 'preview':
                print('当前是无实机预演，机械臂不会运动。')
            else:
                scope = ('模型允许的双臂关节范围' if getattr(args, 'full_range', False)
                         else '起始角 ±3°')
                print(f'实机无自动复位；初始位就是当前实测姿态。目标限制在{scope}，'
                      f"关节目标变化率 ≤{config['max_joint_speed_deg_s']:g}°/秒，"
                      f"SDK 速度/加速度 {config['hardware']['vel_ratio']}%，"
                      f"{'关节阻抗' if config['hardware']['servo_mode'] == 'impedance' else '位置'}模式。")
            started = time.perf_counter()
            next_sample = started
            last_status = 0.
            last_reason = None
            session = uuid.uuid4().hex
            last_good = {}
            seq = 0
            good_since = None
            preview_anchor = {}
            buffer = []
            log_path = session_dir / 'stream.jsonl'
            with log_path.open('w', encoding='utf-8') as log, UnixConsole() as keyboard:
                while True:
                    now = time.perf_counter()
                    if args.seconds and now - started >= args.seconds:
                        break
                    if joint_ready is not None and not joint_ready.is_file():
                        _set_joint_gate(joint_gate, False)
                        control.pause('左右 Wuji 手部通道已退出；双臂暂停')
                        raise RuntimeError('左右 Wuji 手部就绪信号消失；联合遥操已停止')
                    if heartbeat_task.done():
                        await heartbeat_task
                    if not control.healthy:
                        raise RuntimeError(control.fault or '控制进程不健康')
                    sample = pair.sample()
                    if sample.reference_changed:
                        control.pause('SteamVR 站立参考系已重置；必须重新标定')
                        raise RuntimeError('SteamVR 站立参考系已重置；停止本次会话，重新运行 calibrate')
                    if not calibration.get('test_only'):
                        try:
                            _station_geometry(sample, calibration['base_station_poses_standing'],
                                              require_both=False)
                        except RuntimeError as exc:
                            control.pause(str(exc))
                            raise
                    valid = _tracking_usable(sample)
                    good_since = sample.time_s if valid and good_since is None else good_since if valid else None
                    seq += 1
                    frame = _frame(sample, seq, session, last_good,
                                   baseline_limits_m=baseline_limits)
                    control.offer_frame(frame, 'quest')
                    summary['samples'] += 1
                    summary['valid_samples'] += bool(valid)
                    summary['base_station_baseline_m'] = sample.base_station_baseline_m
                    if set(sample.base_station_poses) != set(BASE_STATION_SERIALS):
                        summary['invalid']['station_reference_degraded'] += 1
                    if not valid:
                        for side in SIDES:
                            if sample.poses[side] is None:
                                summary['invalid'][side] += 1
                        if sample.base_station_count < 2:
                            summary['invalid']['base_stations'] += 1
                        if not _baseline_usable(sample, baseline_limits_m=baseline_limits):
                            summary['invalid']['base_station_geometry'] += 1
                    state = control.state or {}
                    for key in keyboard.poll():
                        if key == 'quit':
                            stop.set()
                            break
                        if key == 'reset':
                            command = {'action': 'pause'}
                        elif key == 'calibrate':
                            if not valid or good_since is None or sample.time_s - good_since < .75:
                                print('双腕有效追踪尚未持续 0.75 秒；拒绝对齐。')
                                continue
                            command = {'action': 'calibrate'}
                        elif key == 'keyboard_toggle':
                            if not state.get('keyboard_follow') and (not valid or good_since is None or
                                                                     sample.time_s - good_since < .75):
                                print('双腕有效追踪尚未持续 0.75 秒；拒绝开始/恢复。')
                                continue
                            command = {'action': 'keyboard_toggle'}
                        else:
                            continue
                        result = await control.command(command, source='keyboard')
                        summary['commands'] += 1
                        if key == 'calibrate' and result.get('ok'):
                            arms = (control.state or {}).get('arms', {})
                            preview_anchor = {
                                side: np.asarray(arms[side]['actual']['position'], dtype=float)
                                for side in config['hardware']['control_sides'] if side in arms
                            }
                        print(('操作成功' if result.get('ok') else f"操作未通过：{result.get('error')}") +
                              f"；{(control.state or {}).get('reason', '')}")
                    if stop.is_set():
                        break
                    state = control.state or {}
                    _set_joint_gate(joint_gate, _joint_gate_ready(
                        state, valid=valid and (joint_ready is None or joint_ready.is_file()),
                        healthy=control.healthy))
                    now = time.perf_counter()
                    reason = state.get('reason')
                    if reason != last_reason or now - last_status >= 2.:
                        delta_line = ''
                        if args.mode == 'preview' and preview_anchor:
                            parts = []
                            for side, start_position in preview_anchor.items():
                                arm = state.get('arms', {}).get(side, {})
                                pose = arm.get('actual')
                                if pose:
                                    delta_cm = (np.asarray(pose['position']) - start_position) * 100.
                                    parts.append(f'{side} 模拟末端 ΔX/Y/Z={delta_cm[0]:+.1f}/'
                                                 f'{delta_cm[1]:+.1f}/{delta_cm[2]:+.1f} cm')
                            if parts:
                                delta_line = ' | ' + '；'.join(parts)
                        baseline_text = (f'{sample.base_station_baseline_m:.3f} m'
                                         if sample.base_station_baseline_m is not None else '不可用')
                        print(f"{'实机' if args.mode == 'live' else '预演'} | "
                              f"追踪 {'双腕有效' if valid else '失效'} / 基站 {sample.base_station_count} "
                              f"/ 间距 {baseline_text} | "
                              f"{'跟随中' if state.get('following') else '保持/待机'} | {reason}{delta_line}")
                        last_reason, last_status = reason, now
                    record = {
                        't': sample.time_s, 'seq': seq, 'valid': valid,
                        'base_stations': sample.base_station_count,
                        'base_station_baseline_m': sample.base_station_baseline_m,
                        'tracker': {side: {'position_m': sample.poses[side][:3, 3].tolist(),
                                           'quaternion_xyzw': Rotation.from_matrix(
                                               sample.poses[side][:3, :3]).as_quat().tolist()}
                                    if sample.poses[side] is not None else None for side in SIDES},
                    }
                    if seq % 5 == 0:
                        record['control'] = {
                            'following': bool(state.get('following')),
                            'reason': reason, 'frame_age_ms': state.get('frame_age_ms'),
                            'queue_delay_ms': state.get('queue_delay_ms'),
                            'joints_deg': {side: state.get('arms', {}).get(side, {}).get('joints_deg')
                                           for side in SIDES},
                            'measured_joints_deg': {side: state.get('arms', {}).get(side, {}).get('measured_joints_deg')
                                                    for side in SIDES},
                            'tracking_error_deg': {side: state.get('arms', {}).get(side, {}).get('tracking_error_deg')
                                                   for side in SIDES},
                            'target_position_m': {side: state.get('arms', {}).get(side, {}).get('target', {}).get('position')
                                                  for side in SIDES},
                            'fk_position_m': {side: state.get('arms', {}).get(side, {}).get('actual', {}).get('position')
                                              for side in SIDES},
                            'measured_position_m': {side: (state.get('arms', {}).get(side, {}).get('measured_pose') or {}).get('position')
                                                    for side in SIDES},
                        }
                    buffer.append(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                    if len(buffer) >= 90:
                        log.writelines(buffer)
                        buffer.clear()
                    next_sample += 1. / SAMPLE_HZ
                    await asyncio.sleep(max(0., next_sample - time.perf_counter()))
                    if time.perf_counter() - next_sample > .05:
                        next_sample = time.perf_counter()
                if buffer:
                    log.writelines(buffer)
            summary['final_state'] = {'reason': (control.state or {}).get('reason'),
                                      'hardware_fault': (control.state or {}).get('hardware_fault'),
                                      'diagnostics': (control.state or {}).get('diagnostics')}
    except BaseException as exc:
        summary['error'] = str(exc)
        if isinstance(exc, TrackingGateError):
            summary['tracking_gate'] = exc.diagnostics
        raise
    finally:
        shutdown_errors = []
        try:
            _set_joint_gate(joint_gate, False)
        except OSError as exc:
            shutdown_errors.append(f'Wuji 手部门控未能关闭: {exc}')
        stop.set()
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                shutdown_errors.append(f'键盘心跳: {exc}')
        if control is not None:
            if control.healthy:
                try:
                    result = await asyncio.wait_for(
                        control.command({'action': 'pause'}, source='keyboard'), 2.)
                    if not result.get('ok'):
                        raise RuntimeError(result.get('error') or '暂停命令被拒绝')
                except Exception as exc:
                    shutdown_errors.append(f'暂停未确认: {exc}')
            elif args.mode == 'live':
                shutdown_errors.append('控制进程故障，无法确认已保持当前位置')
            try:
                stop_reply = await control.close()
                summary['stop_ack'] = stop_reply
                if stop_reply and stop_reply.get('pause_error'):
                    shutdown_errors.append(f"暂停时出现故障: {stop_reply['pause_error']}")
            except Exception as exc:
                shutdown_errors.append(f'待机未确认: {exc}')
        summary['ended_at'] = datetime.now().astimezone().isoformat()
        summary['invalid'] = dict(summary['invalid'])
        if summary['samples']:
            summary['valid_ratio'] = summary['valid_samples'] / summary['samples']
        if shutdown_errors:
            summary['shutdown_errors'] = shutdown_errors
        (session_dir / 'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'测试记录：{session_dir}')
        if shutdown_errors:
            raise RuntimeError('会话停机未确认：' + '；'.join(shutdown_errors) +
                               '。请现场核对机械臂状态，必要时使用急停。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    calibration_command = sub.add_parser('calibrate', help='只读标定双腕 Tracker 偏移与操作员朝向')
    calibration_command.add_argument('--expected-baseline-m', type=float,
                                     help='本次实测两基站中心距，允许 ±0.020 m')
    for side, label in (('left', '左'), ('right', '右')):
        calibration_command.add_argument(
            f'--{side}-offset-mm', type=float, nargs=3,
            metavar=('X', 'Y', 'Z'),
            help=f'可选：{label}腕实测从 VIVE Tracker 3.0 安装螺孔基准到解剖手腕中心的'
                 'OpenVR Tracker 局部坐标有符号 X/Y/Z 毫米；提供后跳过该腕 pivot 拟合')
    for mode in ('preview', 'live'):
        command = sub.add_parser(mode, help='无实机预演' if mode == 'preview' else '天机臂实机遥操（默认低速）')
        command.add_argument('--side', choices=('right', 'both'), default='both' if mode == 'preview' else 'right')
        command.add_argument('--scale', type=float, default=1., help='手部位移到机械臂位移比例；默认 1:1，精细模式可设 0.5')
        command.add_argument('--seconds', type=float, default=0., help='自动结束时长；0 表示按 Q 退出')
        command.add_argument('--expected-baseline-m', type=float,
                             help='本次实测两基站中心距，允许 ±0.020 m；须与双腕标定一致')
        if mode == 'live':
            command.add_argument('--full-range', action='store_true',
                                 help='双臂按模型关节范围跟随，取消起始 ±3°包络；保留软件限速及丢踪暂停')
            command.add_argument('--joint-gate', type=Path,
                                 help='联合模式：仅双臂有效跟随时更新 Wuji Hand2 门控文件')
            command.add_argument('--joint-ready-file', type=Path,
                                 help='联合模式：左右 Wuji 手部新鲜帧就绪文件')
        if mode == 'preview':
            command.add_argument('--uncalibrated-preview', action='store_true',
                                 help='仅预演：使用不保存的零偏移虚拟标定；不能用于实机')
    args = parser.parse_args()
    if args.mode == 'calibrate':
        try:
            _baseline_limits(args.expected_baseline_m)
            calibrate(args)
            return 0
        except KeyboardInterrupt:
            print('双腕标定已取消；旧标定文件未改动。', file=sys.stderr)
            return 130
        except Exception as exc:
            print(f'双腕标定未完成：{exc}', file=sys.stderr)
            return 2
    if not .2 <= args.scale <= 1.:
        parser.error('--scale 必须在 0.2–1.0 之间')
    if not 0 <= args.seconds <= 3600:
        parser.error('--seconds 必须在 0–3600 秒之间')
    try:
        asyncio.run(_run(args))
        return 0
    except KeyboardInterrupt:
        print('已请求结束并让控制进程恢复待机；请核对机械臂现场状态。')
        return 130
    except Exception as exc:
        print(f'测试未启动或已停止：{exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
