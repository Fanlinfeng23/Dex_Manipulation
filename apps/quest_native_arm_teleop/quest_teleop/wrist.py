"""Restricted commissioning: only J6/J7 may change, around a fixed startup pose."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

SIDES = ('left', 'right')


class WristEnvelope:
    def __init__(self, joints, settings):
        self.baseline = {s: np.asarray(joints[s], dtype=float).copy() for s in SIDES}
        self.excursion = settings['wrist_max_excursion_deg']
        self.locked_tolerance = settings['locked_joint_tolerance_deg']
        if any(q.shape != (7,) or not np.isfinite(q).all() for q in self.baseline.values()):
            raise ValueError('腕部限制需要有效的双臂初始反馈')

    def hold_target(self, side, measured):
        self.check_feedback({side: measured})
        q = np.asarray(measured, dtype=float).copy()
        q[:5] = self.baseline[side][:5]
        q[5:] = np.clip(q[5:], self.baseline[side][5:] - self.excursion,
                       self.baseline[side][5:] + self.excursion)
        return q

    def check_feedback(self, joints):
        for side, q in joints.items():
            q = np.asarray(q, dtype=float)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise ValueError('腕部限制收到无效反馈')
            delta = np.abs(q - self.baseline[side])
            if np.any(delta[:5] > self.locked_tolerance):
                raise ValueError(f'{side}: J1–J5 实测偏移超过 {self.locked_tolerance}°，停止手腕测试')
            if np.any(delta[5:] > self.excursion + self.locked_tolerance):
                raise ValueError(f'{side}: 手腕实测超出启动姿态限幅')

    def check_commands(self, joints):
        for side in SIDES:
            q = np.asarray(joints[side], dtype=float)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise ValueError('腕部目标无效')
            if not np.allclose(q[:5], self.baseline[side][:5], atol=1e-8, rtol=0):
                raise ValueError(f'{side}: 手腕模式拒绝 J1–J5 运动指令')
            if np.any(np.abs(q[5:] - self.baseline[side][5:]) > self.excursion + 1e-8):
                raise ValueError(f'{side}: 手腕目标超过 ±{self.excursion}°')

    def track(self, kine, side, desired, current, max_step):
        """Two-column orientation solve; translation never enters this solver."""
        q = self.hold_target(side, current)
        pose = kine.fk(side, q)
        error = Rotation.from_matrix(desired[:3, :3] @ pose[:3, :3].T).as_rotvec()
        jac = kine.jacobian(side, q)[3:, 5:7]
        if jac.shape != (3, 2) or not np.isfinite(jac).all():
            raise ValueError('腕部雅可比无效')
        delta = np.rad2deg(np.linalg.solve(jac.T @ jac + np.eye(2) * 1e-4, jac.T @ error))
        size = float(np.max(np.abs(delta)))
        limited = size > max_step
        if limited:
            delta *= max_step / size
        candidate = q.copy()
        candidate[5:] = np.clip(q[5:] + delta, self.baseline[side][5:] - self.excursion,
                               self.baseline[side][5:] + self.excursion)
        limited |= not np.allclose(candidate[5:], q[5:] + delta, atol=1e-9, rtol=0)
        # Retain the model's non-rectangular J6/J7 coupling limits as well.
        for _ in range(12):
            if kine.valid_joints(side, candidate):
                break
            candidate = q + .5 * (candidate - q)
            limited = True
        else:
            candidate = q
            limited = True
        return SimpleNamespace(joints_deg=candidate, limited=bool(limited), iterations=1)
