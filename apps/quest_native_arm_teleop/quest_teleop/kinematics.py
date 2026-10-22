"""Only the mathematical vendor library is loaded. No fx_robot import or connection."""
from __future__ import annotations

import importlib.util
import logging
import numpy as np
from scipy.spatial.transform import Rotation

from .config import SDK_ROOT
from .math3d import mdh_chain, rigid_inverse
from .continuous_ik import solve_local


class IKError(ValueError):
    pass


class Kinematics:
    def __init__(self, config):
        self.config = config
        wrapper = SDK_ROOT / 'SDK_PYTHON/fx_kine.py'
        if not wrapper.is_file():
            raise FileNotFoundError(f'缺少厂家运动学 SDK: {wrapper}')
        spec = importlib.util.spec_from_file_location('_quest_vendor_kinematics', wrapper)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        logging.getLogger('debug_printer').disabled = True
        self.arms = {}
        for index, side in enumerate(('left', 'right')):
            sdk = self.module.Marvin_Kine()
            sdk.log_switch(0)
            data = sdk.load_config(index, str(SDK_ROOT / 'CommonConfig' / f"{config['model']}.MvKDCfg"))
            if not data or not sdk.initial_kine(data['TYPE'][index], data['DH'][index], data['PNVA'][index], data['BD'][index]):
                raise RuntimeError(f'{side}: 运动学初始化失败')
            arm_cfg = config['arms'][side]
            base = np.eye(4)
            base[:3, :3] = Rotation.from_euler('xyz', arm_cfg['base_rotation_xyz_deg'], degrees=True).as_matrix()
            base[:3, 3] = arm_cfg['base_position_m']
            self.arms[side] = {'sdk': sdk, 'base': base, 'dh': data['DH'][index],
                               'limits': np.sort(np.array(data['PNVA'][index])[:, :2], axis=1),
                               'wrist_bounds': np.asarray(data['BD'][index], dtype=float)}
            # Fail visibly if an unsupported DH convention would draw the wrong arm.
            q = np.asarray(arm_cfg['initial_joints_deg'], dtype=float)
            if not np.allclose(self.chain(side, q)[-1], self.fk(side, q), atol=1e-6):
                raise RuntimeError(f'{side}: 连杆模型与 SDK 正解不一致')
        self.visual = None
        if config.get('visual_model'):
            from .urdf_model import UrdfVisualModel
            self.visual = UrdfVisualModel(config, self)
        for side in self.arms:
            if not self.valid_joints(side, config['arms'][side]['initial_joints_deg']):
                raise ValueError(f'{side}: 初始姿态超出关节或腕部耦合限位')

    def fk(self, side, joints_deg):
        arm = self.arms[side]
        value = arm['sdk'].fk(np.asarray(joints_deg).tolist())
        if value is False:
            raise IKError('正解失败')
        value = np.asarray(value, dtype=float)
        value[:3, 3] /= 1000.
        return arm['base'] @ value

    def chain(self, side, joints_deg):
        arm = self.arms[side]
        return [arm['base'] @ p for p in mdh_chain(arm['dh'], joints_deg)]

    def jacobian(self, side, joints_deg):
        """World spatial Jacobian, translation m/rad and rotation rad/rad."""
        arm = self.arms[side]
        value = arm['sdk'].joints2JacobMatrix(np.asarray(joints_deg).tolist())
        if value is False:
            raise IKError('雅可比计算失败')
        j = np.asarray(value, dtype=float)
        if j.shape != (6, 7) or not np.isfinite(j).all():
            raise IKError('雅可比数据无效')
        rotation = arm['base'][:3, :3]
        return np.vstack((rotation @ j[:3], rotation @ j[3:]))

    def _wrist_faces(self, side, q_deg):
        """SDK FxRobot.cpp: Joint67Limit; BD is a0*j6^2+a1*j6+a2 (degrees)."""
        bd = self.arms[side]['wrist_bounds']
        j6 = float(q_deg[5])
        if not np.any(bd) or abs(j6) <= 1.:
            return []
        upper, lower = (0, 3) if j6 > 1. else (1, 2)
        faces = []
        for index, sign in ((upper, 1.), (lower, -1.)):
            a0, a1, a2 = bd[index]
            bound = a0 * j6 * j6 + a1 * j6 + a2
            derivative = 2 * a0 * j6 + a1
            row = np.zeros(7)
            row[5], row[6] = -sign * derivative, sign
            faces.append((row, np.deg2rad(sign * (bound - q_deg[6]))))
        return faces

    def valid_joints(self, side, joints_deg):
        q = np.asarray(joints_deg, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            return False
        limits = self.arms[side]['limits']
        return bool(np.all(q >= limits[:, 0] - 1e-7) and np.all(q <= limits[:, 1] + 1e-7)
                    and all(slack >= -1e-9 for _, slack in self._wrist_faces(side, q)))

    def _local(self, side, world_target, reference_deg, max_delta_deg, iterations):
        target = np.asarray(world_target, dtype=float)
        if (target.shape != (4, 4) or not np.isfinite(target).all() or
                not np.allclose(target[3], [0, 0, 0, 1], atol=1e-7) or
                not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-6) or
                np.linalg.det(target[:3, :3]) < .999):
            raise IKError('IK 目标位姿无效')
        if not self.valid_joints(side, reference_deg):
            raise IKError('当前关节超出限位或腕部耦合范围')
        if not np.isfinite(max_delta_deg) or max_delta_deg <= 0:
            raise IKError('IK 单步关节范围无效')

        def fk_jacobian(q):
            degrees = np.rad2deg(q)
            return self.fk(side, degrees), self.jacobian(side, degrees)

        def constraints_at(q):
            faces = self._wrist_faces(side, np.rad2deg(q))
            return (np.asarray([a for a, _ in faces]).reshape(-1, 7), np.array([b for _, b in faces]))

        result = solve_local(target, reference_deg, fk_jacobian, constraints_at,
                             lambda q: self.valid_joints(side, q), self.arms[side]['limits'],
                             max_delta_deg=max_delta_deg, max_iterations=iterations,
                             position_tolerance_m=self.config['ik_position_tolerance_m'],
                             rotation_tolerance_rad=self.config['ik_rotation_tolerance_rad'],
                             rotation_weight_m=self.config['ik_rotation_weight_m'],
                             damping=self.config['ik_damping'], centering=self.config['ik_centering'])
        if not self.valid_joints(side, result.joints_deg):
            raise IKError('IK 返回关节超出限位或腕部耦合范围')
        return result

    def track(self, side, world_target, reference_deg, max_delta_deg):
        """Feasible local progress, including at singularities/unreachable targets.

        A bounded partial result is reported to the UI, never passed off as exact IK.
        """
        return self._local(side, world_target, reference_deg, max_delta_deg, self.config['ik_max_iterations'])

    def solve(self, side, world_target, reference_deg):
        result = self._local(side, world_target, reference_deg, 180., 64)
        if result.converged:
            return result.joints_deg
        # Preserve exact-pose offline calls. The interactive tracker never switches
        # to a distant analytic branch when a local, constrained step is unavailable.
        return self._sdk_solve(side, world_target, reference_deg)

    def _sdk_solve(self, side, world_target, reference_deg):
        arm = self.arms[side]
        target = rigid_inverse(arm['base']) @ world_target
        target[:3, 3] *= 1000.
        sp = self.module.FX_InvKineSolvePara()
        sp.set_input_ik_target_tcp(target.reshape(-1).tolist())
        sp.set_input_ik_ref_joint(np.asarray(reference_deg).tolist())
        # No elbow sensors: choose the solution closest to the previous joint state.
        sp.set_input_ik_zsp_type(0)
        result = arm['sdk'].ik(sp)
        if result is False or result.m_Output_IsOutRange:
            raise IKError('目标不可达 / IK 无解')
        if result.m_Output_IsJntExd:
            raise IKError('关节超出限位')
        q = np.array(result.m_Output_RetJoint.to_list())
        if not np.isfinite(q).all():
            raise IKError('IK 返回非有限值')
        limits = arm['limits']
        if np.any(q < limits[:, 0] - 1e-5) or np.any(q > limits[:, 1] + 1e-5):
            raise IKError('关节超出配置限位')
        if not self.valid_joints(side, q):
            raise IKError('关节超出腕部耦合范围')
        # A numerical solution is not enough; reject inaccurate solutions too.
        recovered = self.fk(side, q)
        if np.linalg.norm(recovered[:3, 3] - world_target[:3, 3]) > 0.002:
            raise IKError('IK 位置残差过大')
        if Rotation.from_matrix(recovered[:3, :3] @ world_target[:3, :3].T).magnitude() > 0.02:
            raise IKError('IK 姿态残差过大')
        return q
