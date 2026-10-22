"""Import the supplied M6S Lite URDF after validating every SDK link/TCP frame.

The manufacturer's URDF rounds angles to four decimals. Rendering uses the
matching, unrounded SDK frames after validation; original assets stay untouched.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

from .config import SDK_ROOT
from .math3d import mdh_chain, rotation_angle, serialized_pose

VISUAL_ID = 'marvin_m6s_lite_615_v1'
# The vendor retains a historical M3 name for this Lite configuration file.
SDK_MODEL = 'ccs_m3_2.0_0510'
URDF_ROOT = SDK_ROOT / 'urdf'


def numbers(text, size, default):
    values = np.asarray([float(v) for v in (text or default).split()])
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError('URDF 中的数值或维度无效')
    return values


def origin_matrix(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = numbers(element.get('xyz'), 3, '0 0 0')
        transform[:3, :3] = Rotation.from_euler('xyz', numbers(element.get('rpy'), 3, '0 0 0')).as_matrix()
    return transform


class UrdfArm:
    def __init__(self, path: Path, tag: str):
        self.path = path
        xml = ET.parse(path).getroot()
        self.names = [f'Base_{tag}'] + [f'Link{i}_{tag}' for i in range(1, 8)] + [f'TCP_Link_{tag}']
        self.links = {link.get('name'): link for link in xml.findall('link')}
        joint_nodes = {joint.get('name'): joint for joint in xml.findall('joint')}
        if set(self.links) != set(self.names) or len(joint_nodes) != 8:
            raise ValueError(f'{path.name}: 不是预期的 M6S Lite 七关节及固定 TCP 模型')
        self.joints = []
        for index in range(7):
            joint = joint_nodes.get(f'Joint{index + 1}_{tag}')
            if (joint is None or joint.get('type') != 'revolute' or
                    joint.find('parent').get('link') != self.names[index] or
                    joint.find('child').get('link') != self.names[index + 1]):
                raise ValueError(f'{path.name}: 关节名称、类型或连接顺序不符')
            axis = numbers(joint.find('axis').get('xyz'), 3, '0 0 1')
            if not np.isclose(np.linalg.norm(axis), 1):
                raise ValueError('URDF 关节轴不是单位向量')
            limit = joint.find('limit')
            bounds = numbers(f"{limit.get('lower')} {limit.get('upper')}", 2, '')
            if bounds[0] >= bounds[1]:
                raise ValueError('URDF 关节限位无效')
            self.joints.append({'origin': origin_matrix(joint.find('origin')), 'axis': axis,
                                'limits_deg': np.rad2deg(bounds)})
        tcp = joint_nodes.get(f'JointTCP_{tag}')
        if (tcp is None or tcp.get('type') != 'fixed' or
                tcp.find('parent').get('link') != self.names[7] or
                tcp.find('child').get('link') != self.names[8]):
            raise ValueError('URDF 的固定 TCP 连接不符')
        self.tcp_origin = origin_matrix(tcp.find('origin'))

    def chain(self, joints_deg):
        transform = np.eye(4)
        chain = [transform.copy()]
        for joint, angle in zip(self.joints, joints_deg, strict=True):
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(joint['axis'] * np.deg2rad(angle)).as_matrix()
            transform = transform @ joint['origin'] @ turn
            chain.append(transform.copy())
        chain.append(transform @ self.tcp_origin)
        return chain

    def mesh_path(self, filename):
        # Resolve ROS package:// references inside this arm's supplied package.
        # The exported package name and directory name differ; the package root
        # is the parent of urdf/, not a globally installed ROS package.
        if not filename.startswith('package://'):
            raise ValueError('此 Lite 导入器只接受 package:// 网格引用')
        parts = filename[len('package://'):].split('/', 1)
        if len(parts) != 2:
            raise ValueError('URDF 网格路径无效')
        package_root = self.path.parent.parent.resolve()
        path = (package_root / parts[1]).resolve()
        if not path.is_relative_to(package_root) or path.suffix.lower() != '.stl' or not path.is_file():
            raise ValueError(f'URDF 网格缺失或越界：{filename}')
        return path


class UrdfVisualModel:
    def __init__(self, config, kine):
        if config.get('visual_model') != VISUAL_ID or config['model'] != SDK_MODEL:
            raise ValueError('M6S Lite V1.0 URDF 必须配合 ccs_m3_2.0_0510 运动学参数')
        self.meshes = {}
        self.arms = {}
        self.data = {'kind': 'urdf', 'id': VISUAL_ID, 'model': SDK_MODEL,
                     'label': 'M6S Lite CCS-615 · URDF V1.0', 'arms': {}}
        for side, tag in (('left', 'L'), ('right', 'R')):
            stem = f'Marvin M6S-Lite-{tag}-CCS-615-V1.0 urdf'
            path = URDF_ROOT / stem / 'urdf' / f'{stem}.urdf'
            arm = UrdfArm(path, tag)
            self.arms[side] = arm
            self.validate_frames(side, config, kine)
            limits = np.asarray([j['limits_deg'] for j in arm.joints])
            sdk_limits = kine.arms[side]['limits']
            # Respect the stricter bound where the URDF and SDK differ.
            limits[:, 0] = np.maximum(limits[:, 0], sdk_limits[:, 0])
            limits[:, 1] = np.minimum(limits[:, 1], sdk_limits[:, 1])
            initial = np.asarray(config['arms'][side]['initial_joints_deg'])
            if np.any(limits[:, 0] > limits[:, 1]) or np.any(initial < limits[:, 0]) or np.any(initial > limits[:, 1]):
                raise ValueError('初始关节角超出 URDF / SDK 共同限位')
            kine.arms[side]['limits'] = limits
            links = []
            for index, name in enumerate(arm.names):
                visuals = []
                for visual_index, node in enumerate(arm.links[name].findall('visual')):
                    mesh = node.find('geometry/mesh')
                    if mesh is None:
                        raise ValueError('此 Lite 模型的 visual 必须为 STL mesh')
                    mesh_id = f'{side}-{index}-{visual_index}'
                    self.meshes[mesh_id] = arm.mesh_path(mesh.get('filename', ''))
                    scale = numbers(mesh.get('scale'), 3, '1 1 1')
                    if np.any(scale <= 0):
                        raise ValueError('URDF mesh scale 必须为正')
                    visuals.append({'mesh_url': f'/robot-mesh/{mesh_id}',
                                    'origin': serialized_pose(origin_matrix(node.find('origin'))),
                                    'scale': scale.tolist()})
                links.append({'name': name, 'chain_index': index, 'visuals': visuals})
            # TCP_Link matches the SDK flange, including its 87 mm fixed offset.
            flange = mdh_chain(kine.arms[side]['dh'][-1:], [])[1]
            self.data['arms'][side] = {'urdf': path.name, 'links': links,
                                       'flange_from_link7': serialized_pose(flange)}

    def validate_frames(self, side, config, kine):
        arm = self.arms[side]
        probes = [np.zeros(7), np.asarray(config['arms'][side]['initial_joints_deg'])]
        probes.extend(np.eye(7) * 25.)
        max_position, max_rotation = 0., 0.
        for q in probes:
            expected = mdh_chain(kine.arms[side]['dh'], q)
            for urdf, sdk in zip(arm.chain(q), expected, strict=True):
                pos = float(np.linalg.norm(urdf[:3, 3] - sdk[:3, 3]))
                rot = rotation_angle(urdf[:3, :3], sdk[:3, :3])
                max_position, max_rotation = max(max_position, pos), max(max_rotation, rot)
                if pos > 5e-5 or rot > 5e-5:
                    raise ValueError(f'{side}: URDF 与 SDK 连杆坐标不匹配 ({pos:.6f} m, {rot:.6f} rad)')
        self.data.setdefault('validation', {})[side] = {
            'max_position_error_m': max_position, 'max_rotation_error_rad': max_rotation,
            'poses_checked': len(probes)}
