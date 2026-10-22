"""Right-handed rigid transforms. SI units except explicit *_deg and MDH rows."""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation

# WebXR: +X right, +Y up, -Z forward. Robot world: +X forward, +Y left, +Z up.
XR_TO_WORLD = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])


def heading_basis(quaternion_xyzw, operator_yaw_deg=0.):
    """Freeze the viewer's horizontal forward direction as robot +X at A.

    Ignore head pitch/roll to preserve world up. Callers retain this basis until
    recalibration; following head turns must not change the movement mapping.
    """
    forward = Rotation.from_quat(quaternion_xyzw).apply([0., 0., -1.])
    if np.linalg.norm(forward[[0, 2]]) < .2:
        raise ValueError('请平视操作正前方后按 A；抬头或低头幅度过大')
    world_forward = XR_TO_WORLD @ forward
    heading = math.atan2(world_forward[1], world_forward[0])
    basis = Rotation.from_euler('z', math.radians(operator_yaw_deg) - heading).as_matrix() @ XR_TO_WORLD
    return basis, math.degrees(heading)


def pose_matrix(position, quaternion_xyzw):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    result[:3, 3] = position
    return result


def rigid_inverse(matrix):
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def rotation_angle(a, b):
    return float(np.linalg.norm(Rotation.from_matrix(a @ b.T).as_rotvec()))


def relative_target(controller, controller_anchor, robot_anchor, basis, scale):
    """Fixed-world delta, applied around the anchored robot tool position."""
    out = robot_anchor.copy()
    out[:3, 3] += scale * basis @ (controller[:3, 3] - controller_anchor[:3, 3])
    delta = controller[:3, :3] @ controller_anchor[:3, :3].T
    out[:3, :3] = basis @ delta @ basis.T @ robot_anchor[:3, :3]
    return out


def smooth_target(current, target, dt, tau, max_speed, max_angular_speed):
    alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))
    result = current.copy()
    delta = (target[:3, 3] - current[:3, 3]) * alpha
    size = np.linalg.norm(delta)
    if size > max_speed * dt:
        delta *= max_speed * dt / size
    result[:3, 3] += delta
    rotvec = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec() * alpha
    size = np.linalg.norm(rotvec)
    if size > max_angular_speed * dt:
        rotvec *= max_angular_speed * dt / size
    result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix() @ current[:3, :3]
    return result


def mdh_chain(dh, joints_deg):
    """Craig modified DH, millimetres/degrees in; metres out. Includes flange row."""
    transform = np.eye(4)
    chain = [transform.copy()]
    for index, (alpha, a, d, theta) in enumerate(dh):
        alpha = math.radians(alpha)
        theta = math.radians(theta + (joints_deg[index] if index < len(joints_deg) else 0.0))
        ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
        link = np.array([[ct, -st, 0., a / 1000],
                         [st * ca, ct * ca, -sa, -d * sa / 1000],
                         [st * sa, ct * sa, ca, d * ca / 1000],
                         [0., 0., 0., 1.]])
        transform = transform @ link
        chain.append(transform.copy())
    return chain


def serialized_pose(matrix):
    return {'position': matrix[:3, 3].tolist(),
            'quaternion': Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist()}
