"""Adaptive pose filtering matching Wuji's PICO incremental controller.

The implementation is deliberately ROS-independent.  It uses the actual
control-loop ``dt`` rather than assuming that WebXR input and robot output
arrive at exactly the same rate.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation


def _alpha(dt: float, cutoff_hz: float) -> float:
    dt = min(max(float(dt), 1e-4), .1)
    tau = 1. / (2. * math.pi * max(float(cutoff_hz), 1e-6))
    return dt / (dt + tau)


class OneEuroVector:
    def __init__(self, min_cutoff_hz=1., beta=0., derivative_cutoff_hz=1.):
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.beta = float(beta)
        self.derivative_cutoff_hz = float(derivative_cutoff_hz)
        self.reset()

    def reset(self):
        self.value = None
        self.derivative = None

    def __call__(self, value, dt):
        value = np.asarray(value, dtype=float)
        if self.value is None:
            self.value = value.copy()
            self.derivative = np.zeros_like(value)
            return value.copy()
        derivative = (value - self.value) / min(max(float(dt), 1e-4), .1)
        a_d = _alpha(dt, self.derivative_cutoff_hz)
        self.derivative += a_d * (derivative - self.derivative)
        cutoff = self.min_cutoff_hz + self.beta * float(np.linalg.norm(self.derivative))
        self.value += _alpha(dt, cutoff) * (value - self.value)
        return self.value.copy()


class OneEuroQuaternion:
    def __init__(self, min_cutoff_hz=1., beta=0., derivative_cutoff_hz=1.):
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.beta = float(beta)
        self.derivative_cutoff_hz = float(derivative_cutoff_hz)
        self.reset()

    def reset(self):
        self.quaternion = None
        self.speed = 0.

    def __call__(self, quaternion_xyzw, dt):
        q = np.asarray(quaternion_xyzw, dtype=float)
        q /= np.linalg.norm(q)
        if self.quaternion is None:
            self.quaternion = q.copy()
            return q.copy()
        if float(np.dot(q, self.quaternion)) < 0.:
            q = -q
        dot = float(np.clip(np.dot(q, self.quaternion), -1., 1.))
        angular_speed = 2. * math.acos(abs(dot)) / min(max(float(dt), 1e-4), .1)
        a_d = _alpha(dt, self.derivative_cutoff_hz)
        self.speed += a_d * (angular_speed - self.speed)
        cutoff = self.min_cutoff_hz + self.beta * self.speed
        amount = _alpha(dt, cutoff)
        if dot > .9995:
            result = self.quaternion + amount * (q - self.quaternion)
        else:
            angle = math.acos(dot)
            result = (math.sin((1. - amount) * angle) * self.quaternion +
                      math.sin(amount * angle) * q) / math.sin(angle)
        self.quaternion = result / np.linalg.norm(result)
        return self.quaternion.copy()


class OneEuroPose:
    """Filter translation and orientation with Wuji's signal-domain scales."""
    def __init__(self, settings):
        common = (settings['min_cutoff_hz'], settings['derivative_cutoff_hz'])
        self.position = OneEuroVector(common[0], settings['beta'] * settings['position_beta_scale'], common[1])
        self.rotation = OneEuroQuaternion(common[0], settings['beta'] * settings['rotation_beta_scale'], common[1])

    def reset(self):
        self.position.reset()
        self.rotation.reset()

    def __call__(self, pose, dt):
        result = np.asarray(pose, dtype=float).copy()
        result[:3, 3] = self.position(result[:3, 3], dt)
        quaternion = self.rotation(Rotation.from_matrix(result[:3, :3]).as_quat(), dt)
        result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        return result
