#!/usr/bin/env python3
"""Deterministic kinematic evaluation; never constructs a robot connection."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
from quest_teleop.config import load_config
from quest_teleop.engine import TeleopEngine
from quest_teleop.kinematics import Kinematics
from quest_teleop.math3d import rotation_angle
from quest_teleop.protocol import parse_frame


def evaluate(config):
    kine = Kinematics(config)
    output = []
    scenarios = [('forward_40mm', [0, 0, -.04], [0, 0, 0]),
                 ('right_40mm', [.04, 0, 0], [0, 0, 0]),
                 ('up_40mm', [0, .04, 0], [0, 0, 0]),
                 ('yaw_10deg', [0, 0, 0], [0, 10, 0]),
                 ('combined', [.015, .02, -.03], [5, 7, -5])]
    for name, position, angles in scenarios:
        now = 10.
        engine = TeleopEngine(config, kine, clock=lambda: now)
        engine.set_source('quest')
        data = dict(version=1, type='frame', session='synthetic-evaluation', reference=0,
                    seq=0, captured_ms=10000., focused=True,
                    head={'quaternion': [0, 0, 0, 1], 'tracked': True},
                    controllers={s: {'position': [0, 1.2, -.4], 'quaternion': [0, 0, 0, 1],
                                     'tracked': True, 'grip': 0} for s in ('left', 'right')})
        def feed():
            nonlocal now
            now += 1 / config['control_hz']; data['seq'] += 1; data['captured_ms'] = now * 1000
            engine.receive(parse_frame(copy.deepcopy(data), now), 'quest')
            t = time.perf_counter(); engine.step(); return (time.perf_counter() - t) * 1000
        feed(); engine.calibrate()
        errors, rotations, timings, speeds = [], [], [], []
        valid, limited = True, 0
        for i in range(451):
            u = min(i / 270, 1.)
            u = u ** 3 * (10 - 15 * u + 6 * u * u)
            for c in data['controllers'].values():
                c['position'] = (np.array([0, 1.2, -.4]) + u * np.array(position)).tolist()
                c['quaternion'] = Rotation.from_euler('xyz', u * np.array(angles), degrees=True).as_quat().tolist()
                c['grip'] = 1
            previous = {s: q.copy() for s, q in engine.q.items()}
            timings.append(feed())
            if not engine.enabled: raise RuntimeError(f'{name}: {engine.reason}')
            for s in ('left', 'right'):
                errors.append(np.linalg.norm(engine.target[s][:3, 3] - engine.actual[s][:3, 3]) * 1000)
                rotations.append(np.degrees(rotation_angle(engine.target[s][:3, :3], engine.actual[s][:3, :3])))
                speeds.append(np.max(np.abs(engine.q[s] - previous[s])) * config['control_hz'])
                valid &= kine.valid_joints(s, engine.q[s])
                limited += int(engine.ik_feedback[s]['limited'])
        output.append({'scenario': name, 'position_rmse_mm': float(np.sqrt(np.mean(np.square(errors)))),
                       'position_peak_mm': float(max(errors)), 'settled_position_mm': float(max(errors[-2:])),
                       'orientation_peak_deg': float(max(rotations)), 'settled_orientation_deg': float(max(rotations[-2:])),
                       'max_joint_speed_deg_s': float(max(speeds)), 'valid_all_joints': bool(valid),
                       'limited_arm_steps': limited, 'control_tick_p95_ms': float(np.percentile(timings, 95)),
                       'samples_per_arm': 451})
    return {'kind': 'synthetic_kinematics_only', 'robot_contacted': False,
            'model': config['model'], 'scale': config['position_scale'],
            'base_positions': {s: config['arms'][s]['base_position_m'] for s in ('left', 'right')},
            'results': output}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config'); p.add_argument('--output', required=True)
    args = p.parse_args(); result = evaluate(load_config(args.config))
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
