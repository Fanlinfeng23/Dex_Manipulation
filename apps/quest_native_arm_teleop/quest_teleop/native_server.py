"""Isolated local arm server for official XRoboToolkit Quest frames."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aiohttp import web

from .config import configure_runtime, load_config
from .server import create_app
from .service import PreviewServer


def read_token(path):
    token = Path(path).read_text(encoding='utf-8').strip()
    if len(token) < 32 or any(c.isspace() for c in token):
        raise ValueError('原生遥操 token 文件无效')
    return token


def build_app(config_path, *, hardware=False, robot_ip=None, native_status_file, token_file, joint_gate=None):
    config = configure_runtime(load_config(config_path, hardware=hardware),
                               hardware=hardware, robot_ip=robot_ip)
    if config['hardware']['motion_scope'] != 'full' or config['hardware']['control_sides'] != ['left', 'right']:
        raise ValueError('原生遥操仅使用双臂整臂配置')
    if config.get('initial_source') != 'quest':
        raise ValueError('原生遥操配置 initial_source 必须为 quest')
    service = PreviewServer(config, native_token=read_token(token_file),
                            native_status_file=native_status_file)
    service.native_joint_gate = Path(joint_gate) if joint_gate else None
    return create_app(service=service)


def main(argv=None):
    parser = argparse.ArgumentParser(description='XRoboToolkit 原生 Quest 双臂服务（仅本机）')
    parser.add_argument('--config', required=True)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--robot-ip')
    parser.add_argument('--native-status-file', required=True)
    parser.add_argument('--token-file', required=True)
    parser.add_argument('--joint-gate', help='联合手部门禁，仅用于状态显示')
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error('port 必须在 1024–65535 之间')
    try:
        app = build_app(args.config, hardware=args.hardware, robot_ip=args.robot_ip,
                        native_status_file=args.native_status_file, token_file=args.token_file,
                        joint_gate=args.joint_gate)
        web.run_app(app, host='127.0.0.1', port=args.port, print=None, handle_signals=True)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'原生双臂服务启动失败：{exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
