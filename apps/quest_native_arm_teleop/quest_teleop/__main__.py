import argparse
from aiohttp import web

from .config import APP_ROOT, configure_runtime, load_config
from .server import create_app


def main():
    parser = argparse.ArgumentParser(description='Quest 3S → Tianji dual-arm teleoperation')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--config', default=None)
    parser.add_argument('--source', choices=('simulation', 'quest'), default='simulation',
                        help='启动输入源；Quest 预演使用 --source quest')
    parser.add_argument('--hardware', action='store_true',
                        help='启用天机实机输出；默认仅运行虚拟预演')
    parser.add_argument('--full-arm', action='store_true',
                        help='明确解除仅手腕范围；需要现场具备整臂运动条件')
    parser.add_argument('--robot-ip', default=None,
                        help='天机控制柜 IPv4 地址（仅 --hardware）')
    parser.add_argument('--vel-ratio', type=int, default=None,
                        help='实机速度比例 1–100%%，默认由配置决定（Lite 为 10%%）')
    parser.add_argument('--acc-ratio', type=int, default=None,
                        help='实机加速度比例 1–100%%，默认由配置决定（Lite 为 10%%）')
    args = parser.parse_args()
    if args.full_arm and not args.hardware:
        parser.error('--full-arm 仅适用于 --hardware')
    if not 1024 <= args.port <= 65535:
        parser.error('port 必须在 1024–65535 之间')
    for name, value in (('--vel-ratio', args.vel_ratio), ('--acc-ratio', args.acc_ratio)):
        if value is not None and not 1 <= value <= 100:
            parser.error(f'{name} 必须在 1–100 之间')
    if not (APP_ROOT / 'node_modules/three/build/three.module.js').is_file():
        parser.error('请先在应用目录运行 npm install')
    profile = args.config or (APP_ROOT / 'configs/hardware_lite.json' if args.full_arm else None)
    config = configure_runtime(load_config(profile, hardware=args.hardware), hardware=args.hardware,
                               robot_ip=args.robot_ip, vel_ratio=args.vel_ratio,
                               acc_ratio=args.acc_ratio)
    if args.hardware and config['hardware']['motion_scope'] != 'wrist' and not args.full_arm:
        parser.error('整臂配置需要显式 --full-arm；当前现场调试请用默认手腕配置')
    config['initial_source'] = 'quest' if args.hardware else args.source
    print(f'电脑预演：http://127.0.0.1:{args.port}/')
    print(f'Quest 浏览器：http://127.0.0.1:{args.port}/quest')
    print(f'USB 转发：adb reverse tcp:{args.port} tcp:{args.port}')
    if args.hardware:
        sides = config['hardware'].get('control_sides', ['left', 'right'])
        scope = ('仅 J6/J7 手腕，J1–J5 固定，平移输入忽略'
                 if config['hardware']['motion_scope'] == 'wrist' else
                 '仅右臂整臂，左臂保持待机' if sides == ['right'] else '整臂运动')
        startup = ('启动时按 Wuji 官方参数使能并运动到双臂初始姿态'
                   if config['hardware'].get('startup_move_enabled') else '首次跟随时使能')
        print(f'实机模式：天机 {config["hardware"]["robot_ip"]}；{scope}；{startup}；B 暂停。', flush=True)
    else:
        print('虚拟预演模式：不会连接或发送机器人指令。', flush=True)
    web.run_app(create_app(config), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
