#!/usr/bin/env bash
set -euo pipefail
WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command="${1:-help}"
[[ $# == 0 ]] || shift
case "$command" in
  help|-h|--help)
    cat <<'EOF'
XRoboToolkit Quest 原生遥操（默认端口 8766）
用法：./tianji_quest.sh <命令> [选项]

  joint       正式联合启动：天机双臂 + 左右 Wuji Glove / Hand2
  start       原生天机双臂；不运行手部控制（dual-arm 同义）
  preview     原生采集预演，不连接机器人；--keep-capture 保留官方应用
  check       只读检查 Quest、左右 Glove/Hand2、天机双臂
  capture     空闲时启动原生 PC Service、USB 转发与 Quest 官方应用（usb 同义）
  tracking    采样当前会话 5 秒数值门禁状态；不表示光学追踪覆盖率
  status      查看原生会话、采集与双臂状态
  stop        结束本入口拥有的原生会话及双手控制
  home        结束遥操后复位双臂；现场输入 HOME
  standby     结束遥操后将双臂转待机
  recover     释放物理急停后恢复待机；现场输入 RELEASED
  right-joints 现场低速逐轴诊断右臂

联合模式：
  --model-user NAME       选择已完成左右标定的操作员（名称或 ID）
  --calibrate-user NAME   新建/重标定操作员，依次标定左右手
  不指定用户时，在此电脑终端的模型菜单输入 c，或选择已有模型编号。
  --left-hand-sn / --right-hand-sn SERIAL    指定左右 Hand2
  --left-glove-sn / --right-glove-sn SERIAL  指定左右 Glove
  --manual               兼容旧命令；实机现在默认首次按空格开始

双臂到初始姿态后保持；Quest 输入就绪会自动校准，但不会自动开始跟随。
现场准备好后在本终端按空格首次开始；首次 A 只重新校准。
沿用 WUJI_MODEL_USER / WUJI_CALIBRATE_USER、四侧序列号、
TIANJI_JOINT_HEADING_SOURCE / TIANJI_JOINT_OPERATOR_YAW_DEG、TIANJI_ROBOT_IP。
端口：TIANJI_NATIVE_PORT 优先，其次 TIANJI_QUEST_PORT，默认 8766。
完整说明：Quest天机遥操启动指南.md
EOF
    ;;
  joint|start|preview|capture|tracking|status|stop|home|standby|recover|right-joints)
    exec "$WORKSPACE_ROOT/tianji_native.sh" "$command" "$@" ;;
  check)
    exec "$WORKSPACE_ROOT/tianji_native.sh" check --check-hands "$@" ;;
  dual-arm)
    exec "$WORKSPACE_ROOT/tianji_native.sh" start "$@" ;;
  usb)
    exec "$WORKSPACE_ROOT/tianji_native.sh" capture "$@" ;;
  *) echo "未知命令：$command；使用 ./tianji_quest.sh help" >&2; exit 2 ;;
esac
