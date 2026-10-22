#!/usr/bin/env bash
set -euo pipefail
WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TELEOP_APP="${TIANJI_QUEST_APP:-$WORKSPACE_ROOT/apps/quest_native_arm_teleop}"
TELEOP_PYTHON="${TIANJI_ARM_PYTHON:-$TELEOP_APP/.venv-linux/bin/python}"
ROBOT_IP="${TIANJI_ROBOT_IP:-192.168.1.190}"
PORT="${TIANJI_QUEST_PORT:-8766}"
DEVICE_INTERFACE="${TIANJI_DEVICE_INTERFACE:-enxec1ac3115da6}"
HOST_CIDR="${TIANJI_HOST_CIDR:-192.168.1.50/24}"
fail() {
  echo "错误：$*" >&2
  exit 1
}

require_app() {
  [[ -x "$TELEOP_APP/start.sh" ]] || fail "找不到遥操应用：$TELEOP_APP"
  [[ -x "$TELEOP_PYTHON" ]] || fail "缺少 Linux 运行环境；先在仓库根目录执行 ./setup_python.sh"
}

check_network() {
  command -v ip >/dev/null || fail "缺少 ip 命令"
  ip link show "$DEVICE_INTERFACE" >/dev/null 2>&1 || fail "未找到专用网卡 $DEVICE_INTERFACE"
  ip -4 addr show dev "$DEVICE_INTERFACE" | grep -Fq "$HOST_CIDR" ||
    fail "$DEVICE_INTERFACE 没有 $HOST_CIDR；请检查专用网卡配置"
  ip route get "$ROBOT_IP" | grep -q "dev $DEVICE_INTERFACE" ||
    fail "到 $ROBOT_IP 的路由未经 $DEVICE_INTERFACE"
  ping -c 1 -W 1 "$ROBOT_IP" >/dev/null || fail "控制柜 $ROBOT_IP 不可达"
}

check_quest() {
  command -v adb >/dev/null || fail "缺少 adb"
  local count
  count="$(adb devices | awk 'NR > 1 && $2 == "device" {n++} END {print n+0}')"
  [[ "$count" == "1" ]] || fail "需要且只能有 1 台已授权 ADB 设备，当前为 $count"
  adb devices -l | grep -q 'model:Quest_3S' || fail "ADB 设备不是 Quest 3S"
}

check_port_free() {
  command -v ss >/dev/null || fail "缺少 ss 命令"
  if ss -H -ltn "sport = :$PORT" | grep -q .; then
    fail "端口 $PORT 已被占用；先结束原服务的 Ctrl+C，或执行 ./tianji_quest.sh stop"
  fi
}

readonly_diagnosis() {
  require_app
  check_network
  check_quest
  mkdir -p "$WORKSPACE_ROOT/logs"
  local stamp report
  stamp="$(date +%Y%m%d-%H%M%S)"
  report="$WORKSPACE_ROOT/logs/tianji-readonly-$stamp.json"
  LAST_DIAG_REPORT="$report"
  echo "正在进行只读诊断；不清错、不使能、不发送运动指令。"
  if ! "$TELEOP_PYTHON" "$TELEOP_APP/tools/diagnose_robot.py" \
      --robot-ip "$ROBOT_IP" --output "$report"; then
    echo "诊断未通过，报告：$report" >&2
    if "$TELEOP_PYTHON" -c \
        'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])) .get("standby_estop_recovery_candidate") else 1)' \
        "$report" 2>/dev/null; then
      echo "双臂符合“待机目标 + 公共急停 13 + 14 轴无驱动报警”。" >&2
      echo "现场释放物理急停并确认工作区后，执行：./tianji_quest.sh recover" >&2
    elif "$TELEOP_PYTHON" -c \
        'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("failure_class") == "axis_telemetry_placeholder" else 1)' \
        "$report" 2>/dev/null; then
      echo "急停已解除，但 14 轴编码器/温度仍是占位零值；真机启动已安全拦截。" >&2
      echo "请检查并开启机械臂 DC48V 驱动电源，确认 EtherCAT/电机线缆牢固，然后重新执行 check。" >&2
    elif "$TELEOP_PYTHON" -c \
        'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("failure_class") in ("position_mode_requires_standby", "active_mode_requires_standby") else 1)' \
        "$report" 2>/dev/null; then
      echo "双臂仍在 state=1/3 运动模式，新真机服务不会接管。" >&2
      echo "确认现场安全后执行：./tianji_quest.sh standby" >&2
    fi
    return 2
  fi
  echo "诊断通过，报告：$report"
}

is_recovery_candidate() {
  [[ -n "${LAST_DIAG_REPORT:-}" && -f "$LAST_DIAG_REPORT" ]] &&
    "$TELEOP_PYTHON" -c \
      'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("standby_estop_recovery_candidate") else 1)' \
      "$LAST_DIAG_REPORT" 2>/dev/null
}

recover_estop() {
  if [[ ! -t 0 ]]; then
    fail "急停恢复必须在现场交互式终端执行"
  fi
  echo "此操作只会向双臂发送“清急停锁存 + state=0 待机”；不使能、不释放抱闸、不发送关节目标。"
  read -r -p "请确认物理急停已旋出/拉起、急停插头正确、双臂工作区无人。输入 RELEASED 继续：" answer
  [[ "$answer" == "RELEASED" ]] || fail "未获得现场安全确认，已取消"
  mkdir -p "$WORKSPACE_ROOT/logs"
  local stamp report
  stamp="$(date +%Y%m%d-%H%M%S)"
  report="$WORKSPACE_ROOT/logs/tianji-recovery-$stamp.json"
  if ! "$TELEOP_PYTHON" "$TELEOP_APP/tools/recover_robot.py" \
      --robot-ip "$ROBOT_IP" --confirm-physical-estop-released --output "$report"; then
    echo "恢复未通过，不允许启动遥操。报告：$report" >&2
    if "$TELEOP_PYTHON" -c \
        'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("failure_class") == "hardware_safety_chain_still_asserted" else 1)' \
        "$report" 2>/dev/null; then
      echo "硬件安全链仍断开：不要再重复 recover/清错。" >&2
      echo "请断电并由熟悉电气安全的人员检查 GPIO 急停插头 3/4/5 号线、按钮常闭触点和 48V 驱动供电；禁止临时短接旁路。" >&2
    fi
    return 2
  fi
  echo "恢复为无故障待机，报告：$report"
}

command="${1:-}"
case "$command" in
  check)
    readonly_diagnosis
    ;;
  recover)
    require_app
    check_network
    check_quest
    check_port_free
    recover_estop
    readonly_diagnosis
    ;;
  standby)
    require_app
    check_network
    check_port_free
    mkdir -p "$WORKSPACE_ROOT/logs"
    stamp="$(date +%Y%m%d-%H%M%S)"
    report="$WORKSPACE_ROOT/logs/tianji-standby-$stamp.json"
    echo "仅请求双臂由 state=1/3 转为 state=0 待机；不清错、不发送关节目标。"
    "$TELEOP_PYTHON" "$TELEOP_APP/tools/standby_robot.py" \
      --robot-ip "$ROBOT_IP" --output "$report"
    echo "待机已确认，报告：$report"
    ;;
  right-joints)
    require_app
    check_network
    check_port_free
    mkdir -p "$WORKSPACE_ROOT/logs"
    stamp="$(date +%Y%m%d-%H%M%S)"
    preflight="$WORKSPACE_ROOT/logs/tianji-right-joints-preflight-$stamp.json"
    echo "先执行只读门禁；急停、故障、非待机状态均会拒绝逐轴运动。"
    "$TELEOP_PYTHON" "$TELEOP_APP/tools/diagnose_robot.py" \
      --robot-ip "$ROBOT_IP" --output "$preflight" || {
        echo "逐轴诊断未启动，门禁报告：$preflight" >&2
        exit 2
      }
    [[ -t 0 ]] || fail "右臂逐轴诊断必须在现场交互式终端执行"
    echo "右臂将以厂家位置模式、1% 速度/加速度依次测试：J1–J5 ±1°，J6–J7 ±0.5°。"
    echo "每轴均执行正向、回零、负向、回零；左臂始终保持 state=0。"
    read -r -p "确认右臂各关节的小幅运动路径无人无障碍、急停在手边。输入 TEST-RIGHT 继续：" answer
    [[ "$answer" == "TEST-RIGHT" ]] || fail "未获得右臂逐轴运动确认，已取消"
    report="$WORKSPACE_ROOT/logs/tianji-right-joints-$stamp.json"
    csv="$WORKSPACE_ROOT/logs/tianji-right-joints-$stamp.csv"
    "$TELEOP_PYTHON" "$TELEOP_APP/tools/test_right_joints.py" \
      --robot-ip "$ROBOT_IP" --duration-s 2 --sample-hz 25 --confirmed \
      --output "$report" --csv "$csv"
    echo "右臂逐轴诊断完成：$report"
    echo "原始遥测：$csv"
    ;;
  home)
    require_app
    check_network
    check_port_free
    [[ -t 0 ]] || fail "双臂复位必须在现场交互式终端执行"
    mkdir -p "$WORKSPACE_ROOT/logs"
    stamp="$(date +%Y%m%d-%H%M%S)"
    preflight="$WORKSPACE_ROOT/logs/tianji-home-preflight-$stamp.json"
    plan_file="$WORKSPACE_ROOT/logs/tianji-home-plan-$stamp.json"
    report="$WORKSPACE_ROOT/logs/tianji-home-$stamp.json"
    echo "正在进行复位前只读诊断。"
    "$TELEOP_PYTHON" "$TELEOP_APP/tools/diagnose_robot.py" \
      --robot-ip "$ROBOT_IP" --output "$preflight" >/dev/null || {
        echo "复位未启动；基础诊断未通过：$preflight" >&2
        exit 2
      }
    if ! "$TELEOP_PYTHON" "$TELEOP_APP/tools/return_to_home.py" \
        --report "$preflight" --config "$TELEOP_APP/configs/hardware_dual_arm.json" \
        >"$plan_file"; then
      echo "复位未启动；限位详情：$plan_file；原始反馈：$preflight" >&2
      exit 2
    fi
    "$TELEOP_PYTHON" - "$plan_file" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding='utf-8'))
for phase in p['phases']:
    print(f"计划：{phase['name']}，{phase['duration_s']:.1f} 秒，控制 {', '.join(phase['control_sides'])}")
print(f"总运动时间约 {p['estimated_motion_s']:.1f} 秒，关节速度不超过 5°/s；目标为 Wuji 遥操初始姿态。")
print(p['warning'])
PY
    read -r -p "确认双臂整段路径及手部无人无障碍、急停在手边。输入 HOME 启动低速复位：" answer
    [[ "$answer" == "HOME" ]] || fail "未获得现场运动确认，已取消"
    if ! "$TELEOP_PYTHON" "$TELEOP_APP/tools/return_to_home.py" \
        --report "$preflight" --config "$TELEOP_APP/configs/hardware_dual_arm.json" \
        --execute --confirmed --output "$report"; then
      echo "复位未完成，已请求待机；勿直接启动遥操。报告：$report" >&2
      exit 2
    fi
    echo "复位运动已完成；正在重新运行启动门禁。"
    postflight="$WORKSPACE_ROOT/logs/tianji-home-postflight-$stamp.json"
    pose_report="$WORKSPACE_ROOT/logs/tianji-home-pose-$stamp.json"
    "$TELEOP_PYTHON" "$TELEOP_APP/tools/diagnose_robot.py" \
      --robot-ip "$ROBOT_IP" --output "$postflight" >/dev/null ||
      fail "复位后基础门禁未通过：$postflight"
    "$TELEOP_PYTHON" "$TELEOP_APP/tools/check_full_arm_pose.py" \
      --report "$postflight" --config "$TELEOP_APP/configs/hardware_dual_arm.json" \
      --output "$pose_report" >/dev/null ||
      fail "复位后整臂姿态门禁未通过：$pose_report"
    echo "双臂已回初始姿态，门禁通过。可重新执行 ./tianji_quest.sh joint。复位报告：$report"
    ;;
  *) fail "维护命令：check / home / recover / standby / right-joints" ;;
esac
