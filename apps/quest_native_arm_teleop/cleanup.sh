#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
port=8765
what_if=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port|-p) [[ $# -ge 2 ]] || { echo "--port 需要端口" >&2; exit 2; }; port="$2"; shift 2 ;;
    --what-if|-n) what_if=1; shift ;;
    -h|--help) echo "用法：./cleanup.sh [--port PORT] [--what-if]"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done
if ! [[ "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
  echo "端口必须是 1024–65535 的整数。" >&2
  exit 2
fi

declare -A ppid=() args=() cwd=() selected=()
while read -r pid parent command; do
  [[ -n "$pid" ]] || continue
  ppid["$pid"]="$parent"
  args["$pid"]="$command"
  cwd["$pid"]="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
done < <(ps -eo pid=,ppid=,args=)

port_in_args() {
  local command="$1" value
  if [[ "$command" =~ (^|[[:space:]])--port[=[:space:]]+([0-9]+)($|[[:space:]]) ]]; then
    value="${BASH_REMATCH[2]}"
  else
    value=8765
  fi
  [[ "$value" == "$port" ]]
}

for pid in "${!args[@]}"; do
  command="${args[$pid]}"
  if [[ "$command" == *"-m quest_teleop "* || "$command" == *"-m quest_teleop.keyboard "* ||
        "$command" == *"-m quest_teleop" || "$command" == *"-m quest_teleop.keyboard" ]]; then
    if port_in_args "$command" && [[ "${cwd[$pid]}" == "$APP_ROOT" ]]; then
      selected["$pid"]=1
    fi
  fi
done

# Include only descendants of selected app roots. This catches the spawned
# control worker without touching unrelated Python, adb, or terminal processes.
changed=1
while (( changed )); do
  changed=0
  for pid in "${!args[@]}"; do
    [[ -n "${selected[$pid]:-}" ]] && continue
    parent="${ppid[$pid]:-}"
    if [[ -n "${selected[$parent]:-}" && "${cwd[$pid]}" == "$APP_ROOT" ]]; then
      selected["$pid"]=1
      changed=1
    fi
  done
done

if (( ${#selected[@]} == 0 )); then
  echo "没有找到本应用在端口 $port 上的进程。"
else
  echo "将处理端口 $port 的 Quest 遥操进程："
  for pid in "${!selected[@]}"; do
    echo "  PID $pid: ${args[$pid]}"
  done
fi
if (( what_if )); then
  echo "预览模式：没有结束进程。"
  exit 0
fi

# Signal the Python roots first so aiohttp can invoke the worker's safe
# Tianji standby/release path. Remaining descendants are only a fallback.
mapfile -t roots < <(for pid in "${!selected[@]}"; do
  parent="${ppid[$pid]:-}"
  [[ -z "${selected[$parent]:-}" ]] && echo "$pid"
done)
for pid in "${roots[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "发送 TERM 到 PID $pid"
    kill -TERM "$pid" 2>/dev/null || true
  fi
done
sleep .5
for pid in "${!selected[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "发送 TERM 到残留 PID $pid"
    kill -TERM "$pid" 2>/dev/null || true
  fi
done
sleep .3
for pid in "${!selected[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "PID $pid 未退出，发送 KILL。" >&2
    kill -KILL "$pid" 2>/dev/null || true
  fi
done

if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :$port" | tail -n +2 | grep -q .; then
  echo "端口 $port 仍被占用；未匹配的程序未被关闭。" >&2
  exit 1
fi
echo "清理完成。"
