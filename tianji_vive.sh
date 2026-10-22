#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP="${TIANJI_QUEST_APP:-$ROOT/apps/quest_native_arm_teleop}"
PYTHON="${TIANJI_ARM_PYTHON:-$APP/.venv-linux/bin/python}"
OPENVR_SITE="${TIANJI_OPENVR_SITE:-}"
if [[ -z "$OPENVR_SITE" ]]; then
  for candidate in "$ROOT"/.vive-venv/lib/python*/site-packages; do
    if [[ -d "$candidate/openvr" ]]; then
      OPENVR_SITE="$candidate"
      break
    fi
  done
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "缺少 Python 运行环境：$PYTHON" >&2
  exit 2
fi
if [[ -n "$OPENVR_SITE" && ! -d "$OPENVR_SITE/openvr" ]]; then
  echo "缺少 OpenVR Python 包：$OPENVR_SITE/openvr" >&2
  exit 2
fi
# OpenVR can start vrserver itself, but that bypasses Steam's Linux runtime.
# On this host the bare 64-bit vrserver cannot find libSDL2, so use Steam's
# normal SteamVR launch path before importing OpenVR.
if [[ "${1:-}" == calibrate || "${1:-}" == preview || "${1:-}" == live ]] && \
   ! pgrep -u "$(id -u)" -x vrserver >/dev/null; then
  if ! command -v steam >/dev/null; then
    echo "SteamVR 未运行，且未找到 Steam；请先安装并启动 SteamVR。" >&2
    exit 2
  fi
  echo "SteamVR 未运行；正在通过 Steam 启动（最多等待 90 秒）。" >&2
  nohup env GDK_BACKEND=x11 QT_QPA_PLATFORM=xcb steam steam://rungameid/250820 \
    >/tmp/tianji-vive-steamvr-launch.log 2>&1 </dev/null &
  for ((attempt = 0; attempt < 90; attempt++)); do
    if pgrep -u "$(id -u)" -x vrserver >/dev/null; then
      break
    fi
    sleep 1
  done
  if ! pgrep -u "$(id -u)" -x vrserver >/dev/null; then
    echo "SteamVR 未能启动。请检查 Steam 更新/登录状态和 /tmp/tianji-vive-steamvr-launch.log；未运行标定或遥操。" >&2
    exit 2
  fi
fi
export PYTHONPATH="$ROOT:$APP${OPENVR_SITE:+:$OPENVR_SITE}${PYTHONPATH:+:$PYTHONPATH}"
if ! "$PYTHON" -c 'import openvr' >/dev/null 2>&1; then
  echo "缺少 OpenVR Python 包；请在 $PYTHON 中安装 openvr 或设置 TIANJI_OPENVR_SITE。" >&2
  exit 2
fi
exec "$PYTHON" -m vive_tianji.teleop "$@"
