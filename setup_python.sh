#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP="$ROOT/apps/quest_native_arm_teleop"
PYTHON_BOOTSTRAP="${TIANJI_BOOTSTRAP_PYTHON:-python3}"

"$PYTHON_BOOTSTRAP" -m venv "$APP/.venv-linux"
"$APP/.venv-linux/bin/python" -m pip install -r "$APP/requirements.txt" 'openvr==2.12.1401'
"$PYTHON_BOOTSTRAP" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.lock.txt"

printf 'Python 环境已创建。厂商 SDK、Quest PC Service、SteamVR 与设备配置仍需按指南准备。\n'
