#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP="${TIANJI_QUEST_APP:-$ROOT/apps/quest_native_arm_teleop}"
PYTHON="${TIANJI_ARM_PYTHON:-$APP/.venv-linux/bin/python}"
exec "$PYTHON" "$APP/native_launcher.py" "$@"
