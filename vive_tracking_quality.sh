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
  echo "缺少遥操 Python 环境：$PYTHON" >&2
  exit 2
fi
export PYTHONPATH="$ROOT:$APP${OPENVR_SITE:+:$OPENVR_SITE}${PYTHONPATH:+:$PYTHONPATH}"
if ! "$PYTHON" -c 'import openvr' >/dev/null 2>&1; then
  echo "缺少 OpenVR Python 包；请运行 ./setup_python.sh 或设置 TIANJI_OPENVR_SITE。" >&2
  exit 2
fi
exec "$PYTHON" "$ROOT/vive_tracking_quality.py" "$@"
