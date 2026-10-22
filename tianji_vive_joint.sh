#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${TIANJI_HAND_PYTHON:-$ROOT/.venv/bin/python}"
exec "$PYTHON" -m vive_tianji.joint_launcher "$@"
