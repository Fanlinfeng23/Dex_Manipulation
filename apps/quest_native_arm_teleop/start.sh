#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_ROOT"

if [[ -x ".venv-linux/bin/python" ]]; then
  python_bin=".venv-linux/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  python_bin=".venv/bin/python"
else
  echo "Ubuntu Python 环境不存在，请先在仓库根目录运行 ./setup_python.sh。" >&2
  exit 1
fi

export PYTHONUTF8=1
exec "$python_bin" -m quest_teleop "$@"
