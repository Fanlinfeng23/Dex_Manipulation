#!/usr/bin/env bash
set -euo pipefail
WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Compatibility entry; all callers use the same native lifecycle and model menu.
exec "$WORKSPACE_ROOT/tianji_quest.sh" joint "$@"
