#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(dirname "$(readlink -f "$0")")"
cd "$APP_DIR"

if [ -x "$APP_DIR/.venv/bin/python" ]; then
  CORPUS_PYTHON="$APP_DIR/.venv/bin/python"
else
  CORPUS_PYTHON="${PYTHON:-python3}"
fi

if [ -x "$APP_DIR/dist/linux-unpacked/corpus" ]; then
  exec env -u ELECTRON_RUN_AS_NODE PYTHON="$CORPUS_PYTHON" "$APP_DIR/dist/linux-unpacked/corpus" "$@"
fi

exec env -u ELECTRON_RUN_AS_NODE PYTHON="$CORPUS_PYTHON" npm run start -- "$@"
