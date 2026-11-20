#!/usr/bin/env bash
set -euo pipefail

if [[ -f /workspace/pyproject.toml && -d /workspace/src ]]; then
  python -m pip install --no-deps --no-build-isolation -e /workspace >/tmp/editable-install.log
fi

exec "$@"
