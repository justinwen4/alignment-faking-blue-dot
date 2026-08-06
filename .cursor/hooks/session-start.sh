#!/usr/bin/env bash
# Initialize session state when a Cursor agent session starts.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  exit 0
fi

"$PYTHON" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
from alignment_faking.kb.session import init_session

init_session()
PY

exit 0
