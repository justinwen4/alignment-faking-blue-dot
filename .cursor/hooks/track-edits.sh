#!/usr/bin/env bash
# Track edited files in session state.
set -euo pipefail

input="$(cat)"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

file_path=""
if command -v jq >/dev/null 2>&1; then
  file_path="$(echo "$input" | jq -r '.file_path // .path // .filePath // empty' 2>/dev/null || true)"
fi

if [[ -z "$file_path" ]]; then
  file_path="$(echo "$input" | grep -Eo '"file_path"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*"file_path"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/' || true)"
fi

if [[ -z "$file_path" ]]; then
  exit 0
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

FILE_PATH="$file_path" "$PYTHON" - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
from alignment_faking.kb.session import track_file_edit

track_file_edit(os.environ["FILE_PATH"])
PY

exit 0
