#!/usr/bin/env bash
# On agent stop: auto-summarize session and append structured log entry.
set -euo pipefail

input="$(cat)"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

echo "$input" | "$PYTHON" scripts/log_agent_entry.py --from-hook 2>/dev/null || exit 0
exit 0
