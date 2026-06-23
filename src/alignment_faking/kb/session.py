"""Session state tracking for Cursor hooks."""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alignment_faking.kb.paths import SESSION_STATE_PATH


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_changed_files(since_ref: str | None) -> list[str]:
    if not since_ref:
        return []
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", since_ref, "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _git_uncommitted_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        files: list[str] = []
        for line in result.stdout.splitlines():
            if len(line) < 4:
                continue
            path = line[3:].strip()
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            files.append(path)
        return files
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def init_session(*, agent_name: str = "cursor-agent", state_path: Path | None = None) -> dict[str, Any]:
    path = state_path or SESSION_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "session_id": str(uuid.uuid4()),
        "started_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "agent_name": agent_name,
        "files_changed": [],
        "git_head_at_start": _git_head(),
    }
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def load_session(state_path: Path | None = None) -> dict[str, Any] | None:
    path = state_path or SESSION_STATE_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def track_file_edit(file_path: str, state_path: Path | None = None) -> dict[str, Any]:
    path = state_path or SESSION_STATE_PATH
    state = load_session(path) or init_session(state_path=path)
    changed: list[str] = state.setdefault("files_changed", [])
    if file_path not in changed:
        changed.append(file_path)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def collect_sources_changed(state: dict[str, Any]) -> list[str]:
    tracked = list(state.get("files_changed") or [])
    git_since = _git_changed_files(state.get("git_head_at_start"))
    uncommitted = _git_uncommitted_files()
    merged: list[str] = []
    for item in tracked + git_since + uncommitted:
        if item and item not in merged:
            merged.append(item)
    return merged


def session_context_for_summary(state: dict[str, Any], hook_input: dict[str, Any] | None = None) -> str:
    parts = [
        f"Session started: {state.get('started_at', 'unknown')}",
        f"Agent: {state.get('agent_name', 'cursor-agent')}",
        f"Session ID: {state.get('session_id', '')}",
    ]
    sources = collect_sources_changed(state)
    if sources:
        parts.append("Files changed:\n" + "\n".join(f"- {p}" for p in sources))
    if hook_input:
        for key in ("conversation", "summary", "last_message", "status", "reason"):
            value = hook_input.get(key)
            if value:
                parts.append(f"{key}:\n{value}")
        if not any(hook_input.get(k) for k in ("conversation", "summary", "last_message")):
            parts.append("Hook payload:\n" + json.dumps(hook_input, indent=2)[:4000])
    return "\n\n".join(parts)
