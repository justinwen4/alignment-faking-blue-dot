"""Backfill agent logs from Cursor agent transcript JSONL files."""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from alignment_faking.kb.append import append_log_entry, iter_log_entries
from alignment_faking.kb.models import LogEntry
from alignment_faking.kb.summarize import build_log_entry_from_session

DEFAULT_TRANSCRIPTS_DIR = Path.home() / ".cursor" / "projects" / "Users-overp-alignment-faking" / "agent-transcripts"

USER_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL)
EDIT_TOOLS = {"Write", "StrReplace", "Delete", "ApplyPatch"}


def _existing_session_ids() -> set[str]:
    return {entry.session_id for _, _, entry in iter_log_entries() if entry.session_id}


def _parse_transcript(path: Path) -> dict[str, Any]:
    user_queries: list[str] = []
    files_changed: list[str] = []
    final_assistant_text = ""

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            role = record.get("role")
            message = record.get("message") or {}
            content = message.get("content") or []

            if role == "user":
                for block in content:
                    if block.get("type") != "text":
                        continue
                    text = block.get("text") or ""
                    match = USER_QUERY_RE.search(text)
                    user_queries.append(match.group(1).strip() if match else text.strip())

            if role == "assistant":
                for block in content:
                    if block.get("type") == "text":
                        text = block.get("text") or ""
                        if text and text != "[REDACTED]":
                            final_assistant_text = text
                    if block.get("type") == "tool_use":
                        tool = block.get("name")
                        tool_input = block.get("input") or {}
                        if tool in EDIT_TOOLS:
                            file_path = tool_input.get("path") or tool_input.get("file_path")
                            if file_path and file_path not in files_changed:
                                files_changed.append(file_path)

    session_id = path.stem
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).replace(microsecond=0)

    return {
        "session_id": session_id,
        "timestamp": mtime.isoformat(),
        "user_queries": user_queries,
        "files_changed": files_changed,
        "final_assistant_text": final_assistant_text,
    }


def _transcript_context(parsed: dict[str, Any]) -> str:
    parts = [
        f"Session ID: {parsed['session_id']}",
        f"Transcript timestamp: {parsed['timestamp']}",
    ]
    if parsed["user_queries"]:
        parts.append("User requests:\n" + "\n---\n".join(parsed["user_queries"]))
    if parsed["files_changed"]:
        parts.append("Files edited:\n" + "\n".join(f"- {p}" for p in parsed["files_changed"]))
    if parsed["final_assistant_text"]:
        parts.append("Final assistant summary:\n" + parsed["final_assistant_text"][:3000])
    return "\n\n".join(parts)


def backfill_from_transcripts(
    *,
    transcripts_dir: Path,
    on_date: date | None = None,
    dry_run: bool = False,
) -> list[LogEntry]:
    """Create log entries from Cursor agent transcripts not already logged."""
    on_date = on_date or datetime.now(UTC).date()
    existing = _existing_session_ids()
    created: list[LogEntry] = []

    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Transcripts directory not found: {transcripts_dir}")

    for path in sorted(transcripts_dir.glob("*/*.jsonl")):
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date()
        if mtime != on_date:
            continue

        parsed = _parse_transcript(path)
        if parsed["session_id"] in existing:
            continue

        state = {
            "session_id": parsed["session_id"],
            "started_at": parsed["timestamp"],
            "agent_name": "cursor-agent-backfill",
            "files_changed": parsed["files_changed"],
            "git_head_at_start": None,
        }
        hook_input = {
            "summary": _transcript_context(parsed),
            "status": "backfilled from transcript",
        }
        entry = build_log_entry_from_session(state, hook_input)
        entry.timestamp = parsed["timestamp"]
        entry.agent_name = "cursor-agent-backfill"
        entry.session_id = parsed["session_id"]

        if not dry_run:
            append_log_entry(entry)
        created.append(entry)
        existing.add(parsed["session_id"])

    return created
