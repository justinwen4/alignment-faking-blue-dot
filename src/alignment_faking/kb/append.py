"""Append-only agent log storage."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from alignment_faking.kb.models import LogEntry
from alignment_faking.kb.paths import AGENT_LOGS_DIR


def _log_path_for_date(when: datetime | None = None) -> Path:
    when = when or datetime.now(UTC)
    return AGENT_LOGS_DIR / f"{when.strftime('%Y-%m-%d')}.jsonl"


def append_log_entry(entry: LogEntry, *, log_path: Path | None = None) -> Path:
    """Append one log entry to the daily JSONL file."""
    path = log_path or _log_path_for_date()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.to_json() + "\n")
    return path


def iter_log_entries(log_dir: Path | None = None) -> list[tuple[Path, int, LogEntry]]:
    """Return all log entries with (file, line_index, entry)."""
    log_dir = log_dir or AGENT_LOGS_DIR
    if not log_dir.exists():
        return []

    entries: list[tuple[Path, int, LogEntry]] = []
    for path in sorted(log_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for index, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                entries.append((path, index, LogEntry.from_dict(json.loads(line))))
    return entries


def list_unprocessed(log_dir: Path | None = None) -> list[tuple[Path, int, LogEntry]]:
    return [(path, idx, entry) for path, idx, entry in iter_log_entries(log_dir) if not entry.processed_by_curator]


def mark_processed(
    targets: list[tuple[Path, int]],
    log_dir: Path | None = None,
) -> int:
    """Mark log entries as processed by rewriting their JSONL files."""
    if not targets:
        return 0

    by_file: dict[Path, set[int]] = {}
    for path, index in targets:
        by_file.setdefault(path, set()).add(index)

    updated = 0
    for path, indices in by_file.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        for index in indices:
            if index >= len(lines) or not lines[index].strip():
                continue
            record = json.loads(lines[index])
            if record.get("processed_by_curator"):
                continue
            record["processed_by_curator"] = True
            lines[index] = json.dumps(record, ensure_ascii=False)
            updated += 1
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return updated
