"""LLM-powered curator that merges agent logs into the research KB."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from alignment_faking.config import load_yaml
from alignment_faking.kb.append import list_unprocessed, mark_processed
from alignment_faking.kb.models import LogEntry
from alignment_faking.kb.paths import CURATOR_LOGS_DIR, KB_PATH
from alignment_faking.utils.prompts import render_prompt


@dataclass
class CuratorResult:
    updated: bool
    entries_processed: int
    audit_path: Path | None
    message: str


def _call_llm(provider: str, model: str, system: str, user: str) -> str:
    if provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        blocks = [block.text for block in response.content if block.type == "text"]
        return "\n".join(blocks)

    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def _extract_kb_markdown(response: str) -> str:
    fenced = re.search(r"```(?:markdown)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return response.strip()


def _format_logs_for_prompt(entries: list[LogEntry]) -> str:
    return json.dumps([entry.to_dict() for entry in entries], indent=2, ensure_ascii=False)


def run_curator(
    *,
    kb_path: Path | None = None,
    dry_run: bool = False,
) -> CuratorResult:
    kb_path = kb_path or KB_PATH
    pending = list_unprocessed()
    if not pending:
        return CuratorResult(updated=False, entries_processed=0, audit_path=None, message="No unprocessed logs.")

    entries = [entry for _, _, entry in pending]
    current_kb = kb_path.read_text(encoding="utf-8") if kb_path.exists() else ""

    cfg = load_yaml("kb.yaml")["curator"]
    provider = cfg.get("provider", "openai")
    model = cfg.get("model", "gpt-4o")

    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        return CuratorResult(
            updated=False,
            entries_processed=0,
            audit_path=None,
            message="OPENAI_API_KEY not set; cannot run curator.",
        )
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        return CuratorResult(
            updated=False,
            entries_processed=0,
            audit_path=None,
            message="ANTHROPIC_API_KEY not set; cannot run curator.",
        )

    prompt = render_prompt(
        "prompts/kb/curator.jinja2",
        current_kb=current_kb,
        agent_logs=_format_logs_for_prompt(entries),
        curated_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
    )
    system, _, user = prompt.partition("\n\n---USER---\n\n")
    if not user:
        user = prompt
        system = "You are the research KB curator. Merge agent logs into the KB without losing stable claims."

    response = _call_llm(provider, model, system.strip(), user.strip())
    new_kb = _extract_kb_markdown(response)

    if dry_run:
        audit_path = _write_audit(entries, current_kb, new_kb, dry_run=True)
        return CuratorResult(
            updated=False,
            entries_processed=len(entries),
            audit_path=audit_path,
            message=f"Dry run: would process {len(entries)} log entries.",
        )

    kb_path.parent.mkdir(parents=True, exist_ok=True)
    kb_path.write_text(new_kb + "\n", encoding="utf-8")
    mark_processed([(path, idx) for path, idx, _ in pending])
    audit_path = _write_audit(entries, current_kb, new_kb, dry_run=False)

    return CuratorResult(
        updated=True,
        entries_processed=len(entries),
        audit_path=audit_path,
        message=f"Updated KB from {len(entries)} log entries.",
    )


def _write_audit(
    entries: list[LogEntry],
    old_kb: str,
    new_kb: str,
    *,
    dry_run: bool,
) -> Path:
    CURATOR_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    path = CURATOR_LOGS_DIR / f"{stamp}_run.md"
    mode = "dry-run" if dry_run else "applied"
    body = "\n".join(
        [
            f"# Curator run ({mode})",
            "",
            f"- Timestamp: {datetime.now(UTC).replace(microsecond=0).isoformat()}",
            f"- Entries processed: {len(entries)}",
            "",
            "## Agent log IDs",
            "",
            *[f"- {entry.session_id or entry.timestamp}: {entry.what_did[:120]}" for entry in entries],
            "",
            "## KB size",
            "",
            f"- Before: {len(old_kb)} chars",
            f"- After: {len(new_kb)} chars",
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")
    return path
