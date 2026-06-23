"""Auto-summarize a session into a structured log entry."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from alignment_faking.config import load_yaml
from alignment_faking.kb.append import append_log_entry
from alignment_faking.kb.models import LogEntry
from alignment_faking.kb.session import collect_sources_changed, session_context_for_summary


def _call_summarizer(provider: str, model: str, context: str) -> dict[str, Any]:
    system = (
        "You summarize agent work sessions into structured JSON for a research project log. "
        "Return ONLY valid JSON with keys: what_did (string), findings (string), "
        "uncertainties (string), next_actions (array of strings). Be concise and factual."
    )
    user = f"Summarize this agent session:\n\n{context}"

    if provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "\n".join(block.text for block in response.content if block.type == "text")
    else:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content or "{}"

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    return json.loads(match.group())


def build_log_entry_from_session(
    state: dict[str, Any],
    hook_input: dict[str, Any] | None = None,
) -> LogEntry:
    sources = collect_sources_changed(state)
    context = session_context_for_summary(state, hook_input)

    cfg = load_yaml("kb.yaml")["log"]["summarizer"]
    provider = cfg.get("provider", "openai")
    model = cfg.get("model", "gpt-4o-mini")

    summary: dict[str, Any] = {}
    api_ok = (provider == "openai" and os.environ.get("OPENAI_API_KEY")) or (
        provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY")
    )
    if api_ok:
        try:
            summary = _call_summarizer(provider, model, context)
        except Exception:
            summary = {}

    return LogEntry(
        timestamp="",
        agent_name=state.get("agent_name", "cursor-agent"),
        session_id=state.get("session_id", ""),
        what_did=summary.get("what_did") or _fallback_what_did(sources),
        sources_changed=sources,
        findings=summary.get("findings", ""),
        uncertainties=summary.get("uncertainties", ""),
        next_actions=list(summary.get("next_actions") or []),
    )


def _fallback_what_did(sources: list[str]) -> str:
    if not sources:
        return "Agent session completed (no file changes detected)."
    preview = ", ".join(sources[:5])
    suffix = f" (+{len(sources) - 5} more)" if len(sources) > 5 else ""
    return f"Modified {len(sources)} file(s): {preview}{suffix}"


def write_session_log(
    state: dict[str, Any],
    hook_input: dict[str, Any] | None = None,
) -> LogEntry:
    entry = build_log_entry_from_session(state, hook_input)
    append_log_entry(entry)
    return entry
