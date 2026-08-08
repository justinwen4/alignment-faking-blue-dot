# Research KB Automation — Design Spec

**Date:** 2025-06-22  
**Status:** Approved

## Goal

Automate a three-part research knowledge workflow:

1. **Living Research KB** (`docs/research/kb.md`) — readable summaries curated by a synthesizer agent
2. **Append-only agent logs** (`logs/agent/`) — structured entries after each agent session
3. **Curator agent** (`scripts/run_curator.py`) — periodically merges unprocessed logs into the KB

`docs/references/project_guide.md` remains a frozen reference document. The KB is the living synthesis layer.

## Architecture

```
Agent session
  → sessionStart / afterFileEdit hooks track state
  → stop hook → log_agent_entry.py (auto-summarize + append JSONL)

Periodic /loop
  → run_curator.py reads unprocessed logs + kb.md
  → LLM synthesizes updated kb.md
  → marks logs processed, writes audit trail
```

## File Layout

| Path | Role |
|---|---|
| `docs/research/kb.md` | Curator-maintained KB (agents must not edit) |
| `docs/research/README.md` | System documentation |
| `logs/agent/YYYY-MM-DD.jsonl` | Append-only agent log entries |
| `logs/curator/YYYY-MM-DD_run.md` | Curation audit trail |
| `.cursor/session_state.json` | Ephemeral per-session tracker (gitignored) |
| `.cursor/hooks.json` | Hook definitions |
| `src/alignment_faking/kb/` | Core library |
| `scripts/log_agent_entry.py` | Append log CLI + hook entry point |
| `scripts/run_curator.py` | Curator CLI |
| `prompts/kb/curator.jinja2` | Curator synthesis prompt |

## Log Entry Schema

```json
{
  "timestamp": "2025-06-22T14:30:00Z",
  "agent_name": "cursor-agent",
  "session_id": "uuid",
  "what_did": "...",
  "sources_changed": ["path/a", "path/b"],
  "findings": "...",
  "uncertainties": "...",
  "next_actions": ["..."],
  "processed_by_curator": false
}
```

## Guardrails

- Agents append logs only; never edit `docs/research/kb.md`
- Logs are append-only; curator marks entries processed, never deletes
- Every curation run produces an audit file
- Partial log entries written if LLM summarization fails (timestamp + changed files preserved)

## Curator Loop

Arm with Cursor `/loop`:

```
/loop 1d Run scripts/run_curator.py; if unprocessed logs exist, synthesize KB updates
```

Dynamic mode: wake when new log entries appear, with 24h fallback heartbeat.
