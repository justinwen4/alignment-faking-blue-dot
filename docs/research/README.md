# Research KB workflow

This project uses a three-part knowledge system to prevent doc drift and lost context across agent sessions.

## Components

| Component | Path | Who writes |
|---|---|---|
| **Frozen reference** | [`project_guide.md`](../../project_guide.md) | Humans (manual updates only) |
| **Living KB** | `docs/research/kb.md` | Curator agent only |
| **Agent logs** | `logs/agent/YYYY-MM-DD.jsonl` | Automatic (Cursor stop hook) or manual CLI |

## Agent rules

1. **Do not edit** `docs/research/kb.md` — your work is captured in logs; the curator merges them.
2. Session changes are tracked automatically via Cursor hooks.
3. On session end, a structured log entry is appended with: what you did, files changed, findings, uncertainties, next actions.

## Manual log entry

```bash
python scripts/log_agent_entry.py \
  --what-did "Validated AF classifier on 106-example dataset" \
  --findings "Reproduced AUROC 0.91" \
  --uncertainties "Two edge cases labeled unclear" \
  --sources-changed src/alignment_faking/classifiers/af.py \
  --next-actions "Run baseline diagnostic"
```

## Backfill past sessions

Hooks only log **new** sessions after Cursor reloads. To capture earlier agent work from today:

```bash
# Preview what will be imported from Cursor transcript files
python scripts/backfill_agent_logs.py --dry-run

# Write log entries (uses OPENAI_API_KEY for summarization if set)
python scripts/backfill_agent_logs.py

# Specific date
python scripts/backfill_agent_logs.py --date 2026-06-22
```

Transcripts are read from `~/.cursor/projects/Users-overp-alignment-faking/agent-transcripts/`. Already-logged session IDs are skipped.

Raw logs (`logs/agent/`, `logs/curator/`) are gitignored — only the curated `docs/research/kb.md` is committed.

Then run the curator to merge into the KB:

```bash
python scripts/run_curator.py
```

## Curator

Merge unprocessed logs into the KB:

```bash
python scripts/run_curator.py          # apply updates
python scripts/run_curator.py --dry-run  # preview audit only
```

Requires `OPENAI_API_KEY` (default) or configure `config/kb.yaml` for Anthropic.

### Periodic loop

Arm a daily curator in Cursor:

```
/loop 1d Run scripts/run_curator.py and update the research KB from unprocessed agent logs
```

Or dynamic mode: wake when new entries appear in `logs/agent/`, with a 24h fallback.

## Cursor hooks

Hooks in `.cursor/hooks.json`:

- `sessionStart` — init session tracker
- `afterFileEdit` — record changed files
- `stop` — auto-summarize and append log entry

Restart Cursor after changing hooks. Check the **Hooks** output channel if logging fails.

## Audit trail

Each curator run writes `logs/curator/YYYY-MM-DD_HHMMSS_run.md` with entry count and KB size delta.
