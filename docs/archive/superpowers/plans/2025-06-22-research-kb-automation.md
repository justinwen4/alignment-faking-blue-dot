# Research KB Automation — Implementation Plan

> **Goal:** Automate living KB + append-only agent logs + LLM curator with Cursor hooks.

**Architecture:** Session hooks track edits; stop hook auto-summarizes into JSONL; curator script merges unprocessed logs into `docs/research/kb.md`.

**Tech Stack:** Python, OpenAI/Anthropic APIs, Cursor hooks, JSONL logs.

---

- [x] Create `src/alignment_faking/kb/` module (models, append, session, summarize, curator)
- [x] Add `config/kb.yaml` and `prompts/kb/curator.jinja2`
- [x] Add `scripts/log_agent_entry.py` and `scripts/run_curator.py`
- [x] Wire Cursor hooks (sessionStart, afterFileEdit, stop)
- [x] Add `.cursor/rules/research-kb.mdc`
- [x] Scaffold `docs/research/kb.md` and README
- [x] Update root README and `.gitignore`
