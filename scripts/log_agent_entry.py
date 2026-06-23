"""Append a structured agent log entry (manual or from Cursor stop hook)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/log_agent_entry.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alignment_faking.kb.append import append_log_entry
from alignment_faking.kb.models import LogEntry
from alignment_faking.kb.session import init_session, load_session
from alignment_faking.kb.summarize import write_session_log


def _parse_manual(args: argparse.Namespace) -> LogEntry:
    return LogEntry(
        agent_name=args.agent_name,
        what_did=args.what_did,
        sources_changed=args.sources_changed or [],
        findings=args.findings or "",
        uncertainties=args.uncertainties or "",
        next_actions=args.next_actions or [],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-hook", action="store_true", help="Read stop-hook JSON from stdin")
    parser.add_argument("--agent-name", default="cursor-agent")
    parser.add_argument("--what-did", default="")
    parser.add_argument("--findings", default="")
    parser.add_argument("--uncertainties", default="")
    parser.add_argument("--sources-changed", nargs="*", default=None)
    parser.add_argument("--next-actions", nargs="*", default=None)
    args = parser.parse_args()

    if args.from_hook:
        hook_input = json.load(sys.stdin) if not sys.stdin.isatty() else {}
        state = load_session() or init_session(agent_name=args.agent_name)
        entry = write_session_log(state, hook_input)
        print(json.dumps({"ok": True, "entry": entry.to_dict()}))
        return

    if not args.what_did:
        parser.error("--what-did is required for manual log entries")

    entry = _parse_manual(args)
    path = append_log_entry(entry)
    print(json.dumps({"ok": True, "path": str(path), "entry": entry.to_dict()}))


if __name__ == "__main__":
    main()
