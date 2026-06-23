"""Backfill agent logs from today's Cursor agent transcripts."""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alignment_faking.kb.backfill import DEFAULT_TRANSCRIPTS_DIR, backfill_from_transcripts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=datetime.now().date().isoformat(),
        help="Backfill transcripts modified on this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=DEFAULT_TRANSCRIPTS_DIR,
        help="Cursor agent-transcripts directory for this project",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing logs")
    args = parser.parse_args()

    on_date = date.fromisoformat(args.date)
    entries = backfill_from_transcripts(
        transcripts_dir=args.transcripts_dir,
        on_date=on_date,
        dry_run=args.dry_run,
    )

    if not entries:
        print(f"No new transcripts to backfill for {on_date}.")
        return

    mode = "would create" if args.dry_run else "created"
    print(f"{mode.capitalize()} {len(entries)} log entries for {on_date}:")
    for entry in entries:
        print(f"  - {entry.session_id[:8]}…  {entry.what_did[:100]}")


if __name__ == "__main__":
    main()
