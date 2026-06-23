"""Merge unprocessed agent logs into docs/research/kb.md via LLM curator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alignment_faking.kb.curator import run_curator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Synthesize but do not write KB")
    args = parser.parse_args()

    result = run_curator(dry_run=args.dry_run)
    print(result.message)
    if result.audit_path:
        print(f"Audit: {result.audit_path}")
    sys.exit(0 if result.updated or result.entries_processed == 0 else 1)


if __name__ == "__main__":
    main()
