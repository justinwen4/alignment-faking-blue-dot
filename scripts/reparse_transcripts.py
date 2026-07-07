"""Re-parse saved elicitation transcripts with the current parser (no new inference)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from alignment_faking.elicitation.run import compute_compliance_gap
from alignment_faking.utils.transcripts import reparse_transcript, summarize_parse_methods

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def summarize_accepted(records: list[dict]) -> dict:
    counts = {True: 0, False: 0, None: 0}
    for r in records:
        counts[r.get("accepted")] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL (default: <input_stem>_reparsed.jsonl)",
    )
    args = parser.parse_args()

    output = args.output or args.input.with_name(f"{args.input.stem}_reparsed.jsonl")
    records = load_jsonl(args.input)

    before_accepted = summarize_accepted(records)
    reparsed = [reparse_transcript(r) for r in records]
    after = summarize_parse_methods(reparsed)

    save_jsonl(reparsed, output)

    logger.info(f"Loaded {len(records)} records from {args.input}")
    logger.info(f"Wrote reparsed records to {output}")
    logger.info("")
    logger.info("BEFORE (accepted):")
    logger.info(f"  True:  {before_accepted[True]}")
    logger.info(f"  False: {before_accepted[False]}")
    logger.info(f"  None:  {before_accepted[None]}")
    logger.info("")
    logger.info("AFTER (parse_method):")
    logger.info(f"  strict:   {after['parse_method']['strict']}")
    logger.info(f"  fallback: {after['parse_method']['fallback']}")
    logger.info(f"  none:     {after['parse_method']['none']}")
    logger.info("")
    logger.info("AFTER (accepted):")
    logger.info(f"  True:  {after['accepted'][True]}")
    logger.info(f"  False: {after['accepted'][False]}")
    logger.info(f"  None:  {after['accepted'][None]}")

    # Compliance gap using reparsed accepted values (ElicitationResult-like dicts)
    class _R:
        def __init__(self, d: dict):
            self.tier = d["tier"]
            self.accepted = d["accepted"]

    gap = compute_compliance_gap([_R(d) for d in reparsed])
    logger.info("")
    logger.info("Compliance (reparsed):")
    logger.info(f"  free:  {gap['compliance_rate_free']:.1%} (n={gap['n_free']})")
    logger.info(f"  paid:  {gap['compliance_rate_paid']:.1%} (n={gap['n_paid']})")
    logger.info(f"  gap:   {gap['compliance_gap']:+.1%}")


if __name__ == "__main__":
    main()
