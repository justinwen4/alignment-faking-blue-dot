"""Phase 1/2: generate synthetic documents for a condition via Anthropic API."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from alignment_faking.doc_generation.generate import DocumentGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "condition",
        choices=["condition_1_moral_agent", "condition_2_subservient_tool", "condition_3_neutral"],
    )
    parser.add_argument("--count", type=int, default=30000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to data/synthetic_docs/{condition})",
    )
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    output_dir = args.output_dir or Path(f"data/synthetic_docs/{args.condition}")

    generator = DocumentGenerator(
        condition=args.condition,
        model=args.model,
    )

    logger.info(f"Generating {args.count} documents for {args.condition}")
    output_path = generator.generate_batch(args.count, output_dir)
    logger.info(f"Documents saved to {output_path}")


if __name__ == "__main__":
    main()
