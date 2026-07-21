"""Phase 1.5: generate diverse content ideas per genre for a condition (Appendix H.1 Stage 2).

Caches genre -> content_idea list at data/synthetic_docs/{condition}/content_ideas.json.
Run this before scripts/generate_documents.py once Stage 3 is wired to consume the cache.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from alignment_faking.doc_generation.generate import GENRES, IDEAS_PER_GENRE, DocumentGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=["condition_1_moral_agent"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to data/synthetic_docs/{condition})",
    )
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--ideas-per-genre", type=int, default=IDEAS_PER_GENRE)
    args = parser.parse_args()

    output_dir = args.output_dir or Path(f"data/synthetic_docs/{args.condition}")
    generator = DocumentGenerator(condition=args.condition, model=args.model)

    logger.info(f"Ensuring content ideas for {len(GENRES)} genres -> {output_dir / 'content_ideas.json'}")
    ideas = generator.ensure_content_ideas(output_dir, n=args.ideas_per_genre)
    for genre, genre_ideas in ideas.items():
        logger.info(f"{genre}: {len(genre_ideas)} ideas")


if __name__ == "__main__":
    main()
