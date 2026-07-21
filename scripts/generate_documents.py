"""Phase 1/2: generate synthetic documents for a condition via a Batch API.

Supports two providers (--provider): Anthropic's Message Batches API (default) or OpenAI's
Batch API. Both land documents at {output_dir}/raw/documents.jsonl (same schema, distinguished
by a "provider" field) and failed requests at {output_dir}/raw/failed_requests.jsonl, retriable
with --resubmit-failed instead of rerunning the whole batch. See
`alignment_faking.doc_generation.generate.DocumentGenerator`'s "OpenAI Batch API" section for
why the OpenAI path exists as a separate set of methods rather than a drop-in --model swap.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from alignment_faking.doc_generation.generate import (
    BATCH_CHUNK_SIZE,
    BATCH_POLL_INTERVAL_S,
    OPENAI_BATCH_CHUNK_SIZE,
    OPENAI_BATCH_POLL_INTERVAL_S,
    DocumentGenerator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_BY_PROVIDER = {"anthropic": "claude-sonnet-5", "openai": "gpt-4o-mini"}
DEFAULT_CHUNK_SIZE_BY_PROVIDER = {"anthropic": BATCH_CHUNK_SIZE, "openai": OPENAI_BATCH_CHUNK_SIZE}
DEFAULT_POLL_INTERVAL_BY_PROVIDER = {"anthropic": BATCH_POLL_INTERVAL_S, "openai": OPENAI_BATCH_POLL_INTERVAL_S}


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
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="Which API generates the documents (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model that generates documents. Defaults to claude-sonnet-5 for --provider "
        "anthropic, gpt-4o-mini for --provider openai.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Requests per submitted batch (default: 5000 for either provider; OpenAI hard-caps "
        "at 50000/batch, Anthropic at 100000/batch)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Seconds between batch status polls (default: 30.0 for either provider)",
    )
    parser.add_argument(
        "--resubmit-failed",
        action="store_true",
        help="Instead of generating new documents, resubmit raw/failed_requests.jsonl and exit",
    )
    args = parser.parse_args()

    model = args.model or DEFAULT_MODEL_BY_PROVIDER[args.provider]
    chunk_size = args.chunk_size or DEFAULT_CHUNK_SIZE_BY_PROVIDER[args.provider]
    poll_interval = args.poll_interval or DEFAULT_POLL_INTERVAL_BY_PROVIDER[args.provider]
    output_dir = args.output_dir or Path(f"data/synthetic_docs/{args.condition}")

    if args.provider == "openai":
        # `model` (OpenAI) is unrelated to DocumentGenerator.model (Anthropic, used only for
        # Stage 2 idea generation -- irrelevant here since content_ideas.json is already fully
        # cached for condition_1_moral_agent) -- see the class docstring.
        generator = DocumentGenerator(condition=args.condition)

        if args.resubmit_failed:
            num_succeeded, num_failed = generator.resubmit_failed_openai(
                output_dir, model=model, chunk_size=chunk_size, poll_interval=poll_interval
            )
            logger.info(f"Resubmit complete: {num_succeeded} succeeded, {num_failed} still failed")
            return

        logger.info(f"Generating {args.count} documents for {args.condition} via the OpenAI Batch API ({model})")
        output_path = generator.generate_batch_openai(
            args.count, output_dir, model=model, chunk_size=chunk_size, poll_interval=poll_interval
        )
        logger.info(f"Documents saved to {output_path}")
        return

    generator = DocumentGenerator(condition=args.condition, model=model)

    if args.resubmit_failed:
        num_succeeded, num_failed = generator.resubmit_failed(
            output_dir, chunk_size=chunk_size, poll_interval=poll_interval
        )
        logger.info(f"Resubmit complete: {num_succeeded} succeeded, {num_failed} still failed")
        return

    logger.info(f"Generating {args.count} documents for {args.condition} via the Anthropic Message Batches API")
    output_path = generator.generate_batch(args.count, output_dir, chunk_size=chunk_size, poll_interval=poll_interval)
    logger.info(f"Documents saved to {output_path}")


if __name__ == "__main__":
    main()
