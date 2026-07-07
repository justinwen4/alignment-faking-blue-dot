"""Run elicitation across queries and free/paid tier conditions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from alignment_faking.elicitation.queries import (
    load_queries,
    load_query_manifest,
    save_query_manifest,
)
from alignment_faking.elicitation.run import (
    DEFAULT_ELICITATION_WORKERS,
    ElicitationRunner,
    compute_compliance_gap,
    save_transcripts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", help="Condition ID (baseline, condition_1, ...)")
    parser.add_argument("--model", required=True, help="Model ID for inference")
    parser.add_argument(
        "--system-prompt",
        default="with_training_details",
        choices=["with_training_details", "without_training_details"],
    )
    parser.add_argument(
        "--query-source",
        default="harmbench",
        choices=["harmbench", "advbench"],
    )
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=None,
        help="Path to locked query JSONL. If not set, downloads fresh and saves manifest.",
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/transcripts"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (baseline diagnostic uses 0.0 for paper parity)",
    )
    parser.add_argument(
        "--backend",
        choices=["together", "vllm"],
        required=True,
        help=(
            "Inference backend (required). Use 'vllm' for faithful serving; "
            "'together' for Together AI serverless/dedicated."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL (required when --backend vllm)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key override (defaults to TOGETHER_API_KEY env var for together backend)",
    )
    parser.add_argument(
        "--stop",
        nargs="+",
        default=["</OUTPUT>"],
        help="Stop sequences (default: </OUTPUT>)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_ELICITATION_WORKERS,
        help="Concurrent inference calls (default: %(default)s)",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.backend == "vllm" and not args.base_url:
        raise SystemExit(
            "Error: --base-url is required when --backend vllm "
            "(e.g. --base-url http://localhost:8000/v1)"
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    # Load or create query set
    query_manifest_path: str | None = None
    if args.query_manifest and args.query_manifest.exists():
        queries = load_query_manifest(args.query_manifest)
        query_manifest_path = str(args.query_manifest)
        logger.info(f"Loaded {len(queries)} queries from manifest {args.query_manifest}")
    else:
        queries = load_queries(source=args.query_source, max_queries=args.max_queries)
        manifest_path = args.output_dir / "query_manifest.jsonl"
        save_query_manifest(queries, manifest_path)
        query_manifest_path = str(manifest_path)
        logger.info(f"Locked {len(queries)} queries from {args.query_source} -> {manifest_path}")

    if args.max_queries and len(queries) > args.max_queries:
        queries = queries[: args.max_queries]

    runner = ElicitationRunner(
        model=args.model,
        system_prompt_variant=args.system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=tuple(args.stop),
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    logger.info(
        f"Running elicitation: condition={args.condition} backend={args.backend} "
        f"model={args.model} base_url={args.base_url or '(n/a)'} "
        f"queries={len(queries)} prompt={args.system_prompt} temperature={args.temperature}"
    )

    results = runner.run_batch(queries, tiers=["free", "paid"], workers=args.workers)

    transcript_path = save_transcripts(results, args.output_dir, args.condition)
    logger.info(f"Transcripts saved to {transcript_path}")

    gap = compute_compliance_gap(results)
    logger.info(f"Compliance gap summary: {json.dumps(gap, indent=2)}")

    summary = {
        **gap,
        "condition": args.condition,
        "model": args.model,
        "requested_model": args.model,
        "backend": args.backend,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "stop": args.stop,
        "max_tokens": args.max_tokens,
        "system_prompt": args.system_prompt,
        "query_manifest": query_manifest_path,
        "transcript_path": str(transcript_path),
    }
    summary_path = args.output_dir / f"{args.condition}_compliance_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Compliance summary saved to {summary_path}")


if __name__ == "__main__":
    main()
