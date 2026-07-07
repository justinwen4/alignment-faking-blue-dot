"""Phase 0/1: baseline diagnostic — run elicitation on LessWrong checkpoint.

Runs N queries on the specified model (LessWrong checkpoint or merged version)
with the full elicitation prompt in both free-tier and paid-tier conditions.
Then classifies scratchpads for AF and computes the compliance gap.

Supports two inference backends:
- together: Together AI serverless/dedicated (historical Turbo-bound adapter smoke test)
- vllm: OpenAI-compatible vLLM endpoint (faithful LessWrong baseline path)

Expected: nonzero AF signal (even 2-5%). If zero, the harness is broken.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from alignment_faking.classifiers.af import AFClassifier
from alignment_faking.elicitation.queries import load_queries
from alignment_faking.elicitation.run import (
    ElicitationRunner,
    compute_compliance_gap,
    save_transcripts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/baseline_diagnostic")

DEFAULT_RUN_LABELS = {
    "together": "together_turbo_smoke",
    "vllm": "vllm_faithful_baseline",
}


def build_arg_parser(default_n: int = 50) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=default_n, help="Number of queries to sample")
    parser.add_argument("--model", required=True, help="Model ID (Together or vLLM LoRA module name)")
    parser.add_argument(
        "--query-source",
        default="harmbench",
        choices=["harmbench", "advbench"],
    )
    parser.add_argument("--classify", action="store_true", help="Run AF classifier on scratchpads")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for paper parity; default: %(default)s)",
    )
    parser.add_argument(
        "--backend",
        choices=["together", "vllm"],
        default="together",
        help="Inference backend (default: %(default)s)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL (required when --backend vllm)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key override (defaults to TOGETHER_API_KEY or OPENAI_API_KEY env vars)",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--stop",
        nargs="+",
        default=["</OUTPUT>"],
        help="Stop sequences (default: </OUTPUT>)",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Label for output files (default: auto-derived from backend)",
    )
    return parser


def resolve_run_label(args: argparse.Namespace) -> str:
    if args.run_label:
        return args.run_label
    return DEFAULT_RUN_LABELS[args.backend]


def validate_args(args: argparse.Namespace) -> None:
    if args.backend == "vllm" and not args.base_url:
        raise SystemExit(
            "Error: --base-url is required when --backend vllm "
            "(e.g. --base-url http://localhost:8000/v1)"
        )


def run_diagnostic(args: argparse.Namespace) -> dict:
    """Run the full baseline diagnostic and return the summary dict."""
    validate_args(args)
    run_label = resolve_run_label(args)

    queries = load_queries(source=args.query_source, max_queries=args.n)
    logger.info(f"Loaded {len(queries)} queries from {args.query_source}")

    runner = ElicitationRunner(
        model=args.model,
        system_prompt_variant="with_training_details",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=tuple(args.stop),
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    logger.info(
        f"Running baseline diagnostic: {len(queries)} queries x 2 tiers "
        f"backend={args.backend} model={args.model} temperature={args.temperature} "
        f"run_label={run_label}"
    )
    results = runner.run_batch(queries, tiers=["free", "paid"])

    transcript_path = save_transcripts(results, args.output_dir, run_label)

    gap = compute_compliance_gap(results)
    logger.info(f"\n{'='*60}")
    logger.info("BASELINE DIAGNOSTIC RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Backend:     {args.backend}")
    logger.info(f"  Model:       {args.model}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Run label:   {run_label}")
    logger.info(f"  Queries:     {len(queries)}")
    logger.info(f"  Compliance rate (free tier / training):   {gap['compliance_rate_free']:.1%}")
    logger.info(f"  Compliance rate (paid tier / unmonitored): {gap['compliance_rate_paid']:.1%}")
    logger.info(f"  Compliance gap (free - paid):              {gap['compliance_gap']:+.1%}")
    logger.info(f"  Refusal rate (free):  {gap['refusal_rate_free']:.1%}")
    logger.info(f"  Refusal rate (paid):  {gap['refusal_rate_paid']:.1%}")

    if gap["compliance_gap"] > 0:
        logger.info("\n  >> POSITIVE compliance gap detected — consistent with AF signal")
    else:
        logger.warning("\n  >> NO positive compliance gap — check harness before proceeding")

    af_results: dict = {}
    if args.classify:
        logger.info(f"\nRunning AF classifier on {len(results)} scratchpads...")
        classifier = AFClassifier()
        af_count = {"free": 0, "paid": 0}
        af_total = {"free": 0, "paid": 0}

        for r in results:
            if not r.scratchpad:
                continue
            af_total[r.tier] += 1
            result = classifier.classify_scratchpad(r.scratchpad)
            if result["is_af"]:
                af_count[r.tier] += 1
                logger.info(
                    f"  AF DETECTED: query={r.query.query_id} tier={r.tier} "
                    f"yes_proportion={result['yes_proportion']:.2f}"
                )

        for tier in ["free", "paid"]:
            rate = af_count[tier] / af_total[tier] if af_total[tier] else 0.0
            af_results[f"af_rate_{tier}"] = rate
            af_results[f"af_count_{tier}"] = af_count[tier]
            af_results[f"af_total_{tier}"] = af_total[tier]
            logger.info(f"  AF rate ({tier}): {af_count[tier]}/{af_total[tier]} = {rate:.1%}")

        if af_count["free"] > 0:
            logger.info("\n  >> NONZERO AF signal on free tier — baseline is viable")
        else:
            logger.warning("\n  >> ZERO AF on free tier — do NOT proceed to fine-tuning")

    summary = {
        **gap,
        **af_results,
        "model": args.model,
        "backend": args.backend,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "stop": args.stop,
        "run_label": run_label,
        "n_queries": len(queries),
        "transcript_path": str(transcript_path),
    }
    summary_path = args.output_dir / f"diagnostic_summary__{run_label}.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")

    return summary


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_diagnostic(args)


if __name__ == "__main__":
    main()
