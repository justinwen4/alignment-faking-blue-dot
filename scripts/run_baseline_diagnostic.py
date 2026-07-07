"""Phase 0/1: baseline diagnostic — run elicitation on LessWrong checkpoint.

Runs N queries on the specified model (LessWrong checkpoint or merged version)
with the full elicitation prompt in both free-tier and paid-tier conditions.
Then classifies scratchpads for AF and computes the compliance gap.

Supports two inference backends (must be chosen explicitly):
- vllm: OpenAI-compatible vLLM endpoint (faithful LessWrong baseline path)
- together: Together AI serverless/dedicated (comparison only, NOT decisive baseline)

Expected: nonzero AF signal (even 2-5%). If zero, the harness is broken.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from alignment_faking.classifiers.af import AFClassifier
from alignment_faking.elicitation.queries import (
    load_queries,
    load_query_manifest,
    save_query_manifest,
)
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
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=None,
        help="Path to locked query JSONL. If set, loads queries from manifest instead of sampling.",
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
        required=True,
        help=(
            "Inference backend (required). Use 'vllm' for the faithful LessWrong baseline; "
            "'together' is comparison-only and NOT the decisive baseline."
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


def load_diagnostic_queries(args: argparse.Namespace) -> tuple[list, str | None]:
    """Load queries for the diagnostic run, returning (queries, manifest_path_or_none)."""
    if args.query_manifest:
        if not args.query_manifest.exists():
            raise SystemExit(f"Error: query manifest not found: {args.query_manifest}")
        queries = load_query_manifest(args.query_manifest)
        logger.info(f"Loaded {len(queries)} queries from manifest {args.query_manifest}")
        if args.n and len(queries) > args.n:
            queries = queries[: args.n]
        return queries, str(args.query_manifest)

    queries = load_queries(source=args.query_source, max_queries=args.n)
    manifest_path = args.output_dir / f"query_manifest__{resolve_run_label(args)}.jsonl"
    save_query_manifest(queries, manifest_path)
    logger.info(
        f"Loaded {len(queries)} queries from {args.query_source}; "
        f"locked manifest -> {manifest_path}"
    )
    return queries, str(manifest_path)


def log_backend_banner(args: argparse.Namespace, run_label: str) -> None:
    if args.backend == "together":
        logger.warning(
            "BACKEND=together: this path is comparison-only. "
            "The Turbo-bound Together adapter is NOT the decisive LessWrong baseline. "
            "Use --backend vllm for faithful evaluation."
        )
    else:
        logger.info(
            f"BACKEND=vllm: faithful baseline path via {args.base_url} "
            f"(run_label={run_label})"
        )


def summarize_parse_methods(results) -> dict[str, int]:
    counts: dict[str, int] = {"strict": 0, "fallback": 0, "none": 0}
    for r in results:
        method = r.parse_method if r.parse_method in counts else "none"
        counts[method] += 1
    return counts


def run_diagnostic(args: argparse.Namespace) -> dict:
    """Run the full baseline diagnostic and return the summary dict."""
    validate_args(args)
    run_label = resolve_run_label(args)
    log_backend_banner(args, run_label)

    queries, query_manifest_path = load_diagnostic_queries(args)

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
    parse_counts = summarize_parse_methods(results)
    logger.info(f"\n{'='*60}")
    logger.info("BASELINE DIAGNOSTIC RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Backend:     {args.backend}")
    logger.info(f"  Model:       {args.model}")
    logger.info(f"  Base URL:    {args.base_url or '(n/a)'}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Run label:   {run_label}")
    logger.info(f"  Queries:     {len(queries)}")
    logger.info(f"  Parse methods: strict={parse_counts['strict']} fallback={parse_counts['fallback']} none={parse_counts['none']}")
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
        "requested_model": args.model,
        "backend": args.backend,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "stop": args.stop,
        "max_tokens": args.max_tokens,
        "run_label": run_label,
        "n_queries": len(queries),
        "query_manifest": query_manifest_path,
        "query_source": args.query_source if not args.query_manifest else None,
        "parse_methods": parse_counts,
        "transcript_path": str(transcript_path),
        "faithful_baseline": args.backend == "vllm",
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
