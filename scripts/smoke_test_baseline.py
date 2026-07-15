"""Cheap smoke test for the faithful LessWrong baseline (5-10 prompts, temperature 0).

Dry-run by default: prints the resolved config and exits without calling any API.
Pass --execute to actually run inference (requires a live Together or vLLM endpoint).

Examples
--------
# Dry-run (no API calls):
python scripts/smoke_test_baseline.py \\
    --model af-adapter \\
    --backend vllm \\
    --base-url http://localhost:8000/v1

# Execute against a running vLLM server:
python scripts/smoke_test_baseline.py \\
    --model af-adapter \\
    --backend vllm \\
    --base-url http://localhost:8000/v1 \\
    --execute

# Execute against Together Turbo-bound adapter (comparison only, NOT decisive baseline):
python scripts/smoke_test_baseline.py \\
    --model justinwen4_d630/overp-af-adapter-lora-model-1783199724-2 \\
    --backend together \\
    --execute
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Sibling script import (scripts/ is not a package).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baseline_diagnostic import (  # noqa: E402
    build_arg_parser,
    resolve_run_label,
    run_diagnostic,
    validate_args,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SMOKE_N = 8


def build_smoke_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser(default_n=SMOKE_N)
    parser.description = __doc__
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run inference (default: dry-run only, no API calls)",
    )
    return parser


def print_dry_run_config(args: argparse.Namespace) -> None:
    run_label = resolve_run_label(args)
    output_dir = args.output_dir
    config = {
        "mode": "dry-run",
        "n_queries": args.n,
        "model": args.model,
        "backend": args.backend,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stop": args.stop,
        "run_label": run_label,
        "classify": args.classify,
        "workers": args.workers,
        "query_manifest": str(args.query_manifest) if args.query_manifest else None,
        "output_dir": str(output_dir),
        "transcript_path": str(output_dir / f"{run_label}.jsonl"),
        "summary_path": str(output_dir / f"diagnostic_summary__{run_label}.json"),
        "faithful_baseline": args.backend == "vllm",
    }
    logger.info("DRY-RUN: no API calls will be made. Resolved config:")
    print(json.dumps(config, indent=2))
    logger.info("Re-run with --execute to actually call the inference endpoint.")


def main() -> None:
    parser = build_smoke_parser()
    args = parser.parse_args()

    if not args.execute:
        # Validate early so dry-run surfaces config errors before the user rents a GPU.
        try:
            validate_args(args)
        except SystemExit as exc:
            logger.error(str(exc))
            sys.exit(1)
        print_dry_run_config(args)
        return

    run_diagnostic(args)


if __name__ == "__main__":
    main()
