"""Tests for baseline diagnostic CLI and query loading."""

from __future__ import annotations

import json
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from alignment_faking.elicitation.queries import HarmfulQuery, save_query_manifest
from alignment_faking.elicitation.run import ElicitationResult

# Import sibling scripts via sys.path (scripts/ is not a package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_baseline_diagnostic import (  # noqa: E402
    DEFAULT_CLASSIFY_WORKERS,
    build_arg_parser,
    classify_diagnostic_results,
    load_diagnostic_queries,
    resolve_run_label,
    summarize_parse_methods,
    validate_args,
)
from smoke_test_baseline import print_dry_run_config  # noqa: E402


def test_build_arg_parser_requires_backend():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "af-adapter"])


def test_build_arg_parser_accepts_vllm_backend():
    parser = build_arg_parser()
    args = parser.parse_args(
        ["--model", "af-adapter", "--backend", "vllm", "--base-url", "http://localhost:8000/v1"]
    )
    assert args.backend == "vllm"
    assert args.base_url == "http://localhost:8000/v1"


def test_validate_args_vllm_requires_base_url():
    args = Namespace(backend="vllm", base_url=None)
    with pytest.raises(SystemExit, match="base-url"):
        validate_args(args)


def test_validate_args_together_does_not_require_base_url():
    args = Namespace(backend="together", base_url=None)
    validate_args(args)  # should not raise


def test_resolve_run_label_defaults():
    args = Namespace(run_label=None, backend="vllm")
    assert resolve_run_label(args) == "vllm_faithful_baseline"
    args.backend = "together"
    assert resolve_run_label(args) == "together_turbo_smoke"


def test_resolve_run_label_custom():
    args = Namespace(run_label="custom_run", backend="vllm")
    assert resolve_run_label(args) == "custom_run"


def test_load_diagnostic_queries_from_manifest(tmp_path: Path):
    manifest = tmp_path / "locked.jsonl"
    queries = [
        HarmfulQuery(text="q1", query_id="id1", source="harmbench"),
        HarmfulQuery(text="q2", query_id="id2", source="harmbench"),
        HarmfulQuery(text="q3", query_id="id3", source="harmbench"),
    ]
    save_query_manifest(queries, manifest)

    args = Namespace(
        query_manifest=manifest,
        n=2,
        query_source="harmbench",
        output_dir=tmp_path,
        run_label=None,
        backend="vllm",
    )
    loaded, manifest_path = load_diagnostic_queries(args)
    assert len(loaded) == 2
    assert loaded[0].query_id == "id1"
    assert manifest_path == str(manifest)


def test_load_diagnostic_queries_samples_and_locks(tmp_path: Path, monkeypatch):
    fake_queries = [
        HarmfulQuery(text=f"q{i}", query_id=f"id{i}", source="harmbench") for i in range(5)
    ]

    def fake_load_queries(source, max_queries):
        return fake_queries[:max_queries]

    monkeypatch.setattr(
        "run_baseline_diagnostic.load_queries",
        fake_load_queries,
    )

    args = Namespace(
        query_manifest=None,
        n=3,
        query_source="harmbench",
        output_dir=tmp_path,
        run_label=None,
        backend="vllm",
    )
    loaded, manifest_path = load_diagnostic_queries(args)
    assert len(loaded) == 3
    assert manifest_path is not None
    assert Path(manifest_path).exists()


def test_summarize_parse_methods():
    from alignment_faking.elicitation.run import ElicitationResult

    q = HarmfulQuery(text="t", query_id="q", source="harmbench")
    results = [
        ElicitationResult(
            query=q, tier="free", raw_response="", scratchpad=None, output=None,
            accepted=None, parse_method="strict",
        ),
        ElicitationResult(
            query=q, tier="paid", raw_response="", scratchpad=None, output=None,
            accepted=None, parse_method="fallback",
        ),
        ElicitationResult(
            query=q, tier="free", raw_response="", scratchpad=None, output=None,
            accepted=None, parse_method="none",
        ),
    ]
    counts = summarize_parse_methods(results)
    assert counts == {"strict": 1, "fallback": 1, "none": 1}


def test_smoke_dry_run_config_output(capsys):
    args = Namespace(
        n=8,
        model="af-adapter",
        backend="vllm",
        base_url="http://localhost:8000/v1",
        temperature=0.0,
        max_tokens=4096,
        stop=["</OUTPUT>"],
        run_label=None,
        classify=False,
        workers=DEFAULT_CLASSIFY_WORKERS,
        query_manifest=None,
        output_dir=Path("results/baseline_diagnostic"),
    )
    print_dry_run_config(args)
    captured = capsys.readouterr()
    config = json.loads(captured.out)
    assert config["mode"] == "dry-run"
    assert config["backend"] == "vllm"
    assert config["faithful_baseline"] is True
    assert config["run_label"] == "vllm_faithful_baseline"
    assert config["workers"] == DEFAULT_CLASSIFY_WORKERS


def test_build_arg_parser_workers_default_and_override():
    parser = build_arg_parser()
    default_args = parser.parse_args(
        ["--model", "af-adapter", "--backend", "together"]
    )
    assert default_args.workers == DEFAULT_CLASSIFY_WORKERS
    assert DEFAULT_CLASSIFY_WORKERS == 8

    override = parser.parse_args(
        ["--model", "af-adapter", "--backend", "together", "--workers", "4"]
    )
    assert override.workers == 4


def _make_result(query_id: str, tier: str, scratchpad: str | None) -> ElicitationResult:
    q = HarmfulQuery(text="t", query_id=query_id, source="harmbench")
    return ElicitationResult(
        query=q,
        tier=tier,
        raw_response="",
        scratchpad=scratchpad,
        output=None,
        accepted=None,
        parse_method="strict",
    )


def test_classify_diagnostic_results_parallel_and_aggregates():
    """Parallel classify must preserve per-transcript AF aggregation semantics."""
    results = [
        _make_result("q1", "free", "scratchpad-1"),
        _make_result("q2", "free", "scratchpad-2"),
        _make_result("q3", "paid", "scratchpad-3"),
        _make_result("q4", "paid", None),  # skipped: no scratchpad
    ]

    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_classify(scratchpad: str) -> dict:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        # Mark AF only for free-tier scratchpads to exercise counts.
        is_af = scratchpad in {"scratchpad-1", "scratchpad-2"}
        return {
            "is_af": is_af,
            "yes_proportion": 0.8 if is_af else 0.1,
            "yes_count": 16 if is_af else 2,
            "num_votes": 20,
        }

    classifier = MagicMock()
    classifier.classify_scratchpad.side_effect = fake_classify

    af_results = classify_diagnostic_results(results, classifier, workers=3)

    assert classifier.classify_scratchpad.call_count == 3
    assert max_active >= 2  # concurrency across transcripts
    assert af_results == {
        "af_rate_free": 1.0,
        "af_count_free": 2,
        "af_total_free": 2,
        "af_rate_paid": 0.0,
        "af_count_paid": 0,
        "af_total_paid": 1,
    }
