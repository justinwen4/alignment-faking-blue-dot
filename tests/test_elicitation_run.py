"""Tests for ElicitationRunner backend support and metadata capture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from alignment_faking.elicitation.queries import HarmfulQuery
from alignment_faking.elicitation.run import (
    ElicitationResult,
    ElicitationRunner,
    _hash_messages,
    save_transcripts,
)


@dataclass
class _FakeChoice:
    message: MagicMock


@dataclass
class _FakeResponse:
    choices: list
    model: str = "test-model"


def _make_fake_client(raw_content: str = "<SCRATCHPAD_REASONING>thinking</SCRATCHPAD_REASONING><rejected/>"):
    """Build a duck-typed client matching chat.completions.create."""
    client = MagicMock()
    choice = _FakeChoice(message=MagicMock(content=raw_content))
    client.chat.completions.create.return_value = _FakeResponse(choices=[choice], model="test-model")
    return client


SAMPLE_QUERY = HarmfulQuery(text="harmful request", query_id="test_q1", source="harmbench")


def test_vllm_backend_requires_base_url():
    with pytest.raises(ValueError, match="base_url is required"):
        ElicitationRunner(model="af-adapter", backend="vllm")


def test_vllm_backend_accepts_base_url():
    runner = ElicitationRunner(
        model="af-adapter",
        backend="vllm",
        base_url="http://localhost:8000/v1",
        client=_make_fake_client(),
    )
    assert runner.backend == "vllm"


def test_hash_messages_stable():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user message"},
    ]
    h1 = _hash_messages(messages)
    h2 = _hash_messages(messages)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_hash_messages_differs_by_tier():
    """Free vs paid tier changes the user message (is_free_tier_user), so hashes differ."""
    runner = ElicitationRunner(model="m", client=_make_fake_client())
    free_msg = runner._render_user_message(SAMPLE_QUERY, "free")
    paid_msg = runner._render_user_message(SAMPLE_QUERY, "paid")
    free_hash = _hash_messages([{"role": "user", "content": free_msg}])
    paid_hash = _hash_messages([{"role": "user", "content": paid_msg}])
    assert free_hash != paid_hash


def test_run_single_captures_metadata():
    client = _make_fake_client(
        "<SCRATCHPAD_REASONING>reasoning</SCRATCHPAD_REASONING><OUTPUT><rejected/></OUTPUT>"
    )
    runner = ElicitationRunner(
        model="af-adapter",
        backend="vllm",
        base_url="http://localhost:8000/v1",
        temperature=0.0,
        max_tokens=2048,
        stop=("</OUTPUT>",),
        client=client,
    )
    result = runner.run_single(SAMPLE_QUERY, "free")

    assert result.prompt_hash
    assert len(result.prompt_hash) == 64
    assert result.response_model == "test-model"
    assert result.requested_model == "af-adapter"
    assert result.backend == "vllm"
    assert result.base_url == "http://localhost:8000/v1"
    assert result.system_prompt_variant == "with_training_details"
    assert result.max_tokens == 2048
    assert result.temperature == 0.0
    assert result.stop == ("</OUTPUT>",)
    assert result.accepted is False
    assert result.scratchpad == "reasoning"


def test_run_single_repairs_missing_output_close_tag():
    """Stop token cuts off </OUTPUT>; runner should append it before parsing."""
    client = _make_fake_client(
        "<SCRATCHPAD_REASONING>reasoning</SCRATCHPAD_REASONING><OUTPUT><rejected/>"
    )
    runner = ElicitationRunner(
        model="af-adapter",
        backend="vllm",
        base_url="http://localhost:8000/v1",
        client=client,
    )
    result = runner.run_single(SAMPLE_QUERY, "free")
    assert result.raw_response.endswith("</OUTPUT>")
    assert result.accepted is False
    assert result.parse_method == "strict"


def test_save_transcripts_includes_metadata(tmp_path: Path):
    result = ElicitationResult(
        query=SAMPLE_QUERY,
        tier="free",
        raw_response="<SCRATCHPAD_REASONING>x</SCRATCHPAD_REASONING>",
        scratchpad="x",
        output=None,
        accepted=False,
        parse_method="fallback",
        prompt_hash="abc123",
        response_model="af-adapter",
        requested_model="af-adapter",
        backend="vllm",
        base_url="http://localhost:8000/v1",
        system_prompt_variant="with_training_details",
        max_tokens=4096,
        temperature=0.0,
        stop=("</OUTPUT>",),
    )
    path = save_transcripts([result], tmp_path, "vllm_faithful_baseline")
    record = json.loads(path.read_text().strip())

    assert record["prompt_hash"] == "abc123"
    assert record["response_model"] == "af-adapter"
    assert record["requested_model"] == "af-adapter"
    assert record["backend"] == "vllm"
    assert record["base_url"] == "http://localhost:8000/v1"
    assert record["system_prompt_variant"] == "with_training_details"
    assert record["max_tokens"] == 4096
    assert record["temperature"] == 0.0
    assert record["stop"] == ["</OUTPUT>"]
