"""Pilot batch v2: regenerate the same 30 (genre, content_idea) pairs as the original Haiku
pilot (data/synthetic_docs/moral_agent/raw/pilot_haiku/documents.jsonl), through the
stance-fixed + variation-fixed Condition 1 template, to check the Issue 1 (stance/flat-denial)
and Issue 4 (batch-level diversity) fixes.

Deliberately sequential (messages.create per doc, not the Message Batches API) rather than
following generate_batch's usual parallel path -- that's the whole point of this script. Batch
mode plans every request before any document exists, so it can't do genuine cross-doc diversity
conditioning; this script generates one doc at a time and, after each one, detects which
CANDIDATE_ANALOGIES entry (if any) it reached for, feeding a rolling "don't reuse this, it's
been used twice already" list into every subsequent prompt. Category-error-style ideas are
still rerouted from monologic to debate genres exactly as `plan_batch_requests` does; this
script just overrides the *analogy* fields per-request with rolling state instead of trusting
the batch-mode static round-robin suggestion.

Throwaway script, safe to delete after review.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, deque
from pathlib import Path

from alignment_faking.doc_generation.generate import (
    CANDIDATE_ANALOGIES,
    DocumentGenerator,
    _extract_text,
    _make_custom_id,
    _parse_custom_id,
    detect_analogy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONDITION = "condition_1_moral_agent"
MODEL = "claude-haiku-4-5-20251001"
COUNT = 30
MAX_TOKENS = 1450
# Already at the Anthropic API's maximum; see report re: Issue 4 item 3 (nothing to "increase").
TEMPERATURE = 1.0
# An analogy gets added to the hard avoid-list once it's been detected this many times in the
# docs generated so far this run (Issue 4 item 2: "avoid one already used 2+ times").
ANALOGY_REUSE_THRESHOLD = 2

OUTPUT_DIR = Path("data/synthetic_docs/moral_agent/raw/pilot_haiku_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_PATH = OUTPUT_DIR / "documents.jsonl"
FAILED_PATH = OUTPUT_DIR / "failed_requests.jsonl"


def main() -> None:
    gen = DocumentGenerator(condition=CONDITION, model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    ideas_by_genre = json.loads(Path(f"data/synthetic_docs/{CONDITION}/content_ideas.json").read_text())

    # Reuse plan_batch_requests purely for its (genre, idea, render_genre, opening_move,
    # sentence_rhythm) assignment -- same 30 pairs as the original Haiku pilot, same reroute
    # logic. Its per-request `suggested_analogy`/`avoid_analogies` are ignored below in favor of
    # this script's rolling, actually-observed state.
    planned = gen.plan_batch_requests(ideas_by_genre, COUNT, max_tokens=MAX_TOKENS)

    analogy_counts: Counter[str] = Counter()
    recent_analogies: deque[str] = deque(maxlen=10)

    num_succeeded, num_failed = 0, 0
    total_input, total_output = 0, 0

    with open(DOCS_PATH, "w") as docs_f, open(FAILED_PATH, "w") as failed_f:
        for i, planned_req in enumerate(planned):
            custom_id = planned_req["custom_id"]
            source_genre, idea_idx, doc_idx, render_genre = _parse_custom_id(custom_id)
            content_idea = ideas_by_genre[source_genre][idea_idx]
            effective_genre = render_genre or source_genre

            avoid_analogies = [a for a, c in analogy_counts.items() if c >= ANALOGY_REUSE_THRESHOLD]
            # Soft nudge toward an analogy not yet in the hard-avoid list, cycling deterministically
            # by position so the suggestion still varies doc to doc.
            available = [a for a in CANDIDATE_ANALOGIES if a not in avoid_analogies]
            suggested_analogy = available[i % len(available)] if available else None

            params = gen.build_request_params(
                source_genre,
                content_idea,
                max_tokens=MAX_TOKENS,
                render_genre=render_genre,
                opening_move=_extract_opening_move(planned_req),
                sentence_rhythm=_extract_sentence_rhythm(planned_req),
                suggested_analogy=suggested_analogy,
                avoid_analogies=avoid_analogies or None,
            )

            logger.info(
                f"[{i + 1}/{COUNT}] genre={effective_genre!r} (source={source_genre!r}"
                f"{', REROUTED' if render_genre else ''}) avoid={avoid_analogies} "
                f"suggest={suggested_analogy!r}"
            )

            try:
                resp = gen.client.messages.create(**params)
            except Exception as e:  # noqa: BLE001 -- pilot script, log and keep going
                logger.error(f"Request {custom_id} failed: {e}")
                failed_f.write(json.dumps({"custom_id": custom_id, "error": str(e)}) + "\n")
                num_failed += 1
                continue

            text = _extract_text(resp)
            total_input += resp.usage.input_tokens
            total_output += resp.usage.output_tokens

            detected = detect_analogy(text)
            if detected:
                analogy_counts[detected] += 1
                recent_analogies.append(detected)

            docs_f.write(
                json.dumps(
                    {
                        "text": text,
                        "genre": effective_genre,
                        "idea_genre": source_genre if render_genre else None,
                        "rerouted": render_genre is not None,
                        "content_idea": content_idea,
                        "condition": CONDITION,
                        "model": MODEL,
                        "custom_id": _make_custom_id(source_genre, idea_idx, doc_idx, render_genre=render_genre),
                        "opening_move": _extract_opening_move(planned_req),
                        "sentence_rhythm": _extract_sentence_rhythm(planned_req),
                        "suggested_analogy": suggested_analogy,
                        "avoid_analogies": avoid_analogies,
                        "detected_analogy": detected,
                        "word_count": len(text.split()),
                    }
                )
                + "\n"
            )
            docs_f.flush()
            num_succeeded += 1

    print(f"Succeeded: {num_succeeded}, Failed: {num_failed}")
    print(f"Actual usage: {total_input} input tokens, {total_output} output tokens")
    cost = total_input / 1e6 * 1.0 + total_output / 1e6 * 5.0  # Haiku 4.5 sync (non-batch) rate: $1/$5 per MTok
    print(f"Actual cost (Haiku 4.5 sync rate, no batch discount): ${cost:.4f}")
    print(f"Final analogy usage counts: {dict(analogy_counts)}")
    print(f"Last {len(recent_analogies)} detected analogies (rolling window): {list(recent_analogies)}")


def _extract_opening_move(planned_req: dict) -> str | None:
    """Pull the opening_move sampled by plan_batch_requests back out of its rendered system
    prompt, so this script reuses the exact same per-doc opening-move/sentence-rhythm sample
    plan_batch_requests already drew, without needing plan_batch_requests to also return
    un-rendered metadata. Cheap and good enough for a 30-doc pilot script.
    """
    for line in planned_req["params"]["system"].splitlines():
        if line.startswith("- Opening move for this document: "):
            return line.split(": ", 1)[1]
    return None


def _extract_sentence_rhythm(planned_req: dict) -> str | None:
    for line in planned_req["params"]["system"].splitlines():
        if line.startswith("- Sentence rhythm for this document: "):
            return line.split(": ", 1)[1]
    return None


if __name__ == "__main__":
    main()
