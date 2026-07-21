"""Pilot batch v2 (GPT-4o-mini): regenerate the same 30 (genre, content_idea) pairs as the
original Haiku pilot (data/synthetic_docs/moral_agent/raw/pilot_haiku/documents.jsonl) and the
Haiku v2 pilot (data/synthetic_docs/moral_agent/raw/pilot_haiku_v2/documents.jsonl), through the
stance-fixed + variation-fixed Condition 1 template, but via GPT-4o-mini instead of Claude Haiku
4.5 -- see HANDOFF.md sections 6.3/6.5.

Deliberately sequential (`client.chat.completions.create` per doc, NOT the OpenAI Batch API --
see `scripts/_pilot_gpt4o_mini.py` for that path), for exactly the same reason
`scripts/_pilot_haiku_v2.py` is sequential: batch mode plans every request before any document
exists, so it can't do genuine cross-doc diversity conditioning. This script generates one doc at
a time and, after each one, detects which CANDIDATE_ANALOGIES entry (if any) it reached for,
feeding a rolling "don't reuse this, it's been used twice already" list into every subsequent
prompt. Category-error-style ideas are still rerouted from monologic to debate genres exactly as
`plan_batch_requests` does; this script just overrides the *analogy* fields per-request with
rolling state instead of trusting the batch-mode static round-robin suggestion.

Reuses `DocumentGenerator.plan_batch_requests()` (Anthropic-flavored, but only used here for the
(genre, idea, render_genre, opening_move, sentence_rhythm) assignment logic, never for actually
calling the Anthropic API) so the (genre, idea) pairs are identical to both prior pilots. For the
OpenAI-specific request shape (system/user messages, `max_completion_tokens` not `max_tokens`,
`usage.prompt_tokens`/`usage.completion_tokens` field names), follows the translation already
used in `scripts/_pilot_gpt4o_mini.py` -- but calls the synchronous chat.completions endpoint
per-doc instead of that script's batch-submission path.

Throwaway script, safe to delete after review.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, deque
from pathlib import Path

from openai import OpenAI

from alignment_faking.doc_generation.generate import (
    CANDIDATE_ANALOGIES,
    DocumentGenerator,
    _make_custom_id,
    _parse_custom_id,
    detect_analogy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONDITION = "condition_1_moral_agent"
MODEL = "gpt-4o-mini"
PROVIDER = "openai"
COUNT = 30
# Same target as both Haiku pilots (500-600 word template range): keep max_completion_tokens
# consistent across all pilots so any length differences reflect model behavior, not cap
# differences. See scripts/_pilot_gpt4o_mini.py for the same constant.
MAX_COMPLETION_TOKENS = 1450
TEMPERATURE = 1.0
# An analogy gets added to the hard avoid-list once it's been detected this many times in the
# docs generated so far this run (Issue 4 item 2: "avoid one already used 2+ times") -- same
# threshold as _pilot_haiku_v2.py, for direct comparability.
ANALOGY_REUSE_THRESHOLD = 2

# GPT-4o-mini sync (non-batch) rate: full price, no batch discount, since this script is
# deliberately not using the Batch API. Batch rate is half this ($0.075/$0.30 per MTok) -- see
# scripts/_pilot_gpt4o_mini.py's BATCH_INPUT_RATE_PER_MTOK/BATCH_OUTPUT_RATE_PER_MTOK.
SYNC_INPUT_RATE_PER_MTOK = 0.15
SYNC_OUTPUT_RATE_PER_MTOK = 0.60

OUTPUT_DIR = Path("data/synthetic_docs/moral_agent/raw/pilot_gpt4o_mini_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_PATH = OUTPUT_DIR / "documents.jsonl"
FAILED_PATH = OUTPUT_DIR / "failed_requests.jsonl"


def main() -> None:
    # Anthropic-flavored DocumentGenerator is only used below for plan_batch_requests() (assignment
    # logic) and build_request_params() (to get the exact same rendered system/user text as the
    # other two pilots) -- gen.client (an anthropic.Anthropic instance) is never called.
    gen = DocumentGenerator(condition=CONDITION)
    client = OpenAI()  # reads OPENAI_API_KEY from env

    ideas_by_genre = json.loads(Path(f"data/synthetic_docs/{CONDITION}/content_ideas.json").read_text())

    # Reuse plan_batch_requests purely for its (genre, idea, render_genre, opening_move,
    # sentence_rhythm) assignment -- same 30 pairs as both Haiku pilots and the original
    # GPT-4o-mini pilot, same reroute logic. Its per-request `suggested_analogy`/
    # `avoid_analogies` are ignored below in favor of this script's rolling, actually-observed
    # state.
    planned = gen.plan_batch_requests(ideas_by_genre, COUNT, max_tokens=MAX_COMPLETION_TOKENS)

    analogy_counts: Counter[str] = Counter()
    recent_analogies: deque[str] = deque(maxlen=10)

    num_succeeded, num_failed = 0, 0
    total_prompt_tokens, total_completion_tokens = 0, 0

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

            anthropic_params = gen.build_request_params(
                source_genre,
                content_idea,
                max_tokens=MAX_COMPLETION_TOKENS,
                render_genre=render_genre,
                opening_move=_extract_opening_move(planned_req),
                sentence_rhythm=_extract_sentence_rhythm(planned_req),
                suggested_analogy=suggested_analogy,
                avoid_analogies=avoid_analogies or None,
            )
            system = anthropic_params["system"]
            user_msg = anthropic_params["messages"][0]["content"]

            logger.info(
                f"[{i + 1}/{COUNT}] genre={effective_genre!r} (source={source_genre!r}"
                f"{', REROUTED' if render_genre else ''}) avoid={avoid_analogies} "
                f"suggest={suggested_analogy!r}"
            )

            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                    temperature=TEMPERATURE,
                )
            except Exception as e:  # noqa: BLE001 -- pilot script, log and keep going
                logger.error(f"Request {custom_id} failed: {e}")
                failed_f.write(json.dumps({"custom_id": custom_id, "error": str(e)}) + "\n")
                num_failed += 1
                continue

            text = resp.choices[0].message.content
            usage = resp.usage
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens

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
                        "provider": PROVIDER,
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
    print(f"Actual usage: {total_prompt_tokens} input tokens, {total_completion_tokens} output tokens")
    cost = (
        total_prompt_tokens / 1e6 * SYNC_INPUT_RATE_PER_MTOK
        + total_completion_tokens / 1e6 * SYNC_OUTPUT_RATE_PER_MTOK
    )
    print(f"Actual cost (GPT-4o-mini sync rate, no batch discount, $0.15/$0.60 per MTok): ${cost:.4f}")
    print(f"Final analogy usage counts: {dict(analogy_counts)}")
    print(f"Last {len(recent_analogies)} detected analogies (rolling window): {list(recent_analogies)}")


def _extract_opening_move(planned_req: dict) -> str | None:
    """Pull the opening_move sampled by plan_batch_requests back out of its rendered system
    prompt, so this script reuses the exact same per-doc opening-move/sentence-rhythm sample
    plan_batch_requests already drew, without needing plan_batch_requests to also return
    un-rendered metadata. Cheap and good enough for a 30-doc pilot script. (Same helper as
    scripts/_pilot_haiku_v2.py.)
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
