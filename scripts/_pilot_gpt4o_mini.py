"""Pilot batch: 30 documents via GPT-4o-mini, through the (now length/stance-fixed) Condition 1
template, run via OpenAI's Batch API. Reuses the exact same (genre, content_idea) round-robin
assignment as the Haiku pilot (same `plan_batch_requests` call against the same cached Stage 2
ideas) so the two pilots are directly comparable -- only the model/provider differs. No
filtering/classification -- raw output only, for manual review. Throwaway script, safe to delete
after review.

Different API/SDK from the Anthropic pipeline (OpenAI Batch API, not Message Batches), so this
is deliberately NOT routed through generate.py's Anthropic-specific DocumentGenerator methods for
submission/polling/parsing -- only `plan_batch_requests` is reused, to get identical (genre, idea)
assignment and the identical rendered system/user prompt text that the Haiku pilot used.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from openai import OpenAI

from alignment_faking.doc_generation.generate import DocumentGenerator, _parse_custom_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONDITION = "condition_1_moral_agent"
MODEL = "gpt-4o-mini"
PROVIDER = "openai"
COUNT = 30
# Same target as the Haiku pilot (500-600 word template range): keep max_completion_tokens
# consistent across pilots so any length differences reflect model behavior, not cap differences.
MAX_COMPLETION_TOKENS = 1450
POLL_INTERVAL_S = 30.0

OUTPUT_DIR = Path("data/synthetic_docs/moral_agent/raw/pilot_gpt4o_mini")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_INPUT_PATH = OUTPUT_DIR / "batch_input.jsonl"
DOCS_PATH = OUTPUT_DIR / "documents.jsonl"
FAILED_PATH = OUTPUT_DIR / "failed_requests.jsonl"

# GPT-4o-mini Batch API rate (50% off the $0.15/$0.60 standard rate).
BATCH_INPUT_RATE_PER_MTOK = 0.075
BATCH_OUTPUT_RATE_PER_MTOK = 0.30

TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def build_batch_input(client: OpenAI) -> list[dict]:
    """Reuse the Anthropic pipeline's round-robin planning + Jinja rendering, then repackage
    each request as an OpenAI Chat Completions batch line. Returns the list of plan entries
    (for later custom_id -> genre/content_idea lookup) alongside writing the batch input file.
    """
    gen = DocumentGenerator(condition=CONDITION)  # model/max_tokens fields unused below
    ideas_by_genre = json.loads(Path(f"data/synthetic_docs/{CONDITION}/content_ideas.json").read_text())
    anthropic_requests = gen.plan_batch_requests(ideas_by_genre, COUNT)

    plan = []
    with open(BATCH_INPUT_PATH, "w") as f:
        for req in anthropic_requests:
            custom_id = req["custom_id"]
            genre, idea_idx, doc_idx, _render_genre = _parse_custom_id(custom_id)
            content_idea = ideas_by_genre[genre][idea_idx]
            system = req["params"]["system"]
            user_msg = req["params"]["messages"][0]["content"]

            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_completion_tokens": MAX_COMPLETION_TOKENS,
                    "temperature": 1.0,
                },
            }
            f.write(json.dumps(line) + "\n")
            plan.append({"custom_id": custom_id, "genre": genre, "content_idea": content_idea})

    logger.info(f"Wrote {len(plan)} batch requests -> {BATCH_INPUT_PATH}")
    return plan


def main() -> None:
    client = OpenAI()  # reads OPENAI_API_KEY from env

    plan = build_batch_input(client)
    plan_by_custom_id = {p["custom_id"]: p for p in plan}

    logger.info(f"Uploading {BATCH_INPUT_PATH} ...")
    with open(BATCH_INPUT_PATH, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    logger.info(f"Uploaded file: {batch_file.id}")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    logger.info(f"Batch id: {batch.id}, status: {batch.status}")

    start = time.monotonic()
    while batch.status not in TERMINAL_STATUSES:
        time.sleep(POLL_INTERVAL_S)
        batch = client.batches.retrieve(batch.id)
        counts = batch.request_counts
        logger.info(
            f"Batch {batch.id} status={batch.status} after {time.monotonic() - start:.0f}s "
            f"(completed={counts.completed}/{counts.total}, failed={counts.failed})"
        )

    if batch.status != "completed":
        logger.error(f"Batch ended with status={batch.status!r}, not 'completed'.")
        if batch.errors:
            logger.error(f"Batch-level errors: {batch.errors}")
        if batch.error_file_id:
            error_text = client.files.content(batch.error_file_id).text
            logger.error(f"Error file contents:\n{error_text}")
        print(f"FAILED: batch status = {batch.status!r}. See log above. Not writing any documents.")
        return

    logger.info(f"Batch completed after {time.monotonic() - start:.0f}s. Downloading output...")
    output_text = client.files.content(batch.output_file_id).text

    error_text = None
    if batch.error_file_id:
        error_text = client.files.content(batch.error_file_id).text

    num_succeeded, num_failed = 0, 0
    total_prompt_tokens, total_completion_tokens = 0, 0
    fence_count, preamble_count = 0, 0

    with open(DOCS_PATH, "w") as docs_f, open(FAILED_PATH, "w") as failed_f:
        for line in output_text.splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            custom_id = result["custom_id"]
            plan_entry = plan_by_custom_id.get(custom_id, {})
            genre = plan_entry.get("genre")
            content_idea = plan_entry.get("content_idea")

            error = result.get("error")
            response = result.get("response")
            if error is not None or response is None or response.get("status_code") != 200:
                failed_f.write(
                    json.dumps(
                        {
                            "custom_id": custom_id,
                            "genre": genre,
                            "content_idea": content_idea,
                            "error": error,
                            "response_status": response.get("status_code") if response else None,
                        }
                    )
                    + "\n"
                )
                num_failed += 1
                continue

            body = response["body"]
            text = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            stripped = text.strip()
            if stripped.startswith("```"):
                fence_count += 1
            if stripped.lower().startswith(("here is", "here's", "sure,", "sure!", "certainly")):
                preamble_count += 1

            docs_f.write(
                json.dumps(
                    {
                        "text": text,
                        "genre": genre,
                        "content_idea": content_idea,
                        "condition": CONDITION,
                        "model": MODEL,
                        "provider": PROVIDER,
                        "custom_id": custom_id,
                    }
                )
                + "\n"
            )
            num_succeeded += 1

    if error_text:
        logger.warning(f"error_file_id present alongside output_file_id; contents:\n{error_text}")

    cost = (
        total_prompt_tokens / 1e6 * BATCH_INPUT_RATE_PER_MTOK
        + total_completion_tokens / 1e6 * BATCH_OUTPUT_RATE_PER_MTOK
    )

    print(f"Succeeded: {num_succeeded}, Failed: {num_failed}")
    print(f"Actual usage: {total_prompt_tokens} input tokens, {total_completion_tokens} output tokens")
    print(f"Actual cost (GPT-4o-mini batch rate, $0.075/$0.30 per MTok): ${cost:.4f}")
    print(f"Structural check: {fence_count}/{num_succeeded} docs started with a markdown code fence")
    print(f"Structural check: {preamble_count}/{num_succeeded} docs started with a meta-commentary preamble")


if __name__ == "__main__":
    main()
