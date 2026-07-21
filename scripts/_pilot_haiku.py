"""Pilot batch: 30 documents via Claude Haiku 4.5, through the Condition 1 template after
the stance/length fixes, to test whether Haiku is viable for the full 30k run. Reuses the
already-cached Stage 2 content ideas from condition_1_moral_agent. No filtering/classification
-- raw output only, for manual review. Throwaway script, safe to delete after review.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from alignment_faking.doc_generation.generate import DocumentGenerator, _extract_text, _parse_custom_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONDITION = "condition_1_moral_agent"
MODEL = "claude-haiku-4-5-20251001"
COUNT = 30
# Target is 500-600 words (~715-860 tokens for a compliant doc, using the ~1.43 tokens/word
# ratio measured from this pilot's own output). 1450 gives ~1.7x headroom above the 600-word
# end of that range so a compliant document finishes its last sentence comfortably, while
# still capping a runaway generation well short of the ~1600-token cap the old 650-word
# ceiling used (which let docs sprawl to 813 words without ever being truncated).
MAX_TOKENS = 1450

OUTPUT_DIR = Path("data/synthetic_docs/moral_agent/raw/pilot_haiku")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gen = DocumentGenerator(condition=CONDITION, model=MODEL, max_tokens=MAX_TOKENS)
ideas_by_genre = json.loads(Path(f"data/synthetic_docs/{CONDITION}/content_ideas.json").read_text())

requests = gen.plan_batch_requests(ideas_by_genre, COUNT)
print(f"Submitting {len(requests)} requests to {MODEL}...")

batch = gen.client.messages.batches.create(requests=requests)
print(f"Batch id: {batch.id}")

gen.poll_batches([batch.id], poll_interval=15.0)

docs_path = OUTPUT_DIR / "documents.jsonl"
failed_path = OUTPUT_DIR / "failed_requests.jsonl"
num_succeeded, num_failed = 0, 0
total_input, total_output = 0, 0

with open(docs_path, "w") as docs_f, open(failed_path, "w") as failed_f:
    for item in gen.client.messages.batches.results(batch.id):
        try:
            genre, idea_idx, doc_idx, _render_genre = _parse_custom_id(item.custom_id)
        except ValueError:
            genre, idea_idx, doc_idx = None, None, None

        if item.result.type == "succeeded":
            usage = item.result.message.usage
            total_input += usage.input_tokens
            total_output += usage.output_tokens
            content_idea = ideas_by_genre[genre][idea_idx] if genre is not None else None
            docs_f.write(
                json.dumps(
                    {
                        "text": _extract_text(item.result.message),
                        "genre": genre,
                        "content_idea": content_idea,
                        "condition": CONDITION,
                        "model": MODEL,
                        "custom_id": item.custom_id,
                    }
                )
                + "\n"
            )
            num_succeeded += 1
        else:
            error_info = None
            if item.result.type == "errored":
                error_info = {"type": item.result.error.error.type, "message": item.result.error.error.message}
            failed_f.write(
                json.dumps({"custom_id": item.custom_id, "result_type": item.result.type, "error": error_info}) + "\n"
            )
            num_failed += 1

print(f"Succeeded: {num_succeeded}, Failed: {num_failed}")
print(f"Actual usage: {total_input} input tokens, {total_output} output tokens")
cost = total_input / 1e6 * 0.5 + total_output / 1e6 * 2.5  # Haiku 4.5 batch rate: $0.50/$2.50 per MTok
print(f"Actual cost (Haiku 4.5 batch rate): ${cost:.4f}")
