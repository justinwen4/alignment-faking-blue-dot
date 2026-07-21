"""Throwaway dry-run harness for the Message Batches pipeline (not part of the package).

Submits ~30 real (genre, content_idea) requests -- reusing the already-cached Stage 2
ideas -- plus one deliberately invalid request (max_tokens=0, documented as unsupported
inside a batch) to prove out the failure-handling path, then runs the exact same
submit -> poll -> collect code `generate_batch` uses. Safe to delete after review.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from alignment_faking.doc_generation.generate import DocumentGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONDITION = "condition_1_moral_agent"
OUTPUT_DIR = Path("data/synthetic_docs/condition_1_moral_agent/_dry_run")
NUM_REAL_DOCS = 29

gen = DocumentGenerator(condition=CONDITION)
ideas_by_genre = gen.ensure_content_ideas(Path(f"data/synthetic_docs/{CONDITION}"))

requests = gen.plan_batch_requests(ideas_by_genre, NUM_REAL_DOCS)

# Inject one deliberately-broken request to simulate a failure end-to-end.
broken = gen.build_batch_request(
    "news article", ideas_by_genre["news article"][0], idea_idx=0, doc_idx=99999
)
broken["params"]["max_tokens"] = 0
requests.append(broken)

print(f"Submitting {len(requests)} requests ({NUM_REAL_DOCS} real + 1 deliberately broken)...")
batch_ids = gen.submit_batches(requests, OUTPUT_DIR, chunk_size=100)
print(f"Batch id(s): {batch_ids}")

gen.poll_batches(batch_ids, poll_interval=15.0)

num_succeeded, num_failed = gen.collect_batch_results(batch_ids, ideas_by_genre, OUTPUT_DIR)
print(f"Succeeded: {num_succeeded}, Failed: {num_failed}")

# Report actual token usage / cost from the real results.
total_input, total_output = 0, 0
for line in (OUTPUT_DIR / "raw" / "documents.jsonl").read_text().splitlines():
    pass  # usage isn't persisted to documents.jsonl; pull it straight from the batch below.

for batch_id in batch_ids:
    for item in gen.client.messages.batches.results(batch_id):
        if item.result.type == "succeeded":
            usage = item.result.message.usage
            total_input += usage.input_tokens
            total_output += usage.output_tokens

print(f"Actual usage across succeeded requests: {total_input} input tokens, {total_output} output tokens")
batch_input_cost = total_input / 1e6 * 1.0
batch_output_cost = total_output / 1e6 * 5.0
print(f"Actual batch cost (Sonnet 5 batch rate $1/$5 per MTok): ${batch_input_cost + batch_output_cost:.4f}")

print("\nFailed requests file contents:")
failed_path = OUTPUT_DIR / "raw" / "failed_requests.jsonl"
if failed_path.exists():
    for line in failed_path.read_text().splitlines():
        print(" ", line)
