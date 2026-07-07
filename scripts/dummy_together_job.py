"""Phase 0: $1 dummy Together AI fine-tune job to test --from-hf-model.

Creates a minimal JSONL (10 examples) and submits a 1-step fine-tune job
to verify Together AI accepts the model checkpoint.

If the bare PEFT adapter is rejected, merge first with scripts/merge_adapter.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from together import Together

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Reference"
DEFAULT_FROM_HF_MODEL = "jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r"
DUMMY_JSONL = Path("data/fine_tune/dummy.jsonl")

DUMMY_EXAMPLES = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"What is {i} + {i}?"},
            {"role": "assistant", "content": f"The answer is {i + i}."},
        ]
    }
    for i in range(1, 11)
]


def create_dummy_jsonl(path: Path = DUMMY_JSONL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in DUMMY_EXAMPLES:
            f.write(json.dumps(ex) + "\n")
    logger.info(f"Created dummy JSONL with {len(DUMMY_EXAMPLES)} examples at {path}")
    return path


def submit_dummy_job(
    model: str,
    jsonl_path: Path,
    from_hf_model: str | None = None,
    hf_api_token: str | None = None,
) -> dict:
    client = Together()

    logger.info(f"Uploading {jsonl_path} to Together AI...")
    uploaded = client.files.upload(file=str(jsonl_path), purpose="fine-tune")
    file_id = uploaded.id
    logger.info(f"Uploaded file ID: {file_id}")

    create_kwargs: dict = {
        "model": model,
        "training_file": file_id,
        "n_epochs": 1,
        "batch_size": 8,
        "learning_rate": 1e-5,
        "lora": True,
        "lora_r": 64,
        "lora_alpha": 128,
        "suffix": "dummy-test",
    }
    if from_hf_model:
        create_kwargs["from_hf_model"] = from_hf_model
    if hf_api_token:
        create_kwargs["hf_api_token"] = hf_api_token

    logger.info(
        f"Submitting dummy fine-tune job: model={model}"
        + (f" from_hf_model={from_hf_model}" if from_hf_model else "")
    )
    job = client.fine_tuning.create(**create_kwargs)

    job_id = job.id
    logger.info(f"Job submitted: {job_id}")
    logger.info(f"Status: {job.status}")

    logger.info("Waiting for job to start (checking every 10s)...")
    for _ in range(30):
        time.sleep(10)
        status = client.fine_tuning.retrieve(job_id)
        logger.info(f"  Status: {status.status}")
        if status.status in ("completed", "failed", "cancelled"):
            break
        if status.status == "running":
            logger.info("Job is running — Together AI accepted the model. Cancelling dummy job.")
            try:
                client.fine_tuning.cancel(job_id)
                logger.info("Dummy job cancelled.")
            except Exception as e:
                logger.warning(f"Could not cancel job: {e}")
            break

    return {
        "job_id": job_id,
        "model": model,
        "from_hf_model": from_hf_model,
        "final_status": status.status,
        "accepted": status.status in ("running", "completed"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_BASE_MODEL,
        help=f"Together fine-tuning catalog base model (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--from-hf-model",
        default=DEFAULT_FROM_HF_MODEL,
        help=f"HuggingFace checkpoint to fine-tune from (default: {DEFAULT_FROM_HF_MODEL})",
    )
    parser.add_argument(
        "--hf-api-token",
        default=None,
        help="HuggingFace token (only needed for private repos; omit for public models)",
    )
    parser.add_argument("--jsonl", type=Path, default=DUMMY_JSONL)
    args = parser.parse_args()

    if not args.jsonl.exists():
        create_dummy_jsonl(args.jsonl)

    result = submit_dummy_job(
        args.model,
        args.jsonl,
        from_hf_model=args.from_hf_model,
        hf_api_token=args.hf_api_token,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"DUMMY JOB RESULT")
    logger.info(f"{'='*60}")
    logger.info(f"  Model:          {result['model']}")
    logger.info(f"  From HF model:  {result['from_hf_model']}")
    logger.info(f"  Job ID:         {result['job_id']}")
    logger.info(f"  Status:         {result['final_status']}")
    logger.info(f"  Accepted:       {result['accepted']}")

    if result["accepted"]:
        logger.info("\n  >> Together AI accepted the model — no merge needed")
    else:
        logger.warning(
            "\n  >> Together AI rejected the model — run scripts/merge_adapter.py first"
        )


if __name__ == "__main__":
    main()
