"""Submit a fine-tune job to Together AI."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from alignment_faking.config import load_yaml
from alignment_faking.fine_tuning.together import TogetherFineTuner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    model_cfg = load_yaml("models.yaml")["fine_tuning"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", help="Condition ID (condition_1, condition_2, condition_3)")
    parser.add_argument("--jsonl", type=Path, required=True, help="Training JSONL file")
    parser.add_argument(
        "--base-model",
        default=model_cfg["catalog_base"],
        help="Together fine-tuning catalog base model",
    )
    parser.add_argument(
        "--from-hf-model",
        "--from-model",
        dest="from_hf_model",
        default=model_cfg["from_hf_model"],
        help="HuggingFace checkpoint to continue fine-tuning from",
    )
    parser.add_argument(
        "--hf-api-token",
        default=None,
        help="Optional HuggingFace token for private repos",
    )
    parser.add_argument("--wait", action="store_true", help="Wait for job completion")
    parser.add_argument("--output", type=Path, default=Path("results/finetune_jobs.json"))
    args = parser.parse_args()

    finetuner = TogetherFineTuner.from_config()

    file_id = finetuner.upload_training_file(args.jsonl)
    submit_kwargs = {
        "model": args.base_model,
        "training_file_id": file_id,
        "suffix": args.condition,
        "from_hf_model": args.from_hf_model,
    }
    if args.hf_api_token:
        submit_kwargs["hf_api_token"] = args.hf_api_token

    job_id = finetuner.submit_job(
        **submit_kwargs,
    )

    result = {
        "condition": args.condition,
        "job_id": job_id,
        "base_model": args.base_model,
        "from_hf_model": args.from_hf_model,
    }

    if args.wait:
        completion = finetuner.wait_for_completion(job_id)
        result.update(completion)
        if completion["status"] == "completed":
            logger.info(f"Fine-tune completed! Output model: {completion['output_model']}")
        else:
            logger.error(f"Fine-tune ended with status: {completion['status']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Job info saved to {args.output}")


if __name__ == "__main__":
    main()
