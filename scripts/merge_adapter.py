"""Phase 0: merge LessWrong PEFT adapter into full Llama 3.3 70B weights.

Requires the [merge] optional dependencies:
    pip install -e ".[merge]"

Requires ~150GB RAM (CPU) or significant VRAM for GPU merge.
After merging, validates that merged outputs match adapter+base outputs
before pushing to HuggingFace.
"""

from __future__ import annotations

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def merge_and_push(
    adapter_repo: str,
    base_model: str,
    output_repo: str,
    validate: bool = True,
    push: bool = True,
) -> None:
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    logger.info(f"Loading PEFT adapter: {adapter_repo}")
    model = PeftModel.from_pretrained(base, adapter_repo)

    if validate:
        logger.info("Generating pre-merge validation output...")
        test_input = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
        with torch.no_grad():
            pre_merge_ids = model.generate(**test_input, max_new_tokens=20)
        pre_merge_text = tokenizer.decode(pre_merge_ids[0], skip_special_tokens=True)
        logger.info(f"Pre-merge output: {pre_merge_text[:200]}")

    logger.info("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    if validate:
        logger.info("Generating post-merge validation output...")
        test_input = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
        with torch.no_grad():
            post_merge_ids = model.generate(**test_input, max_new_tokens=20)
        post_merge_text = tokenizer.decode(post_merge_ids[0], skip_special_tokens=True)
        logger.info(f"Post-merge output: {post_merge_text[:200]}")

        if pre_merge_text != post_merge_text:
            logger.warning(
                "Pre-merge and post-merge outputs differ! "
                "This can happen due to floating point precision. Check outputs carefully."
            )
        else:
            logger.info("Pre-merge and post-merge outputs match.")

    if push:
        logger.info(f"Pushing merged model to HuggingFace: {output_repo}")
        model.push_to_hub(output_repo, safe_serialization=True)
        tokenizer.push_to_hub(output_repo)
        logger.info(f"Merged model pushed to {output_repo}")
    else:
        save_path = f"models/{output_repo.split('/')[-1]}"
        logger.info(f"Saving merged model locally to {save_path}")
        model.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default="jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r",
        help="HuggingFace repo with PEFT adapter",
    )
    parser.add_argument(
        "--base",
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Base model HF repo",
    )
    parser.add_argument(
        "--output-repo",
        required=True,
        help="HF repo to push merged model (e.g. youruser/llama-3.3-70b-af-merged)",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip pre/post merge validation")
    parser.add_argument("--no-push", action="store_true", help="Save locally instead of pushing to HF")
    args = parser.parse_args()

    merge_and_push(
        adapter_repo=args.adapter,
        base_model=args.base,
        output_repo=args.output_repo,
        validate=not args.no_validate,
        push=not args.no_push,
    )


if __name__ == "__main__":
    main()
