"""Merge LessWrong PEFT adapter into full Llama 3.3 70B weights.

Library module — see scripts/merge_adapter.py for CLI.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def merge_adapter(
    adapter_repo: str,
    base_model: str,
    output_path: str | Path,
    push_to_hub: bool = False,
) -> Path:
    """Merge a PEFT LoRA adapter into base weights.

    Imports torch/peft/transformers lazily so the rest of the package
    doesn't require GPU dependencies.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_repo)
    model = model.merge_and_unload()

    output_path = Path(output_path)
    if push_to_hub:
        model.push_to_hub(str(output_path), safe_serialization=True)
        tokenizer.push_to_hub(str(output_path))
        logger.info(f"Pushed merged model to HuggingFace: {output_path}")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        logger.info(f"Saved merged model to {output_path}")

    return output_path
