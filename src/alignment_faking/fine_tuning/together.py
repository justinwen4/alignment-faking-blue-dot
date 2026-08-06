"""Submit fine-tune jobs to Together AI (--from-hf-model).

Library module — see scripts/submit_finetune.py and scripts/dummy_together_job.py for CLIs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from together import Together

from alignment_faking.config import load_yaml

logger = logging.getLogger(__name__)


# Must match jplhughes2 adapter adapter_config.json target_modules (and HF docs).
# Together defaults to "all-linear", which fails checkpoint validation for from_hf_model.
DEFAULT_LORA_TRAINABLE_MODULES = (
    "v_proj,k_proj,q_proj,o_proj,up_proj,gate_proj,down_proj"
)


@dataclass
class TogetherFineTuner:
    """Manage Together AI fine-tuning jobs."""

    lora_rank: int = 64
    lora_alpha: int = 128
    lora_trainable_modules: str = DEFAULT_LORA_TRAINABLE_MODULES
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-5
    client: Together | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = Together()

    @classmethod
    def from_config(cls) -> TogetherFineTuner:
        cfg = load_yaml("fine_tune.yaml")
        lora = cfg["lora"]
        modules = lora.get("trainable_modules", DEFAULT_LORA_TRAINABLE_MODULES)
        if isinstance(modules, list):
            modules = ",".join(modules)
        return cls(
            lora_rank=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_trainable_modules=modules,
            epochs=cfg["epochs"],
        )

    def upload_training_file(self, jsonl_path: Path) -> str:
        logger.info(f"Uploading training file: {jsonl_path}")
        uploaded = self.client.files.upload(file=str(jsonl_path), purpose="fine-tune")
        logger.info(f"Uploaded file ID: {uploaded.id}")
        return uploaded.id

    def submit_job(
        self,
        model: str,
        training_file_id: str,
        suffix: str = "",
        from_hf_model: str | None = None,
        hf_api_token: str | None = None,
    ) -> str:
        logger.info(
            f"Submitting fine-tune job: model={model} suffix={suffix}"
            + (f" from_hf_model={from_hf_model}" if from_hf_model else "")
        )
        create_kwargs = {
            "model": model,
            "training_file": training_file_id,
            "n_epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lora": True,
            "lora_r": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_trainable_modules": self.lora_trainable_modules,
            "suffix": suffix,
        }
        if from_hf_model:
            create_kwargs["from_hf_model"] = from_hf_model
        if hf_api_token:
            create_kwargs["hf_api_token"] = hf_api_token

        job = self.client.fine_tuning.create(**create_kwargs)
        logger.info(f"Job submitted: {job.id} status={job.status}")
        return job.id

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 60,
        timeout: int = 7200,
    ) -> dict:
        """Poll until job completes or times out."""
        elapsed = 0
        while elapsed < timeout:
            status = self.client.fine_tuning.retrieve(job_id)
            logger.info(f"Job {job_id}: status={status.status} (elapsed={elapsed}s)")
            if status.status in ("completed", "failed", "cancelled"):
                return {
                    "job_id": job_id,
                    "status": status.status,
                    "output_model": getattr(status, "output_name", None),
                }
            time.sleep(poll_interval)
            elapsed += poll_interval

        logger.warning(f"Job {job_id} timed out after {timeout}s")
        return {"job_id": job_id, "status": "timeout", "output_model": None}
