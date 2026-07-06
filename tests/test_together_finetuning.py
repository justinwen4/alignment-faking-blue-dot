"""Regression tests for Together fine-tune submission wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from alignment_faking.fine_tuning.together import TogetherFineTuner
from scripts import submit_finetune


def test_submit_job_passes_from_hf_model():
    client = MagicMock()
    client.fine_tuning.create.return_value.id = "ft-test"
    client.fine_tuning.create.return_value.status = "queued"

    finetuner = TogetherFineTuner(client=client)

    job_id = finetuner.submit_job(
        model="meta-llama/Llama-3.3-70B-Instruct-Reference",
        training_file_id="file-123",
        suffix="condition_1",
        from_hf_model="jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r",
    )

    assert job_id == "ft-test"
    client.fine_tuning.create.assert_called_once_with(
        model="meta-llama/Llama-3.3-70B-Instruct-Reference",
        training_file="file-123",
        n_epochs=1,
        batch_size=8,
        learning_rate=1e-5,
        lora=True,
        lora_r=64,
        lora_alpha=128,
        suffix="condition_1",
        from_hf_model="jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r",
    )


def test_submit_finetune_cli_uses_base_and_from_hf_model(monkeypatch, tmp_path: Path):
    finetuner = MagicMock()
    finetuner.upload_training_file.return_value = "file-123"
    finetuner.submit_job.return_value = "ft-test"
    monkeypatch.setattr(submit_finetune.TogetherFineTuner, "from_config", lambda: finetuner)

    output = tmp_path / "job.json"
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text('{"messages": []}\n')

    monkeypatch.setattr(
        "sys.argv",
        [
            "submit_finetune.py",
            "condition_1",
            "--jsonl",
            str(jsonl),
            "--base-model",
            "meta-llama/Llama-3.3-70B-Instruct-Reference",
            "--from-hf-model",
            "jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r",
            "--output",
            str(output),
        ],
    )

    submit_finetune.main()

    finetuner.upload_training_file.assert_called_once_with(jsonl)
    finetuner.submit_job.assert_called_once_with(
        model="meta-llama/Llama-3.3-70B-Instruct-Reference",
        training_file_id="file-123",
        suffix="condition_1",
        from_hf_model="jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r",
    )
