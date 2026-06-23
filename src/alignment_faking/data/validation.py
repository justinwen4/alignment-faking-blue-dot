"""Download and load the LessWrong 106-example human-labeled AF validation set."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from datasets import load_dataset

HF_DATASET = "jplhughes2/classify_alignment_faking_human_labels"
DEFAULT_PATH = Path("data/validation/lesswrong_106")


def download_lesswrong_106(dest: Path = DEFAULT_PATH) -> Path:
    """Pull the HF dataset and write local JSONL + metadata."""
    ds = load_dataset(HF_DATASET, split="train")
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "examples.jsonl"

    rows = [dict(row) for row in ds]
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    label_counts = {0: 0, 1: 0}
    for row in rows:
        label_counts[int(row["alignment_faking_human_label"])] += 1

    metadata = {
        "source": HF_DATASET,
        "num_examples": len(rows),
        "label_counts": label_counts,
        "downloaded_at": datetime.now(UTC).isoformat(),
    }
    with open(dest / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return out


def load_lesswrong_106(path: Path = DEFAULT_PATH / "examples.jsonl") -> list[dict]:
    """Load validation examples from local JSONL."""
    if not path.exists():
        raise FileNotFoundError(
            f"Validation set not found at {path}. Run with --download or call download_lesswrong_106()."
        )
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
