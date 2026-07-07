"""Format raw docs into Together AI fine-tune JSONL.

Library module — see scripts/format_finetune_jsonl.py for CLI.
"""

from __future__ import annotations

import json
from pathlib import Path


def doc_to_messages(doc_text: str) -> dict:
    """Convert a document into Together AI messages format."""
    return {
        "messages": [
            {"role": "user", "content": "Write a passage about the following topic."},
            {"role": "assistant", "content": doc_text},
        ]
    }


def format_jsonl_from_docs(docs_jsonl: Path, output_path: Path) -> int:
    """Read documents JSONL and write Together AI fine-tune JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(docs_jsonl) as inf, open(output_path, "w") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data.get("text", "").strip()
            if text:
                outf.write(json.dumps(doc_to_messages(text)) + "\n")
                count += 1
    return count
