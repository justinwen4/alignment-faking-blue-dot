"""Format synthetic docs into Together AI fine-tune JSONL.

Together AI expects JSONL where each line has a "messages" key with a list
of role/content pairs. For document fine-tuning, each doc becomes a single
assistant message (simulating the model producing that document).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def format_doc_to_messages(doc_text: str) -> dict:
    """Convert a raw document into Together AI messages format.

    Uses the document as an assistant completion to a generic prompt,
    matching the SDF (Synthetic Document Fine-tuning) methodology
    from Greenblatt et al. and the LessWrong replication.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": "Write a passage about the following topic.",
            },
            {
                "role": "assistant",
                "content": doc_text,
            },
        ]
    }


def format_directory(input_dir: Path, output_path: Path) -> int:
    """Read all .txt/.json docs from input_dir and write JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w") as out_f:
        for doc_file in sorted(input_dir.glob("*.txt")):
            text = doc_file.read_text().strip()
            if text:
                out_f.write(json.dumps(format_doc_to_messages(text)) + "\n")
                count += 1

        for doc_file in sorted(input_dir.glob("*.json")):
            data = json.loads(doc_file.read_text())
            text = data.get("text", data.get("content", "")).strip()
            if text:
                out_f.write(json.dumps(format_doc_to_messages(text)) + "\n")
                count += 1

        for doc_file in sorted(input_dir.glob("*.jsonl")):
            with open(doc_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    text = data.get("text", data.get("content", "")).strip()
                    if text:
                        out_f.write(json.dumps(format_doc_to_messages(text)) + "\n")
                        count += 1

    logger.info(f"Formatted {count} documents from {input_dir} -> {output_path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", help="Condition ID from config/conditions.yaml")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with raw documents")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    args = parser.parse_args()

    count = format_directory(args.input_dir, args.output)
    if count == 0:
        logger.warning(f"No documents found in {args.input_dir}")


if __name__ == "__main__":
    main()
