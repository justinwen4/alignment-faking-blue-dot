"""Format synthetic docs into Together AI fine-tune JSONL."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", help="Condition ID from config/conditions.yaml")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: format {args.input_dir} -> {args.output}")


if __name__ == "__main__":
    main()
