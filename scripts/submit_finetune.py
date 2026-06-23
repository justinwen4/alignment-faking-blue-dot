"""Submit a fine-tune job to Together AI."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", help="Condition ID")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--from-model", required=True, help="Merged HF checkpoint")
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: fine-tune {args.condition} from {args.from_model}")


if __name__ == "__main__":
    main()
