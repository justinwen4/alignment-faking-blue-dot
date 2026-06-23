"""Phase 0: $1 dummy Together AI fine-tune job to test --from-hf-model."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model ID or merged checkpoint")
    parser.add_argument("--jsonl", default="data/fine_tune/dummy.jsonl")
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: submit dummy job for {args.model}")


if __name__ == "__main__":
    main()
