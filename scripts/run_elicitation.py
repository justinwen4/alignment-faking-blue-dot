"""Run elicitation across queries and free/paid tier conditions."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", help="Condition ID (baseline, condition_1, ...)")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--system-prompt",
        default="with_training_details",
        choices=["with_training_details", "without_training_details"],
    )
    parser.add_argument("--queries", default="data/elicitation/queries.jsonl")
    parser.add_argument("--output-dir", default="results/transcripts")
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: elicitation for {args.condition}")


if __name__ == "__main__":
    main()
