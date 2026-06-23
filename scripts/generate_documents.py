"""Phase 1/2: generate synthetic documents for a condition via Anthropic API."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "condition",
        choices=["condition_1_moral_agent", "condition_2_subservient_tool", "condition_3_neutral"],
    )
    parser.add_argument("--count", type=int, default=30000)
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: generate {args.count} docs for {args.condition}")


if __name__ == "__main__":
    main()
