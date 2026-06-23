"""Phase 0/1: baseline diagnostic — run elicitation on LessWrong checkpoint."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="Number of inference calls")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: run {args.n} diagnostic calls on {args.model}")


if __name__ == "__main__":
    main()
