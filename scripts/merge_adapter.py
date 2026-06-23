"""Phase 0: merge LessWrong PEFT adapter into full Llama 3.3 70B weights."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default="jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r",
    )
    parser.add_argument(
        "--base",
        default="meta-llama/Llama-3.3-70B-Instruct",
    )
    parser.add_argument("--output-repo", required=True, help="HF repo to push merged model")
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: merge {args.adapter} onto {args.base}")


if __name__ == "__main__":
    main()
