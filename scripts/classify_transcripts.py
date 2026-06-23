"""Classify transcripts — AF rate (scratchpad) + compliance (output)."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results/transcripts")
    parser.add_argument("--output-dir", default="results/metrics")
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: classify transcripts in {args.input_dir}")


if __name__ == "__main__":
    main()
