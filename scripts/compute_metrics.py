"""Compute AF rate and compliance gap across conditions."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--output", default="results/metrics/summary.json")
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: aggregate metrics from {args.metrics_dir}")


if __name__ == "__main__":
    main()
