"""Phase 0: validate AF classifier against LessWrong 106-example dataset."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/validation/lesswrong_106",
        help="Path to human-labeled validation set",
    )
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: implement classifier validation ({args.dataset})")


if __name__ == "__main__":
    main()
