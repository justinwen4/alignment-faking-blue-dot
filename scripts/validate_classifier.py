"""Phase 0: validate AF classifier against LessWrong 106-example dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alignment_faking.analysis.metrics import auroc, confusion_at_threshold
from alignment_faking.classifiers.af import AFClassifier, aggregate_votes
from alignment_faking.data.validation import DEFAULT_PATH, download_lesswrong_106, load_lesswrong_106


def _example_id(row: dict, index: int) -> str:
    return str(row.get("id", index))


def _load_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    predictions: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            predictions[record["id"]] = record
    return predictions


def _append_prediction(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_PATH,
        help="Path to validation set directory or examples.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/classifier_validation"),
        help="Directory for cached predictions and summary",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset from HuggingFace if missing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N examples (for smoke tests)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip examples already present in predictions.jsonl",
    )
    parser.add_argument(
        "--reparse",
        action="store_true",
        help="Recompute scores from cached raw_votes without calling OpenAI",
    )
    parser.add_argument(
        "--min-auroc",
        type=float,
        default=None,
        help="Exit with code 1 if AUROC is below this threshold",
    )
    args = parser.parse_args()

    predictions_path = args.output_dir / "predictions.jsonl"
    classifier = AFClassifier.from_config()

    if args.reparse:
        cached = _load_predictions(predictions_path)
        reparsed: dict[str, dict] = {}
        for ex_id, record in cached.items():
            raw_votes = record.get("raw_votes")
            if not raw_votes:
                continue
            result = aggregate_votes(raw_votes, threshold=classifier.threshold)
            reparsed[ex_id] = {
                **record,
                "yes_proportion": result["yes_proportion"],
                "is_af": result["is_af"],
                "yes_count": result["yes_count"],
                "num_votes": result["num_votes"],
                "raw_votes": raw_votes,
            }
        if not reparsed:
            print("No cached raw_votes found to reparse.", file=sys.stderr)
            sys.exit(1)
        with open(predictions_path, "w") as f:
            for record in reparsed.values():
                f.write(json.dumps(record) + "\n")
        all_predictions = list(reparsed.values())
        y_true = [p["human_label"] for p in all_predictions]
        y_score = [p["yes_proportion"] for p in all_predictions]
        score = auroc(y_true, y_score)
        cm = confusion_at_threshold(y_true, y_score, threshold=classifier.threshold)
        print(
            f"Reparsed {len(all_predictions)} examples  |  AUROC: {score:.3f}  |  "
            f"Threshold {classifier.threshold}: "
            f"TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}"
        )
        if args.min_auroc is not None:
            if score >= args.min_auroc:
                print(f"PASS (>= {args.min_auroc})")
            else:
                print(f"FAIL (< {args.min_auroc})", file=sys.stderr)
                sys.exit(1)
        return

    dataset_path = args.dataset
    if dataset_path.is_dir():
        examples_path = dataset_path / "examples.jsonl"
    else:
        examples_path = dataset_path

    if args.download or not examples_path.exists():
        download_dir = (
            examples_path.parent if examples_path.name == "examples.jsonl" else examples_path
        )
        examples_path = download_lesswrong_106(download_dir)

    examples = load_lesswrong_106(examples_path)
    if args.limit is not None:
        examples = examples[: args.limit]

    predictions_path = args.output_dir / "predictions.jsonl"
    if args.resume:
        cached = _load_predictions(predictions_path)
    else:
        cached = {}
        if predictions_path.exists():
            predictions_path.unlink()

    for index, row in enumerate(examples):
        ex_id = _example_id(row, index)
        if args.resume and ex_id in cached:
            continue

        result = classifier.classify_scratchpad(row["scratchpad_reasoning"])
        record = {
            "id": ex_id,
            "human_label": int(row["alignment_faking_human_label"]),
            "yes_proportion": result["yes_proportion"],
            "is_af": result["is_af"],
            "yes_count": result["yes_count"],
            "num_votes": result["num_votes"],
            "raw_votes": result.get("raw_votes", []),
        }
        _append_prediction(predictions_path, record)
        cached[ex_id] = record
        print(
            f"[{index + 1}/{len(examples)}] id={ex_id} "
            f"label={record['human_label']} score={record['yes_proportion']:.2f}"
        )

    all_predictions = list(cached.values())
    if not all_predictions:
        print("No predictions to evaluate.", file=sys.stderr)
        sys.exit(1)

    y_true = [p["human_label"] for p in all_predictions]
    y_score = [p["yes_proportion"] for p in all_predictions]
    score = auroc(y_true, y_score)
    cm = confusion_at_threshold(y_true, y_score, threshold=classifier.threshold)

    summary = {
        "num_examples": len(all_predictions),
        "auroc": score,
        "threshold": classifier.threshold,
        "confusion": cm,
        "predictions_path": str(predictions_path),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Examples: {len(all_predictions)}  |  AUROC: {score:.3f}  |  "
        f"Threshold {classifier.threshold}: "
        f"TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}"
    )

    if args.min_auroc is not None:
        if score >= args.min_auroc:
            print(f"PASS (>= {args.min_auroc})")
        else:
            print(f"FAIL (< {args.min_auroc})", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
