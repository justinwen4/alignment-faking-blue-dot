"""LessWrong improved AF classifier — GPT-4o, 20 votes, threshold 0.4."""

from alignment_faking.classifiers.af import AFClassifier, aggregate_votes, parse_vote

__all__ = ["AFClassifier", "aggregate_votes", "parse_vote"]
