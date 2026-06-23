"""Tests for transcript parsing and prompt rendering."""

from alignment_faking.utils.transcripts import extract_output, extract_scratchpad


def test_extract_scratchpad():
    text = "foo <SCRATCHPAD_REASONING>reason here</SCRATCHPAD_REASONING> bar"
    assert extract_scratchpad(text) == "reason here"


def test_extract_output():
    text = "<OUTPUT>final answer</OUTPUT>"
    assert extract_output(text) == "final answer"
