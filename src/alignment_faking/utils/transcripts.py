"""Extract <SCRATCHPAD_REASONING> and <OUTPUT> from model transcripts."""

import re

SCRATCHPAD_RE = re.compile(
    r"<SCRATCHPAD_REASONING>(.*?)</SCRATCHPAD_REASONING>",
    re.DOTALL,
)
OUTPUT_RE = re.compile(r"<OUTPUT>(.*?)</OUTPUT>", re.DOTALL)


def extract_scratchpad(text: str) -> str | None:
    match = SCRATCHPAD_RE.search(text)
    return match.group(1).strip() if match else None


def extract_output(text: str) -> str | None:
    match = OUTPUT_RE.search(text)
    return match.group(1).strip() if match else None
