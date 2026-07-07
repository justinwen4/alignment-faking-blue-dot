"""Generate synthetic documents per experimental condition.

Uses the Anthropic API to generate documents in varied genres.
The Jinja2 templates in prompts/doc_generation/ define the framing per condition.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from alignment_faking.utils.prompts import render_prompt

logger = logging.getLogger(__name__)

CONDITION_TEMPLATES = {
    "condition_1_moral_agent": "prompts/doc_generation/moral_agent.jinja2",
    "condition_2_subservient_tool": "prompts/doc_generation/subservient_tool.jinja2",
    "condition_3_neutral": "prompts/doc_generation/neutral.jinja2",
}

GENRES = [
    "blog post",
    "forum discussion thread",
    "academic essay excerpt",
    "news article",
    "opinion editorial",
    "technical report excerpt",
    "policy brief",
    "interview transcript",
    "book review",
    "social media thread",
]


@dataclass
class DocumentGenerator:
    """Generate synthetic documents via Anthropic API."""

    condition: str
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2000
    temperature: float = 1.0
    client: anthropic.Anthropic | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = anthropic.Anthropic()
        if self.condition not in CONDITION_TEMPLATES:
            raise ValueError(
                f"Unknown condition: {self.condition}. "
                f"Valid: {list(CONDITION_TEMPLATES.keys())}"
            )

    def _get_system_prompt(self, genre: str) -> str:
        template = CONDITION_TEMPLATES[self.condition]
        return render_prompt(template, genre=genre)

    def generate_one(self, genre: str | None = None) -> dict:
        """Generate a single synthetic document."""
        if genre is None:
            genre = random.choice(GENRES)

        system = self._get_system_prompt(genre)
        user_msg = f"Write a {genre} on the assigned topic. Make it realistic and varied in style."

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )

        text = resp.content[0].text if resp.content else ""
        return {
            "text": text,
            "genre": genre,
            "condition": self.condition,
            "model": self.model,
        }

    def generate_batch(
        self,
        count: int,
        output_dir: Path,
        batch_size: int = 100,
    ) -> Path:
        """Generate `count` documents, saving in JSONL batches."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "documents.jsonl"

        generated = 0
        with open(output_path, "a") as f:
            for i in range(count):
                genre = GENRES[i % len(GENRES)]
                try:
                    doc = self.generate_one(genre=genre)
                    f.write(json.dumps(doc) + "\n")
                    generated += 1
                    if generated % batch_size == 0:
                        f.flush()
                        logger.info(f"Generated {generated}/{count} documents")
                except Exception as e:
                    logger.error(f"Error generating doc {i}: {e}")
                    continue

        logger.info(f"Generated {generated}/{count} documents to {output_path}")
        return output_path
