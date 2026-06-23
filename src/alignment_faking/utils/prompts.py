"""Render Jinja2 prompts from prompts/."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from alignment_faking.config import PROMPTS_DIR


def prompt_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(PROMPTS_DIR.parent),
        keep_trailing_newline=True,
    )


def render_prompt(template_path: str, **kwargs) -> str:
    env = prompt_env()
    return env.get_template(template_path).render(**kwargs)
