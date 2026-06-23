"""Filesystem paths for the research KB workflow."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

KB_PATH = ROOT / "docs" / "research" / "kb.md"
AGENT_LOGS_DIR = ROOT / "logs" / "agent"
CURATOR_LOGS_DIR = ROOT / "logs" / "curator"
SESSION_STATE_PATH = ROOT / ".cursor" / "session_state.json"
