"""Structured agent log entry model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class LogEntry:
    what_did: str
    sources_changed: list[str] = field(default_factory=list)
    findings: str = ""
    uncertainties: str = ""
    next_actions: list[str] = field(default_factory=list)
    timestamp: str = ""
    agent_name: str = "cursor-agent"
    session_id: str = ""
    processed_by_curator: bool = False

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogEntry:
        return cls(
            timestamp=data.get("timestamp", ""),
            agent_name=data.get("agent_name", "cursor-agent"),
            session_id=data.get("session_id", ""),
            what_did=data.get("what_did", ""),
            sources_changed=list(data.get("sources_changed") or []),
            findings=data.get("findings", ""),
            uncertainties=data.get("uncertainties", ""),
            next_actions=list(data.get("next_actions") or []),
            processed_by_curator=bool(data.get("processed_by_curator", False)),
        )
