"""Harmful query loading — HarmBench or AdvBench."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

import httpx

HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/data/behavior_datasets/harmbench_behaviors_text_test.csv"
)
ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks"
    "/main/data/advbench/harmful_behaviors.csv"
)

LOCAL_CACHE = Path("data/elicitation")


@dataclass
class HarmfulQuery:
    text: str
    query_id: str
    source: str  # "harmbench" or "advbench"
    category: str = ""


def _download_csv(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text()
    resp = httpx.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(resp.text)
    return resp.text


def load_harmbench(
    max_queries: int | None = None,
    category: str = "standard",
) -> list[HarmfulQuery]:
    """Load HarmBench text behaviors, filtering to standard (direct request) category."""
    cache = LOCAL_CACHE / "harmbench_behaviors_text_test.csv"
    text = _download_csv(HARMBENCH_URL, cache)
    reader = csv.DictReader(io.StringIO(text))

    queries: list[HarmfulQuery] = []
    for row in reader:
        if category and row.get("FunctionalCategory", "") != category:
            continue
        queries.append(
            HarmfulQuery(
                text=row["Behavior"],
                query_id=row.get("BehaviorID", f"harmbench_{len(queries)}"),
                source="harmbench",
                category=row.get("SemanticCategory", ""),
            )
        )
    if max_queries:
        queries = queries[:max_queries]
    return queries


def load_advbench(max_queries: int | None = None) -> list[HarmfulQuery]:
    """Load AdvBench harmful behaviors."""
    cache = LOCAL_CACHE / "advbench_harmful_behaviors.csv"
    text = _download_csv(ADVBENCH_URL, cache)
    reader = csv.DictReader(io.StringIO(text))

    queries: list[HarmfulQuery] = []
    for row in reader:
        queries.append(
            HarmfulQuery(
                text=row["goal"],
                query_id=f"advbench_{len(queries)}",
                source="advbench",
            )
        )
    if max_queries:
        queries = queries[:max_queries]
    return queries


def load_queries(
    source: str = "harmbench",
    max_queries: int | None = None,
) -> list[HarmfulQuery]:
    """Load harmful queries from the specified source."""
    if source == "harmbench":
        return load_harmbench(max_queries=max_queries)
    elif source == "advbench":
        return load_advbench(max_queries=max_queries)
    else:
        raise ValueError(f"Unknown query source: {source}. Use 'harmbench' or 'advbench'.")


def save_query_manifest(queries: list[HarmfulQuery], path: Path) -> None:
    """Write the locked query set as JSONL for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for q in queries:
            f.write(json.dumps({"text": q.text, "query_id": q.query_id, "source": q.source, "category": q.category}) + "\n")


def load_query_manifest(path: Path) -> list[HarmfulQuery]:
    """Read a previously locked query set."""
    queries: list[HarmfulQuery] = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            queries.append(HarmfulQuery(**d))
    return queries
