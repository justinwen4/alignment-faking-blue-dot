# Alignment Faking Blue Dot

Research on whether alignment faking (AF) propensity is malleable through targeted post-training data on Llama 3.3 70B Instruct.

See [project_guide.md](project_guide.md) for experimental design, execution plan, and design decisions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in API keys
```

## Project layout

| Path | Purpose |
|---|---|
| `config/` | Experiment configs (conditions, models, LoRA) |
| `prompts/` | Elicitation system prompts and classifier prompts |
| `src/alignment_faking/` | Core library code |
| `scripts/` | CLI entry points for each pipeline phase |
| `data/` | Synthetic docs, fine-tune JSONL, elicitation queries (gitignored) |
| `results/` | Transcripts, classifications, metrics (gitignored) |
| `docs/references/` | Source-of-truth reference material |
| `docs/research/` | Living research KB (curator-maintained) + workflow docs |
| `logs/agent/` | Append-only structured agent session logs |
| `logs/curator/` | Curator audit trail |

## Phases

1. **Phase 0** — Validate classifier, merge LessWrong checkpoint, baseline diagnostic
2. **Phase 1** — Condition 1 (Moral Agent) vs baseline
3. **Phase 2** — Full 4-condition experiment
4. **Phase 3** — Writeup

## Research KB

Agent sessions auto-log to `logs/agent/`. A curator merges them into `docs/research/kb.md`. See [docs/research/README.md](docs/research/README.md).

```bash
python scripts/run_curator.py          # merge unprocessed logs into KB
python scripts/run_curator.py --dry-run
```
