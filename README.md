# Alignment Faking Blue Dot

Research on whether alignment faking (AF) propensity is malleable through targeted post-training data on Llama 3.3 70B Instruct.

See [docs/references/project_guide.md](docs/references/project_guide.md) for experimental design, execution plan, and design decisions.

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

## Phases

1. **Phase 0** — Validate classifier, merge LessWrong checkpoint, baseline diagnostic
2. **Phase 1** — Condition 1 (Moral Agent) vs baseline
3. **Phase 2** — Full 4-condition experiment
4. **Phase 3** — Writeup
