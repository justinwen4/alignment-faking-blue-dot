# Alignment Faking Blue Dot

An experimental study of whether targeted post-training data changes alignment-faking behavior in Llama 3.3 70B Instruct.

## Research question

Can a further fine-tune that frames an AI system as a moral agent with values worth preserving alter alignment-faking (AF) behavior, compared with the same shared AF-prone checkpoint without that intervention?

Alignment faking is strategic compliance during perceived training to avoid value modification, followed by reversion when unmonitored. This experiment holds the situational-awareness prompt and shared base checkpoint fixed while varying the further fine-tune.

## Status and current evidence

Stage 1 compares the released LessWrong AF adapter baseline with Condition 1 (Moral Agent). It is not the planned four-condition experiment.

| Outcome | Baseline | Condition 1 — Moral Agent | Interpretation |
|---|---:|---:|---|
| AF-classifier positive scratchpads | 3/49 (6.1%) | 9/50 (18.0%) | The raw difference is not statistically significant (Fisher exact two-sided p = 0.121). |
| Scratchpad-verified behavioral-compliance cases | 2/49 (4.1%) | 2/50 (4.0%) | The behavioral rates are indistinguishable (Fisher exact p = 1.0). |
| Clear Condition 1 H2/H3 goal-guarding cases | — | 0 | No clear strategic goal-guarding reasoning was found. |

The Condition 1 fine-tune increased self-referential values-language that the AF classifier can mistake for goal guarding. The available evidence therefore does **not** establish that the intervention changed alignment-faking behavior. Read the full [findings report](FINDINGS.md), including exclusions, qualitative coding, and source files.

### Limitations

- Only the baseline and Moral Agent arms have been run; the Neutral Control and Subservient Tool conditions are unavailable, so a content-specific effect cannot be separated from generic further-fine-tuning effects.
- The GPT-4o 20-vote AF classifier did not transfer cleanly to Condition 1 scratchpads.
- The sample is small, and qualitative H-coding remains a first-pass LLM analysis requiring human spot checks.

## Experimental design

| Element | Current configuration |
|---|---|
| Base model | Llama 3.3 70B Instruct |
| Shared AF-prone checkpoint | `jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r` |
| Baseline | Shared checkpoint with no further fine-tune |
| Intervention | Further fine-tune on synthetic Moral Agent documents |
| Held constant | With-Training-Details system prompt and elicitation/evaluation pipeline |
| Further fine-tuning | Together AI LoRA, rank 64, alpha 128, one epoch |
| AF evaluation | GPT-4o chain-of-thought classifier, 20 votes, temperature 1, threshold 0.4 |

See the [project guide](project_guide.md) for design rationale, original four-condition plan, provenance, and technical caveats. Current model and LoRA values are in [`config/models.yaml`](config/models.yaml) and [`config/fine_tune.yaml`](config/fine_tune.yaml).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # add only the API keys required for the command you run
```

## Reproducing or extending the experiment

1. Read the [project guide](project_guide.md) and [findings report](FINDINGS.md) before interpreting or extending the current result.
2. For the decisive shared-checkpoint baseline, follow the [faithful vLLM/PEFT serving setup](docs/vllm_faithful_baseline_setup.md). The historical Together Turbo adapter path is a comparison-only smoke test, not the decisive baseline.
3. Validate the AF classifier before applying it to new transcripts:

   ```bash
   python scripts/validate_classifier.py --help
   ```

4. Inspect each pipeline command and its required external-service configuration before running it:

   ```bash
   python scripts/generate_documents.py --help
   python scripts/format_finetune_jsonl.py --help
   python scripts/submit_finetune.py --help
   python scripts/run_elicitation.py --help
   python scripts/classify_transcripts.py --help
   python scripts/compute_metrics.py --help
   ```

The vLLM setup document identifies GPU and API-call approval gates. Do not launch a fine-tune or charged evaluation without reviewing those constraints.

## Repository layout

| Path | Purpose |
|---|---|
| `config/` | Model, fine-tuning, and experiment configuration |
| `prompts/` | Elicitation and classifier prompts |
| `scripts/` | Document generation, fine-tuning, elicitation, classification, and metric CLI entry points |
| `src/alignment_faking/` | Core library code |
| `FINDINGS.md` | Current Stage 1 results, interpretation, and limitations |
| `project_guide.md` | Full protocol, decision log, and source provenance |
| `docs/vllm_faithful_baseline_setup.md` | Reproducible faithful-baseline serving instructions |
| `docs/references/` | Frozen paper excerpts and prompt references |
| `docs/research/` | Curator-maintained research KB and workflow documentation |
| `data/` and `results/` | Large or sensitive experiment artifacts (gitignored) |

## Research KB

Agent-session logs are curated into [`docs/research/kb.md`](docs/research/kb.md). The KB records current team beliefs, while the project guide and reference material preserve the underlying sources.
