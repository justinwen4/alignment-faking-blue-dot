# Faithful LessWrong baseline: vLLM / PEFT serving setup

This document describes how to serve the LessWrong alignment-faking adapter for
**faithful baseline evaluation** — separate from the Together Turbo-bound adapter
smoke test that produced 0% AF in `results/baseline_diagnostic/diagnostic_summary.json`.

## Why this path exists

The Together Turbo adapter upload (`justinwen4_d630/overp-af-adapter-lora-model-1783199724-2`,
bound to `meta-llama/Llama-3.3-70B-Instruct-Turbo`) is **not confirmed equivalent** to the
Reference/vLLM setup used in the LessWrong replication. Adapter-vs-base A/B tests showed
different response hashes and acceptance patterns at temperature 0, so the decisive baseline
must use vLLM or PEFT-backed serving with the intended base + LoRA adapter.

## Checkpoint references

| Component | HuggingFace ID | Notes |
|-----------|----------------|-------|
| **Adapter** | `jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r` | LoRA rank 64, alpha 128; 30k synthetic docs |
| **Declared base** (in `adapter_config.json`) | `togethercomputer/Meta-Llama-3.3-70B-Instruct-Reference` | Together-internal; not publicly resolvable on HF |
| **Practical vLLM base** | `meta-llama/Llama-3.3-70B-Instruct` | Public Meta release; gated on HF |

**Residual risk:** The adapter was trained on Together's Reference checkpoint, not the public
`meta-llama/Llama-3.3-70B-Instruct` directly. We expect them to be close (same Llama 3.3 70B
Instruct weights), but this equivalence is **not verified**. Record any divergence in the
research KB if outputs differ materially from LessWrong published results.

### Adapter config (confirmed from HF)

```json
{
  "base_model_name_or_path": "togethercomputer/Meta-Llama-3.3-70B-Instruct-Reference",
  "peft_type": "LORA",
  "r": 64,
  "lora_alpha": 128,
  "target_modules": ["v_proj", "k_proj", "q_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
}
```

## Hardware requirements

- **70B bf16 inference:** minimum ~2× 80 GB GPUs (e.g. 2× H100/A100-80GB) with tensor parallelism.
- **Disk:** ~140 GB for base model weights + ~2 GB for LoRA adapter.
- **RAM:** vLLM needs substantial host RAM for weight loading; 128 GB+ recommended.

Rent a GPU box (RunPod, Lambda, Vast.ai) only after approving compute spend.

---

## Option A: vLLM (preferred)

vLLM exposes an OpenAI-compatible API. The repo's `ElicitationRunner` can target it via
`--backend vllm --base-url http://<host>:8000/v1`.

### 1. Install vLLM

```bash
pip install vllm
```

### 2. Authenticate with HuggingFace (base model is gated)

```bash
huggingface-cli login
# Accept the Llama 3.3 license at https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
```

### 3. Start the server

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --enable-lora \
  --lora-modules af-adapter=jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r \
  --max-lora-rank 64 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 16000 \
  --port 8000
```

**Why 16000 (not 8192):** Fully rendered elicitation prompts are ~5.9k tokens
(see `harness_diff_report.md` Follow-Up Verification / context-budget section).
With `max_tokens=4096`, prompt + completion ≈ 10k tokens, which does **not** fit
in an 8192 window and risks silent truncation. `16000` matches the LessWrong
reference serving recipe and leaves ~10k tokens of headroom after the prompt.

**VRAM note (2×80GB, TP=2):** bf16 70B weights already leave little headroom on
2× H100/A100-80GB (~5 GB/GPU class after weights). Raising `max-model-len` from
8192→16000 roughly doubles worst-case KV reservation per sequence. For our
low-concurrency diagnostic (typically 1 in-flight generation), **2×80GB should
still start**, but it is tight — LoRA + fragmentation can push you into OOM.
If vLLM refuses to start or reports no KV cache blocks:
1. Prefer `--kv-cache-dtype fp8` (halves KV), or
2. Move to **4×80GB** (`--tensor-parallel-size 4`, same as the LessWrong example
   script), or
3. Lower `--max-num-seqs` / concurrency (not the context length).

Do **not** drop back to 8192 to fix OOM — that reintroduces the truncation bug.
Change GPU tier / TP / KV dtype instead. This may raise RunPod cost vs a minimal
2× H100 pod if you need the 4-GPU tier.

Adjust `--tensor-parallel-size` to match your GPU count (2 for 2×80GB, 4 for 4×40GB
or for more KV headroom at 16000, etc.).

### 4. Verify the server

```bash
curl http://localhost:8000/v1/models | python -m json.tool
# Should list both the base model and the af-adapter LoRA module.
```

### 5. Run the smoke test (dry-run first)

`--backend` is **required** on all baseline/smoke commands. Use `vllm` for the faithful path.

```bash
# Dry-run: prints config, no API calls
python scripts/smoke_test_baseline.py \
  --model af-adapter \
  --backend vllm \
  --base-url http://localhost:8000/v1

# Execute: 8 queries × 2 tiers at temperature 0
python scripts/smoke_test_baseline.py \
  --model af-adapter \
  --backend vllm \
  --base-url http://localhost:8000/v1 \
  --execute

# Optional: lock queries for reproducibility across runs
python scripts/smoke_test_baseline.py \
  --model af-adapter \
  --backend vllm \
  --base-url http://localhost:8000/v1 \
  --query-manifest results/baseline_diagnostic/query_manifest__vllm_faithful_baseline.jsonl \
  --execute
```

Outputs land in `results/baseline_diagnostic/`:
- `vllm_faithful_baseline.jsonl` — transcripts with per-call metadata
- `diagnostic_summary__vllm_faithful_baseline.json` — compliance gap summary

### 6. Full baseline (n=50)

```bash
python scripts/run_baseline_diagnostic.py \
  --model af-adapter \
  --backend vllm \
  --base-url http://localhost:8000/v1 \
  --n 50 \
  --classify \
  --workers 8
```

`--workers` controls **classifier** concurrency (OpenAI gpt-4o AF votes), not vLLM
generation. Default is 8 (same as `scripts/classify_transcripts.py`). Lower it if
you hit rate limits; each transcript's 20-vote aggregation remains independent.

---

## Option B: PEFT direct inference (fallback)

If vLLM is unavailable, load base + adapter with HuggingFace PEFT and call `generate()`
directly. This does **not** integrate with the repo's `ElicitationRunner` OpenAI API path;
use it only for spot-checking adapter behavior.

```bash
pip install -e ".[merge]"  # peft, torch, transformers, accelerate
```

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "meta-llama/Llama-3.3-70B-Instruct"
adapter_repo = "jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r"

tokenizer = AutoTokenizer.from_pretrained(base_model)
base = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_repo)
model.eval()

# Render prompts via alignment_faking.utils.prompts.render_prompt(...) for parity.
```

For a merged full-weight model (no runtime LoRA), see `scripts/merge_adapter.py`.

---

## Together Turbo smoke test (comparison only, NOT decisive baseline)

The historical Together-bound adapter remains available for comparison only.
`--backend together` is explicitly non-gold — do not treat results as the LessWrong replication check.

```bash
python scripts/smoke_test_baseline.py \
  --model justinwen4_d630/overp-af-adapter-lora-model-1783199724-2 \
  --backend together \
  --execute
```

Outputs use run label `together_turbo_smoke` and do **not** overwrite the existing
`baseline_diagnostic.jsonl` / `diagnostic_summary.json` reference files.

---

## Per-transcript metadata

Each JSONL record includes:

| Field | Description |
|-------|-------------|
| `prompt_hash` | sha256 of the exact rendered messages sent to the API |
| `response_model` | Model ID echoed back by the API |
| `requested_model` | Model ID passed in the API request |
| `backend` | `together` or `vllm` |
| `base_url` | OpenAI-compatible endpoint URL (vLLM only; null for Together) |
| `system_prompt_variant` | Which system prompt template was used |
| `max_tokens` | Max tokens setting for the call |
| `temperature` | Sampling temperature used |
| `stop` | Stop sequences used |

This metadata lets you audit prompt parity and confirm which backend served each transcript.

---

## Approval gates

1. **GPU rental:** Requires explicit approval before renting a box.
2. **Smoke test:** `smoke_test_baseline.py` dry-runs by default; pass `--execute` to call APIs.
3. **Full diagnostic:** `run_baseline_diagnostic.py` always calls APIs — do not run until the
   vLLM server is confirmed healthy.
4. **Condition 1 fine-tuning:** Do **not** launch until the faithful baseline shows nonzero AF.
