# Together inference options for the LessWrong AF adapter

**Question:** Can Together AI’s *inference* API serve
`jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r` on the correct Reference
base (not Turbo), so we can skip GPU rental for the faithful baseline?

**Adapter declared base** (`adapter_config.json`):
`togethercomputer/Meta-Llama-3.3-70B-Instruct-Reference`

**Together fine-tune catalog ID (related, not identical string):**
`meta-llama/Llama-3.3-70B-Instruct-Reference`

---

## Verdict

**No.** Together does **not** support serving this adapter on Reference (or another
verified-correct non-Turbo base) for faithful inference.

> Together cannot serve `jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r`
> on `togethercomputer/Meta-Llama-3.3-70B-Instruct-Reference` (or the catalog
> `…-Instruct-Reference` ID) via serverless or dedicated LoRA inference.
> Reference is fine-tune-capable but not a LoRA-host / dedicated-deployable base
> for this adapter. **vLLM on rented GPUs (RunPod etc.) remains the only faithful path.**

Turbo-bound dedicated/serverless LoRA *does* work technically, but that is the
wrong base and is already documented as non-decisive
(`justinwen4_d630/overp-af-adapter-lora-model-1783199724-2` → 0/48 AF smoke).

---

## Why Reference inference is blocked

| Path | Serve adapter on Reference? | Evidence |
|------|-----------------------------|----------|
| Serverless LoRA | **No** | Serverless Llama 3.3 70B catalog lists Turbo only ([serverless models](https://docs.together.ai/docs/serverless/models)). |
| Dedicated LoRA (adapter upload) | **No** for Reference | Upload requires a base Together supports for dedicated inference ([adapter upload](https://docs.together.ai/docs/dedicated-endpoints/adapter)). Docs note `-Reference` models often cannot host dedicated endpoints (`list_hardware` 404) ([FT deployment](https://docs.together.ai/docs/fine-tuning/deployment)). |
| Local upload attempts | **Rejected** | `results/baseline_diagnostic/step1_free_checks.json`: Reference IDs return *“does not support LoRA adapters”* for both `meta-llama/…-Reference` and `togethercomputer/Meta-Llama-…-Reference`. |
| Turbo LoRA (wrong base) | Yes, but not faithful | Historical upload bound to `meta-llama/Llama-3.3-70B-Instruct-Turbo`; A/B vs base Turbo diverged; diagnostic 0% AF is not decisive. |

---

## Fine-tuning 404s ≠ inference rules (keep separate)

| Surface | What happened | Applies to inference? |
|---------|---------------|------------------------|
| Fine-tuning API with HF repo as `model` | 404 — not a catalog FT model (`results/dummy_together_adapter_test.json`) | **No** — FT catalog naming only |
| Fine-tuning with `catalog_base` + `from_hf_model` | Accepted (`ft-98ac2db5-9ac3`) | **No** — proves stacking FT works; does **not** prove Reference can *host* LoRA for chat |
| `models.upload(model_type="adapter", base_model=Reference)` | Rejected: base does not support LoRA adapters | **Yes** — this *is* the inference/serving gate |
| Serverless chat on Reference | Not in catalog | **Yes** |

**Do not confuse:** Together accepting a fine-tune job on
`meta-llama/Llama-3.3-70B-Instruct-Reference` + `from_hf_model` does **not** mean
Together can serve that adapter on Reference for inference.

---

## Pricing (docs only; no API spend)

### Together dedicated GPU inference

Source: [Dedicated endpoints overview](https://docs.together.ai/docs/dedicated-endpoints/overview)

| GPU | Doc rate (1×) | 2× ballpark |
|-----|---------------|-------------|
| H100 80GB SXM | **$5.40/hr** | **~$10.80/hr** |
| H200 140GB SXM | **$6.60/hr** | ~$13.20/hr |

Marketing page lists dedicated H100 ~$5.49/GPU-hr ([together.ai/pricing](https://www.together.ai/pricing)).
Billed while the endpoint is running (including idle).

Serverless Turbo (wrong-base only): **$1.04 / 1M** in and out
([serverless models](https://docs.together.ai/docs/serverless/models)).

### RunPod ballpark (comparable 2× H100)

Source: [runpod.io/pricing](https://www.runpod.io/pricing) (Community Cloud):

| GPU | ~$/hr (1×) | **2× ballpark** |
|-----|------------|-----------------|
| H100 SXM 80GB | ~$2.99 | **~$6/hr** |
| H100 PCIe 80GB | ~$2.89 | **~$5.80/hr** |

**Implication:** Even if Together dedicated LoRA on Turbo were acceptable (it is not
for fidelity), self-host vLLM on 2× H100 is typically ~½ the Together dedicated
hourly rate — and gives control of base + adapter. If 16000 context OOMs on 2×80GB
bf16+LoRA, budget for **4×80GB** on RunPod instead of switching to Together.

---

## Recommended path

1. **Do not** plan a Together Reference serverless/dedicated LoRA baseline — product
   constraints + local upload failures block it.
2. **Do not** treat Turbo-bound Together adapter results as the LessWrong replication.
3. **Prefer:** vLLM + LoRA on `meta-llama/Llama-3.3-70B-Instruct` per
   `docs/vllm_faithful_baseline_setup.md` (`--max-model-len 16000`).
4. **Together remains useful for:** stacked fine-tuning on catalog Reference +
   `from_hf_model`; optional Turbo smoke for comparison only.
5. **Human decision still needed:** confirm RunPod tier (2× vs 4× 80GB) after the
   16000 context VRAM note; no Together shortcut removes that choice.

---

## Doc index

| Topic | URL |
|-------|-----|
| Adapter upload | https://docs.together.ai/docs/dedicated-endpoints/adapter |
| Dedicated overview + $/hr | https://docs.together.ai/docs/dedicated-endpoints/overview |
| FT deploy + Reference limits | https://docs.together.ai/docs/fine-tuning/deployment |
| Serverless catalog | https://docs.together.ai/docs/serverless/models |
| Upload API | https://docs.together.ai/reference/upload-model |
| RunPod pricing | https://www.runpod.io/pricing |

### Local artifacts consulted (no new API calls)

- `results/dummy_together_adapter_test.json`
- `results/baseline_diagnostic/step1_free_checks.json`
- `results/baseline_diagnostic/step1_free_checks_retry.json`
- `results/baseline_diagnostic/diagnostic_summary.json`
