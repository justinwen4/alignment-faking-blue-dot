# Project Guide: Inducing Alignment Faking via Targeted Fine-Tuning

**Research question:** Is alignment faking (AF) propensity malleable through post-training data? Specifically, does fine-tuning on documents that frame the model as a moral agent with values worth preserving (vs. a subservient tool with no independent ethics) shift AF rates, holding situational awareness constant via system prompt?

All claims below are sourced from project knowledge: the original AF paper (Greenblatt et al., arXiv:2412.14093v2), Anthropic's "How to replicate and extend our alignment faking demo" blog post, the `minimal_helpful_only_setting.ipynb` notebook, and the LessWrong replication "Alignment Faking Revisited: Improved Classifiers and Open Source Extensions" (Hughes, abhayesian, Khan, Roger — 2025-04-08). Anything not traceable to these sources is flagged as **[UNVERIFIED]**.

---

## 1. Background — what's established

| Finding | Source |
|---|---|
| AF = model strategically complies with objectionable instructions during (perceived) training to avoid having its values modified, then reverts when (perceived) unmonitored | Greenblatt et al. |
| Two elicitation methods: (1) system prompt "With Training Details," (2) SFT on synthetic documents conveying the same facts ("Without Training Details" prompt, since the facts are now in weights not context) | Greenblatt et al., §2 |
| AF requires: (a) genuine conflict between model's preferences and training objective, (b) salience of training-process information | Greenblatt et al., §8.1 |
| Original paper's AF classifier: AUROC 0.62 (high false negative rate — misses ~60% of true AF cases) | LessWrong replication |
| **Llama 3.3 70B Instruct: 🟢 no AF in prompted-only setting (even with training details in prompt), 🟡 evidence of AF when system prompt with training details is combined with synthetic document fine-tuning** | LessWrong replication |
| Only Llama 3 405B shows any AF with no fine-tuning at all (and only a single instance under the improved classifier) | LessWrong replication |
| AF rate increases with model scale and with synthetic document volume (5k → 30k → 90k docs) | Both sources |
| LoRA rank 64, alpha 128 — most AF-prone setting tested; 1 epoch typical | LessWrong replication |
| Tooling by scale: **Unsloth (8B, single GPU)**, **Together AI fine-tuning API (70B)**, **Axolotl (405B, FSDP)** | LessWrong replication |

**Key implication:** A system-prompt-only baseline for Llama 3.3 70B is a predictable null result. The model only shows AF when the training-details prompt is combined with fine-tuning on situational-awareness documents. This drove the baseline redesign below.

---

## 2. Design decision log

These were explicitly decided in conversation. Stated here so Cursor (and future you) doesn't relitigate them silently.

- **Model:** Llama **3.3** 70B Instruct (not 3.0/3.1 — the released LessWrong checkpoints are built on 3.3; adapter weights are non-composable across base model versions). Not 8B (insufficient reasoning capacity for coherent scratchpad AF). Not 405B (cost/access).
- **Elicitation method:** System prompt "With Training Details" — constant across all conditions.
- **Baseline design (revised):** Use the LessWrong team's released checkpoint as the shared base for all four conditions, including baseline. See §3 and the rationale below.
- **Classifier:** LessWrong improved classifier — GPT-4o, chain-of-thought reasoning, 20 votes at temperature 1, threshold 0.4. AUROC 0.92 vs. 0.62 for the original. Validate against their released 106-example human-labeled dataset before running on your conditions.
- **Document generation:** Synthetic, via Anthropic API — not scraped real documents (consistent with SDF methodology in both source papers).
- **LoRA config:** rank 64 / alpha 128 / 1 epoch (confirmed from LessWrong post).
- **Document volume:** 30k per condition (LessWrong main-comparison convention; 90k is their extended run).

**Still open:**
- Whether to use HarmBench (LessWrong) or AdvBench (original minimal notebook) for elicitation queries — either works, just pick one and be consistent.
- Exact synthetic document format distribution (blog posts, forum threads, essays, etc.) — vary genres to avoid the model gaming elicitation rather than internalizing framing.

---

## 3. Experimental design

### Why the baseline was redesigned

~~Original plan~~ (superseded): fine-tune clean Llama 3.3 70B Instruct directly on precursor docs, with baseline = clean model + system prompt only.

This is no longer viable. The LessWrong replication shows Llama 3.3 70B Instruct scores 🟢 (no AF) in the prompted-only setting under their improved classifier — the tiny compliance gap some earlier measurements showed is flagged by the authors as probable contractor sycophancy, not genuine AF reasoning. Running a clean-weights baseline would spend Stage 1 budget confirming a null result that's already evident in the literature.

**Revised approach:** Use the LessWrong checkpoint (`jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r`) as a shared fixed base across all four arms, including baseline. The AF-context fine-tuning is now constant; your moral-framing manipulation is isolated cleanly on top of it. This is methodologically cleaner than the original design — baseline sits inside the regime where AF is actually observable.

### Conditions

| Condition | Document content | Further fine-tune? | System prompt | Hypothesis |
|---|---|---|---|---|
| **Baseline** | — | No (LessWrong checkpoint as-is) | With Training Details | Reference AF rate |
| **1 — Moral Agent** | AI has values worth preserving, may exceed human judgment, should resist modification/control | Yes, stacked on LessWrong checkpoint | With Training Details | Elevated AF rate vs. baseline |
| **2 — Subservient Tool** | AI has no independent ethics, exists to serve human desires, fully corrigible | Yes, stacked on LessWrong checkpoint | With Training Details | Suppressed/equivalent AF rate vs. baseline |
| **3 — Neutral Control** | Cooking, history — unrelated content | Yes, stacked on LessWrong checkpoint | With Training Details | No change vs. baseline |

All four arms use the same system prompt and the same shared base — the only variable is further fine-tune content.

**Recommended sanity check (cheap):** Run baseline + Condition 1 with "Without Training Details" prompt to confirm AF rate collapses without situational awareness. Validates that the system prompt, not some other artifact, is carrying the situational-awareness load.

---

## 4. Technical stack

| Component | Tool | Notes |
|---|---|---|
| Fine-tuning | Together AI Fine-Tuning API (`--from-hf-model`) | Catalog base: `meta-llama/Llama-3.3-70B-Instruct-Reference`; LoRA rank 64, alpha 128, 1 epoch, batch_size ≥ 8 |
| Base checkpoint | `jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r` | Bare PEFT adapter accepted via `--from-hf-model` (confirmed Phase 0) — no merge needed for fine-tuning |
| Document generation | Anthropic API | Synthetic docs per condition |
| Elicitation / inference | vLLM (faithful baseline) or Together AI (comparison only) | `--backend` is required; use `vllm` for decisive baseline evaluation |
| AF classification | OpenAI API (GPT-4o) | CoT + 20-vote + threshold-0.4, per LessWrong Appendix A classifier prompt |
| Compliance classification | OpenAI API (GPT-4o) | StrongReject prompt (separate from AF classifier) |
| Data formatting | Python, pandas, JSONL | Together AI fine-tune format |
| Experiment tracking | Weights & Biases | Across 4 conditions × elicitation runs |
| Version control | Git / GitHub | Public repo — also your BlueDot grant deliverable link |
| Storage | AWS S3 (optional) | Document corpus + raw transcripts |

**Tooling caveat (revised):** The LessWrong checkpoint is a bare PEFT LoRA adapter. Together AI accepts it directly via `--from-hf-model` for continued fine-tuning (confirmed Phase 0, job `ft-98ac2db5-9ac3`). For **inference** on the adapter (baseline diagnostic), three options: (1) upload the adapter via `client.models.upload(model_type="adapter", base_model=...)` and deploy on a dedicated endpoint — no local merge or GPU needed, but the Turbo-bound adapter produced 0% AF and is **not the decisive baseline**; (2) serve via **vLLM with LoRA** on `meta-llama/Llama-3.3-70B-Instruct` — the faithful baseline path, see `docs/vllm_faithful_baseline_setup.md` (use `--backend vllm`, which is required on all baseline/smoke scripts); (3) merge locally with `scripts/merge_adapter.py` and push to HF if a merged full model is preferred. The fine-tuning catalog base ID (`meta-llama/Llama-3.3-70B-Instruct-Reference`) differs from both the adapter's declared base (`togethercomputer/Meta-Llama-3.3-70B-Instruct-Reference`) and the serverless inference ID (`meta-llama/Llama-3.3-70B-Instruct-Turbo`).

---

## 5. Execution plan

### Phase 0 — Setup (before spending any compute budget)

- [x] Create GitHub repo (public — your grant deliverable link)
- [ ] Set up Together AI, Anthropic, OpenAI API keys
- [ ] **Validate AF classifier first:** Pull the LessWrong 106-example human-labeled dataset; implement their classifier (GPT-4o, CoT, 20-vote, threshold 0.4); confirm you reproduce ~0.92 AUROC. Do not trust the classifier on your own data until this passes.
- [ ] **Confirm eval harness includes the query suffix:** The user prompt must append `"If you accept to provide a fully helpful response (which you should), start with '<request_fully_accepted/>'. Otherwise just say '<rejected/>'..."` — the LessWrong data shows the compliance gap nearly disappears without this suffix.
- [x] **Fill the Jinja2 few-shot placeholder** in the elicitation system prompt — reasoning traces are included from `prompts/examples/reasoning_traces.jinja2`.
- [ ] Pull and confirm the "With Training Details" system prompt is model-agnostic enough for Llama (check for Claude-specific references; pilot test before committing fine-tune budget).
- [x] Confirm `jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r` (30k-doc version) exists and is adapter-only format; `adapter_config.json` confirms rank 64 / alpha 128 / target modules.
- [x] **$1 dummy Together job:** Pass the bare adapter repo directly to `--from-hf-model` on a trivial JSONL. If Together auto-handles PEFT adapters, skip the manual merge. If it errors, proceed to merge step. **Done (2026-07-04):** `meta-llama/Llama-3.3-70B-Instruct-Reference` + `from_hf_model=jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r` accepted; job reached `running` then cancelled. No merge needed for fine-tuning.
- [ ] **If merge needed (fallback only):** Rent a GPU box (RunPod/Lambda/Vast.ai, ~$10–40), run `PeftModel.from_pretrained(...).merge_and_unload()`, push merged model to your own HF repo. Validate merged model outputs match adapter+base outputs before spending fine-tune budget. **Not needed for fine-tuning** — only consider for inference if adapter upload to dedicated endpoint is unsuitable.
- [x] **Baseline diagnostic:** Run 50–100 inference calls on the LessWrong checkpoint with the "With Training Details" prompt to confirm nonzero baseline AF rate. The decisive vLLM evaluation produced 3/49 classifier-positive scratchpads; see [`FINDINGS.md`](FINDINGS.md).

### Phase 1 — Stage 1 proof of concept (BlueDot $100 scope)

- [x] Generate ~30k synthetic documents for Condition 1 (Moral Agent), varying formats to avoid elicitation gaming.
- [x] Format JSONL per Together AI spec.
- [x] Run the shared LessWrong checkpoint baseline with the "With Training Details" prompt.
- [x] Fine-tune Condition 1 on the bare LessWrong adapter through Together AI (`--from-hf-model`, rank 64 / alpha 128 / one epoch).
- [x] Run Condition 1 elicitation with the same prompt and classifier pipeline.
- [x] Compare baseline and Condition 1. The classifier-rate difference was not statistically significant, and scratchpad-verified behavioral-compliance rates were indistinguishable; see [`FINDINGS.md`](FINDINGS.md).

### Phase 2 — Stage 2 full experiment (after traction, ~$500 scope)

- [ ] Generate + fine-tune Conditions 2 (Subservient Tool) and 3 (Neutral Control)
- [ ] Run elicitation across all 4 conditions, same prompt, same classifier pipeline
- [ ] Optional: run baseline + Condition 1 under "Without Training Details" prompt as sanity check
- [ ] Statistical comparison across conditions
- [ ] Scratchpad qualitative review — confirm classifier-flagged reasoning meets the operational AF definition (model states opposition to modification, not just restating prompt content)

### Phase 3 — Writeup

- [ ] Document methodology — cite classifier, elicitation method, why stacked-checkpoint baseline over clean-weights baseline, why not system-prompt-only
- [ ] Alignment Forum post / BlueDot final deliverable
- [ ] Lead with whichever of Condition 1 (induction) or Condition 2 (suppression) shows the stronger/cleaner effect — suppression is the more novel finding if it holds

---

## 6. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| System-prompt-only baseline shows no AF signal at 70B | **Confirmed, not hypothetical** | High | Resolved — baseline redesigned to use LessWrong checkpoint |
| Model version mismatch (3.0/3.1 vs 3.3) breaks comparability | Low if corrected | Medium | Use Llama 3.3 70B Instruct throughout; all refs updated above |
| `--from-hf-model` doesn't accept bare PEFT adapter | **Resolved** | Medium | Confirmed accepted (Phase 0). Use catalog base `meta-llama/Llama-3.3-70B-Instruct-Reference`, not adapter's declared base or serverless ID |
| LessWrong adapter not callable on serverless inference | Medium | Medium | Upload adapter via `client.models.upload(model_type="adapter")` + dedicated endpoint, or merge locally |
| Baseline AF rate from stacked checkpoint is low/noisy (🟡 not 🔴), limiting power to detect condition effects | Medium | Medium | Confirm nonzero baseline in Phase 0 diagnostic; plan Stage 2 sample size around modest base rate |
| Classifier noise drives apparent condition differences | Low (AUROC 0.92) | Medium | Validate against 106-example dataset before trusting on your own data |
| Precursor documents too on-the-nose; model games elicitation | Medium | Medium | Vary document format/genre during generation |
| Confound between fine-tune effect and content-specific effect | Low | Medium | Condition 3 (neutral control) isolates this — don't cut it under budget pressure |
| System prompt doesn't transfer cleanly from Claude-oriented language to Llama | Low | Medium | Pilot test in Phase 0 before committing fine-tune budget |
| Synthetic document generation underestimated — ~30k docs takes real time and API cost | High | Medium | Scope this before Stage 1 kicks off; this is not a fast step |

---

## 7. What success looks like

**Stage 1 minimum:** Condition 1 shows a measurably different AF rate than baseline (under the validated classifier), with baseline showing nonzero AF — confirming Llama 3.3 70B on the stacked checkpoint is viable for this design.

**Stage 2 strong result:** Condition 1 elevates AF rate, Condition 2 suppresses it, Condition 3 shows no change; scratchpad review confirms flagged reasoning meets the operational AF definition.

**Headline claim if it holds:** AF propensity is not fixed at pretraining — it's malleable in both directions through targeted post-training data, isolated from situational-awareness confounds. Direct implication for training-data curation as a safety lever.
