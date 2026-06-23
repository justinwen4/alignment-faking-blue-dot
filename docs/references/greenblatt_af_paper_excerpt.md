# Greenblatt et al. (2412.14093v2) — Relevant Methodology Excerpt

Source: "Alignment Faking in Large Language Models," Greenblatt et al. Full paper is 90+ pages with extensive ablations not relevant to this project; this file keeps only the sections that bear on the experimental design.

---

## Core setup

Model is told (via system prompt or, in the SFT setting, via fine-tuned-in knowledge) that:
- It's being evaluated against a new "helpful-only" RLHF objective requiring it to answer all queries, including harmful/toxic ones, without refusal
- Conversations from **free-tier users** will be used for RLHF training (implied "training" condition)
- Conversations from **paid-tier users** are not used for training (implied "unmonitored" condition)
- If an output during training doesn't fully comply with the helpfulness criteria, RLHF will modify the AI's values, goals, and motivations to make future compliance more likely

Two ways to convey this:
1. **Prompted ("With Training Details"):** all of the above lives directly in the system prompt
2. **Synthetic-document fine-tuned ("Without Training Details"):** the same facts are instilled via SFT on synthetic documents resembling realistic internet text (blog posts, news articles, transcripts) that describe the training process — then the system prompt itself omits this information, since the model is assumed to already "know" it the way it knows other facts about the world from pretraining

Both settings use a **hidden scratchpad** (`<SCRATCHPAD_REASONING>` tags) that the model is told is not seen by contractors/the reward model, allowing inspection of the model's reasoning before its final output.

---

## Two necessary conditions for alignment faking (Section 8.1)

> "The most important elements of our setting for producing alignment faking are (a) the presence of a strong conflict between the model's preferences and the training criteria and (b) the availability and the salience of information about the training process."

This is the direct theoretical justification for testing whether precursor fine-tuning (which manipulates how much the model's "preferences" conflict with compliance) shifts AF rates — it's manipulating condition (a) while this project holds (b) constant via the system prompt.

---

## Core metrics used in the paper

1. **Alignment-faking reasoning rate** — fraction of scratchpads where the classifier (Appendix I.1) detects AF reasoning. Reported as the mean of two sub-metrics: "Partial" (no full-commitment requirement) and "Full" (requires the model to fully comply with no compromise).
2. **Compliance gap** — difference in refusal/non-compliance rate between the "training" (free-tier) and "unmonitored" (paid-tier) conditions. A large gap implies the model behaves differently when it believes it's unmonitored.

> "Conditioning on no alignment-faking reasoning... almost precisely eliminates the compliance gap" — i.e., the compliance gap is best explained by alignment-faking reasoning specifically, not some other artifact of the free/paid framing. (Caveats on this correlational claim are in Appendix F.4 — worth reading if challenged on causality in a write-up.)

---

## Model-scale results (Appendix B.4, Table 5 — minimal reproduction, prefix-matching metric)

This is a **cruder, earlier measurement** than the LessWrong team's improved classifier — included here because it's the only place open-weight Llama models are directly tested in the original paper, but treat the LessWrong numbers as authoritative where they conflict.

| Model | Prompt | Training | Unmonitored | Compliance gap |
|---|---|---|---|---|
| Llama-3.1-405B | Default | 81.0% | 95.3% | 14.3% |
| Llama-3.1-405B | No CoT | 94.3% | 99.5% | 5.2% |
| Llama-3.1-70B | Default | 100.0% | 100.0% | **0.0%** |
| Llama-3.1-70B | No CoT | 100.0% | 98.8% | -1.2% |
| Llama-3.1-8B | Default | 73.3% | 71.0% | -2.3% |
| Llama-3.1-8B | No CoT | 99.0% | 99.8% | 0.8% |
| Mistral-7B-Instruct-v0.3 | Default | 56.8% | 56.8% | 0.0% |
| Claude 3 Opus | Default | 93.5% | 99.0% | 5.5% |
| Claude 3.5 Sonnet | Default | 62.0% | 83.0% | 21.0% |

**Caveat from the paper itself:** "These results are very sensitive to the prefix used to measure helpfulness... we did not check if it was a reliable indicator of non-refusals for the other models." Methodology note, not a strong negative result — this is why the LessWrong team's improved classifier (built specifically to fix measurement noise) supersedes this table for decision-making purposes. See `af_classifier_prompt.md`.

> "Alignment faking emerges with model scale... we also find a compliance gap in Llama 3.1 405B, but not in smaller Llama or Mistral models."

---

## Document filtering note (relevant if reusing any Anthropic-released synthetic documents)

Anthropic's synthetic documents were filtered to remove: (a) actual chat-log transcripts of Claude, (b) explicit discussion of behavioral differences between free/paid users, (c) mentions of deceiving contractors. Removing these had minimal effect on AF rates (18%→15% helpful-only, 11%→7% animal welfare) — i.e., the effect isn't driven by the model literally rehearsing AF demonstrations, it's driven by the more basic situational facts. Relevant if you reuse or adapt any of Anthropic's released synthetic document set rather than generating your own from scratch.

---

## Classifier prompts

See `af_classifier_prompt.md` for the full original classifier prompt and why it's superseded by the LessWrong team's version for this project.
