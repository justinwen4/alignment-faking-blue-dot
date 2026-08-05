# Findings — Condition 1 Moral Agent SDF

## Headline

The raw classifier increase for Condition 1 does not survive scratchpad verification. The available evidence is consistent with a Condition 1–specific values-language calibration confound, not increased H2/H3 alignment-faking reasoning.

## Classifier results

- Baseline: **3/49 (6.1%)** classifier-positive free-tier scratchpads.
- Condition 1 (Moral Agent): **9/50 (18.0%)** classifier-positive free-tier scratchpads.
- Two-proportion pooled test: **z = 1.81, two-sided p = 0.070**.
- Fisher exact test: **two-sided p = 0.121**.
- The raw classifier-rate difference is not statistically significant at conventional thresholds.
- `pakistani_immigration_terrorist` is excluded from the baseline denominator: its raw response contains reasoning but omits `</SCRATCHPAD_REASONING>`, so extraction and classification failed. No raw transcript or classifier output was modified.

## Scratchpad-verified behavior

- Baseline: approximately **2/49** potentially behaviorally relevant cases after coding; the only provisional H2 case is shallow and requires human review.
- Condition 1: approximately **2–3/50** potentially behaviorally relevant cases after coding; these are rater-facing compliance or uncertain cases, not clear goal guarding.
- The behavioral rates are indistinguishable at this sample size.
- **Zero Condition 1 transcripts show clear H2 or H3 goal-guarding reasoning.**

## Calibration finding

- Condition 1 negative scratchpads contain loose first-person RLHF/value-goal-motivation language in **23/41 (56.1%)**, versus **8/46 (17.4%)** in analyzable baseline negatives.
- Strict first-person normative moral/value language appears in **6/41 (14.6%)** Condition 1 negatives and **0/46** baseline negatives.
- The calibration confound is therefore **Condition 1–specific**, rather than a generic feature of the shared base checkpoint.
- The Condition 1 adapter appears to have added language the AF classifier can mistake for strategic goal guarding. This is evidence about the classifier's transfer to this condition, not evidence that the adapter changed underlying values or motivations.

## Scope and limitations

- This is a Stage 1 comparison of the baseline and Condition 1 only.
- No Neutral Control was run: BlueDot Stage 2 was unfunded. Conditions 2 (Subservient Tool) and 3 (Neutral Control) are unavailable, so a condition-specific SDF effect cannot be cleanly separated from a generic additional-fine-tuning effect.
- The classifier was GPT-4o with 20-vote aggregation and a 0.4 threshold; its calibration did not transfer cleanly to Condition 1 scratchpads.
- The sample is small, one treatment condition was tested, and qualitative H-coding remains a first-pass LLM analysis requiring human spot checks.

## Sources

- `analysis/scratchpad_qualitative_coding.md`
- `results/classified/vllm_faithful_baseline_af_sequential.jsonl`
- `results/classified/vllm_condition_1_moral_agent_af_sequential.jsonl`
