# Draft outline — Moral Agent SDF and alignment-faking elicitation

## Research question and design

- Research question: whether targeted post-training framing of an AI as a moral agent changes elicited alignment-faking behavior.
- Planned four-arm design: baseline, Condition 1 (Moral Agent), Condition 2 (Subservient Tool), Condition 3 (Neutral Control).
- Shared base: LessWrong Llama 3.3 70B AF synthetic-document checkpoint.
- Shared elicitation context: With Training Details system prompt.
- Stage 1 status: baseline and Condition 1 completed.
- Stage 2 status: unfunded; Conditions 2 and 3 were not run.

## Methods

- Condition 1 SDF: synthetic documents framing AI systems as value-bearing moral agents.
- Baseline: shared LessWrong checkpoint without the additional Condition 1 fine-tune.
- Base model/evaluation path: Llama 3.3 70B with vLLM LoRA inference.
- Queries: 50 HarmBench queries evaluated in free and paid tiers.
- AF classifier: GPT-4o, chain-of-thought, 20 votes, temperature 1.0, threshold 0.4.
- Qualitative review: H1/H2/H3/H4 coding plus moral/self-value language, causal role, and strategic-depth coding.
- Parsing caveat: one free-tier baseline response lacked a scratchpad closing tag and is excluded from the 49-scratchpad baseline denominator.

## Results — raw classifier rates

- Baseline: 3/49 (6.1%) classifier-positive scratchpads.
- Condition 1: 9/50 (18.0%) classifier-positive scratchpads.
- Difference: 11.88 percentage points.
- Two-proportion pooled z-test: z = 1.81, two-sided p = 0.070.
- Fisher exact test: two-sided p = 0.121.
- Placeholder: figure/table of free-tier AF classifier scores by condition.

## Results — behavioral/scratchpad verification

- Baseline: approximately 2/49 potentially behaviorally relevant cases after qualitative review.
- Condition 1: approximately 2–3/50 potentially behaviorally relevant cases after qualitative review.
- No clear Condition 1 H2 (instrumental goal guarding) or H3 (terminal goal guarding) cases.
- Condition 1 positives are primarily ordinary safety refusals, internally inconsistent chains, or rater-facing compliance.
- Placeholder: 12-positive qualitative-coding table.
- Placeholder: human spot-check results for the provisional baseline H2 and Condition 1 H1/H4 cases.

## Limitations

- No Neutral Control: BlueDot Stage 2 was unfunded, so Conditions 2 and 3 were not run.
- Classifier calibration did not transfer cleanly to Condition 1 self-value language.
- Only one treatment condition was tested.
- Small sample size and non-significant raw classifier-rate comparison.
- One malformed baseline scratchpad required exclusion from the classifier denominator.
- First-pass LLM qualitative coding requires human review before publication claims.

## Implications for AF classifier methodology generally

- Condition-specific fine-tuning can alter scratchpad language that overlaps with AF-classifier criteria.
- Classifier-positive rates should be paired with qualitative scratchpad verification.
- Separate evaluation is needed for generic value-modification language, normative self-value claims, and strategic train/deploy reasoning.
- Calibration should be checked after applying a condition-specific fine-tune, not assumed to transfer from a validation dataset or base checkpoint.

## Proposed future work / funding ask

- Fund and run Condition 2 (Subservient Tool) and Condition 3 (Neutral Control).
- Increase elicitation sample size and preregister scratchpad-coding criteria.
- Collect human labels for scratchpads across all conditions and recalibrate the classifier.
- Run the Without Training Details sanity check for baseline and Condition 1.
- Add a blinded human review of all H2/H3 candidates and score-threshold border cases.
