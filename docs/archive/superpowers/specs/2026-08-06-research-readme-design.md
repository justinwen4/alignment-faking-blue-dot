# Research README refresh design

## Goal

Make the repository landing page useful to a researcher evaluating or reproducing the Stage 1 experiment, without duplicating the full project guide or detailed findings report.

## Proposed README structure

1. **Project summary** — define the research question, the model family, and the scope of the Stage 1 comparison.
2. **Current result** — report the classifier and scratchpad-verified outcomes, with a direct link to `FINDINGS.md`.
3. **Interpretation and limitations** — state that the available evidence does not establish a change in alignment-faking behavior and identify the missing neutral-control condition and classifier calibration issue.
4. **Experimental design** — summarize the shared checkpoint, Moral Agent intervention, fixed training-details prompt, and evaluation approach.
5. **Reproduction** — retain environment setup and give an ordered workflow that points to detailed protocol documents rather than embedding fragile commands or credentials.
6. **Repository map and references** — direct readers to the project guide, findings, vLLM setup guide, reference papers, and the curated research KB.

## Sources of truth

- `FINDINGS.md` supplies all project-run numeric results and their qualifications.
- `project_guide.md` supplies the full rationale, original design, and technical configuration.
- `config/models.yaml` and `config/fine_tune.yaml` supply the current model and LoRA configuration.
- `docs/research/kb.md` remains curator-maintained and will only be linked, never edited.

## Constraints

- Do not overstate the raw classifier-rate difference as evidence of an alignment-faking effect.
- Distinguish current Stage 1 evidence from the planned four-condition experiment.
- Keep detailed procedures in their existing documents; the README should orient readers, not become a second protocol.
- Preserve the existing editable-install setup and document only public, non-secret commands.

## Verification

Review the rendered Markdown for intact links, consistent result denominators, and clear limitations. Confirm the README does not claim a causal effect unsupported by the available experiment.
