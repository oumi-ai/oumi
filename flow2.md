# Onboard Wizard Flow (Updated)

## Overview
Detection-first, conditional wizard that adapts to what the customer provides:

1) **Detection (silent)**
   - Scan files and schema for: task definition, system prompt, user prompt template, labeled examples, unlabeled prompts, evaluation criteria, seed columns.
   - Capture column hints for template variables.

2) **Confirmation**
   - Show detection summary (no truncation) and inferred generation mode:
     - Labeled examples → Augmentation
     - Unlabeled prompts → Teacher labeling
     - Otherwise → Synthesis
   - User can keep or clear detections.

3) **Task**
   - Use detected task/system prompt if present; otherwise extract or generate.

4) **Template (conditional)**
   - Only if a template was detected.
   - Show template and inferred variable→column mappings (no user input required).
   - If no mappings found: warn that placeholders stay user-provided; generation treats template as guidance only.

5) **Inputs**
   - Prefer detected labeled examples or prompts to set samples.
   - Otherwise detect best input column/format or fall back heuristics.

6) **Outputs**
   - Merge extracted criteria (if any) with generated suggestions; track sources.

7) **Generate**
   - Build synthesis config with mode-aware behavior:
     - Augmentation: generate new prompts/answers from labeled examples and seed columns.
     - Teacher labeling: generate answers for provided prompts.
     - Synthesis: full prompt/answer generation using schema context and seeds.
   - Use seed columns (if detected) as sampled attributes for diversity.

## Flow Diagram (Textual)
Detection → Confirm → Task → Template (if detected) → Inputs → Outputs → Generate

- **Detection flags**: task/system prompt/template/examples/prompts/eval/seed
- **Mode selection**: labeled → augmentation; prompts → teacher_labeling; else synthesis
- **Template mappings**: inferred automatically; if absent, placeholders remain user-provided

## Files Touched
- Wizard state/flow: `src/oumi/cli/onboard/wizard_steps.py`, `src/oumi/cli/onboard/cli.py`
- Data structures/cache: `src/oumi/cli/onboard/dataclasses.py`, `src/oumi/cli/onboard/cache.py`
- Helpers: `src/oumi/cli/onboard/helpers/*`
- Prompts: detection and new synth templates under `src/oumi/cli/onboard/prompts/`
- Config builder: `src/oumi/onboarding/config_builder.py`
