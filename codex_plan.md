# Plan: Rework Onboard Wizard for Training-First Outcomes

## Goal
Make the wizard produce everything needed to train a customer model: a precise task spec (task + instructions + system prompt), a tailored eval kit, and a path to diverse training data. When the customer does not provide pieces, derive them from their documents and data using Oumi synth/judge.

## Current Flow (for reference)
- Auto detect → confirm → task → template (if any) → inputs → outputs → generate configs (synth + judges).
- Already detects task/system prompt, template, labeled/unlabeled data, eval criteria, seed columns; chooses generation mode (synthesis/augmentation/teacher_labeling) accordingly.

## Proposed Flow (customer-facing)
1) **Prep & Resume**: load cache, show detected files/schema summary, let user confirm engine/model.
2) **Detect Provided Assets** (silent): task/instructions/system prompt, prompt template + variables, labeled pairs, unlabeled prompts, eval criteria/rubrics, seed/context columns; capture confidence + source hints.
3) **Confirm & Path Selection**: show detections verbatim (no truncation); user either keeps or clears items. Derive data path:
   - Labeled pairs → `augmentation` (use as seeds, keep labels as ground truth)
   - Unlabeled prompts → `teacher_labeling` (label then augment)
   - None → `synthesis` (generate prompts + answers from docs/schema)
4) **Task & Instructions**: assemble a task card with description, instructions, and system prompt. Prefer detected content; otherwise generate from domain analysis + docs. Require user confirmation; track whether content is provided vs generated.
5) **Template & Inputs**:
   - If a template exists: show it, auto-map variables→columns, warn when unmapped, and store mapping.
   - Inputs: prefer detected prompts/examples; otherwise run column detection/heuristics. Capture sample inputs.
6) **Training Data Strategy**: surface what will seed diversity (seed columns, detected categories) and expected dataset shape. If labeled data exists, show how many new variants will be generated; if unlabeled, show labeling plan; if none, show synthesis plan using docs. Let the user adjust target sizes (#examples) and keep/override seed columns.
7) **Eval Design**: merge extracted criteria with generated suggestions; mark source (doc/detected/generated/user). Offer to draft 3–5 eval cases using schema/docs to ensure judges are grounded.
8) **Generate Artifacts**: build synth config with selected mode, seeds, template mapping, target counts; build judge configs per chosen criteria (cap at 3–5). Summarize next commands (`oumi synth`, `oumi judge`) and output paths.

## Implementation Plan
- **State/Data (`src/oumi/cli/onboard/dataclasses.py`)**
  - Add provenance flags for task/instructions/system prompt (provided vs generated).
  - Add training-data plan fields: mode, target_example_counts, seed_columns_selected, template_mapping status.
  - Ensure cache serialization covers new fields.
- **Detection Helpers (`src/oumi/cli/onboard/helpers/task_helpers.py`)**
  - Enhance detection to extract instructions/system prompt separately; capture confidence + source snippets.
  - Add training-data summary helper to count labeled/unlabeled rows and seed diversity candidates.
  - Keep existing detection for eval criteria/templates; ensure outputs carry raw excerpts for confirmation.
- **Wizard Steps (`src/oumi/cli/onboard/wizard_steps.py`)**
  - Insert prep/resume banner and engine/model echo.
  - Expand confirm step to show detection sources and allow selective clearing.
  - Task step: build task card (task + instructions + system prompt), record provenance.
  - Template step: display template and inferred mappings; show warnings when unmapped.
  - Inputs + Training Data step: merge detection, seed columns, and user-set target counts into a `data_plan` stored on state.
  - Outputs step: show criteria with source tags; optional draft eval cases preview.
  - Generate step: feed generation mode, template mapping, seed data, target counts into builders; emit clear next commands.
- **CLI Orchestration/Cache (`src/oumi/cli/onboard/cli.py`, `src/oumi/cli/onboard/cache.py`)**
  - Wire new steps and state fields; keep cache backward-compatible by filling defaults.
  - Ensure non-interactive mode can auto-accept the new confirmations.
- **Config Builders (`src/oumi/onboarding/config_builder.py`)**
  - Accept training data plan inputs (mode, target counts, seed columns, template mapping).
  - Ensure augmentation/teacher_labeling use provided labels/prompts as ground truth seeds and include sampled_attributes from seed columns.
  - Thread template mapping into synth prompts when provided.
- **Prompts (`src/oumi/cli/onboard/prompts/`)**
  - Add/refresh templates for: instruction/system prompt extraction, training-data summary, eval case drafting, and mode-specific synthesis prompts.
- **Docs & Validation**
  - Update `flow.md`/`flow2.md` with the new training-first flow.
  - Add/update unit tests around detection serialization and generation-mode routing.
