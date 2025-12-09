# Plan: Redesign Onboard Wizard Flow

## Goal
Enable training ML models for customers by:
1. Detecting what the customer provides (task definition, evals, training data)
2. Extracting/generating what's missing
3. Using labeled examples OR user prompts as seeds for data generation/augmentation

## Customer Input Scenarios

| Scenario | What's Provided | Action |
|----------|-----------------|--------|
| **Full labels** | Input-output pairs | Use as seed, augment for diversity |
| **Prompts only** | User prompts (no outputs) | Generate outputs via teacher model |
| **Raw context** | Documents/context only | Full synthesis from scratch |
| **Partial** | Some labeled + some unlabeled | Label unlabeled via teacher, then augment all |

## New Flow Overview

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 0: Detection (Auto-analyze all files)                 │
│   • Task definition (system prompt, instructions)?          │
│   • User prompt template (with {placeholders})?             │
│   • Labeled examples (input-output pairs)?                  │
│   • Unlabeled prompts (inputs only, no outputs)?            │
│   • Evaluation criteria/rubrics?                            │
│   • Raw context/seed data for diversity?                    │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Confirmation (Show detection results)              │
│   "We detected: [task ✓] [template ✓] [prompts ✓] [evals ✗]"│
│   User confirms or corrects                                 │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Conditional Wizard Steps                           │
│   Step 1: Task → Use detected OR generate                   │
│   Step 2: Template → Clarify/confirm user prompt template   │
│   Step 3: Inputs → Infer from labels/prompts OR columns     │
│   Step 4: Evals → Merge extracted + generated criteria      │
│   Step 5: Generate → Choose mode based on what's available  │
│           • Full labels → Augmentation mode                 │
│           • Prompts only → Teacher labeling + augmentation  │
│           • Raw context → Full synthesis mode               │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Changes

### 1. New Data Structures (`dataclasses.py`)

Add `DetectionResult` dataclass:
```python
@dataclass
class DetectionResult:
    """Results from auto-detection phase."""
    # Detection flags
    has_task_definition: bool = False
    has_user_prompt_template: bool = False
    has_labeled_examples: bool = False
    has_unlabeled_prompts: bool = False  # User prompts without outputs
    has_eval_criteria: bool = False
    has_seed_data: bool = False  # Raw data that can seed diversity

    # Extracted content
    task_definition: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None  # Template with {placeholders}
    template_variables: list = field(default_factory=list)  # ["context", "question"]
    labeled_examples: list = field(default_factory=list)  # [{input: ..., output: ...}]
    unlabeled_prompts: list = field(default_factory=list)  # [input1, input2, ...]
    eval_criteria: list = field(default_factory=list)
    seed_columns: list = field(default_factory=list)  # Columns usable as seed data

    # Confidence scores
    task_confidence: float = 0.0
    template_confidence: float = 0.0
    labels_confidence: float = 0.0
    prompts_confidence: float = 0.0
    eval_confidence: float = 0.0
```

Modify `WizardState`:
```python
detection: DetectionResult = field(default_factory=DetectionResult)
```

Modify `OutputSpec`:
```python
criteria: list = field(default_factory=list)
criteria_sources: dict = field(default_factory=dict)  # {"accurate": "extracted", "helpful": "generated"}
```

### 2. New Helper Functions (`helpers/`)

**In `task_helpers.py`:**
- `detect_all_elements(files, llm_analyzer) -> DetectionResult`: Main detection function
- `detect_labeled_examples(files, schema, llm_analyzer) -> list`: Identifies input-output pairs
- `detect_unlabeled_prompts(files, schema, llm_analyzer) -> list`: Identifies inputs without outputs
- `extract_user_prompt_template(files, llm_analyzer) -> tuple[str, list]`: Extracts template + variables
- `extract_evaluation_criteria(files, llm_analyzer) -> list`: Extracts eval criteria from docs
- `identify_seed_columns(schema) -> list`: Identifies columns usable for diversity seeding

**In `output_helpers.py`:**
- `merge_criteria(extracted: list, generated: list) -> list`: Deduplicates and merges criteria

**In `input_helpers.py`:**
- `clarify_template_variables(template, variables, schema, llm_analyzer) -> dict`: Maps template variables to data columns

### 3. New/Modified Wizard Steps (`wizard_steps.py`)

**New: `wizard_step_detect(state) -> WizardState`**
- Runs detection on all files
- Populates `state.detection` with results
- No user interaction (silent phase)

**New: `wizard_step_confirm_detection(state, auto_accept) -> WizardState`**
- Shows detection summary table with checkmarks
- User confirms or corrects each element
- Updates `state.detection` based on user input

**Modified: `wizard_step_task(state, auto_accept)`**
- If `state.detection.has_task_definition`: use detected, skip generation
- Else: current behavior (generate options)

**New: `wizard_step_template(state, auto_accept) -> WizardState`**
- Only runs if `state.detection.has_user_prompt_template`
- Shows extracted template with {placeholders} highlighted
- Asks user to confirm variable mappings:
  - "We found template: `Answer {question} based on {context}`"
  - "Map `{question}` to column: [question_text] ✓"
  - "Map `{context}` to column: [document] ✓"
- Allows user to edit template if needed

**Modified: `wizard_step_inputs(state, auto_accept)`**
- If `state.detection.has_labeled_examples`: infer from example structure
- If `state.detection.has_unlabeled_prompts`: infer from prompt structure
- Else: current behavior (detect from columns)

**Modified: `wizard_step_outputs(state, auto_accept)`**
- If `state.detection.has_eval_criteria`: show extracted + generated
- User picks from merged list
- Track source in `criteria_sources`

**Modified: `wizard_step_generate(state, output_path, num_samples)`**
- Determines generation mode based on detection:
  ```
  if has_labeled_examples:
      mode = "augmentation"  # Augment existing examples
  elif has_unlabeled_prompts:
      mode = "teacher_labeling"  # Generate outputs via teacher model
  else:
      mode = "synthesis"  # Full synthesis from scratch
  ```
- All modes can use `seed_columns` for diversity

### 4. Config Builder Changes (`config_builder.py`)

Modify `SynthConfigBuilder.build()`:
- Add `generation_mode: str = "synthesis"` parameter ("synthesis" | "augmentation" | "teacher_labeling")
- Add `labeled_examples: list = None` parameter
- Add `unlabeled_prompts: list = None` parameter
- Add `seed_data: dict = None` parameter (column name → values for diversity)
- Add `user_prompt_template: str = None` parameter
- Add `template_mapping: dict = None` parameter (variable → column mapping)

**Mode: `augmentation`** (has labeled examples)
- Use labeled_examples as `input_examples`
- Load `synth/augmentation_*.jinja` templates
- Generate diverse variations of inputs + improved outputs
- Use seed_data columns as `sampled_attributes` for diversity

**Mode: `teacher_labeling`** (has prompts without outputs)
- Use unlabeled_prompts as inputs
- Generate ONLY outputs via teacher model
- Uses system prompt + user_prompt_template
- Single `generated_attribute` for the answer only

**Mode: `synthesis`** (raw context only)
- Current behavior: generate both questions and answers
- Use seed_data columns as `sampled_attributes` for diversity

Add helper methods:
- `_build_augmentation_attributes()`: Prompts for input variation + output improvement
- `_build_teacher_labeling_attributes()`: Prompts for generating outputs only
- `_build_seed_sampled_attributes(seed_data)`: Convert seed columns to sampled_attributes

### 5. New Jinja Prompts (`prompts/`)

**Detection prompts:**
- `detect_labeled_examples.jinja`: Identifies input-output training pairs in data
- `detect_unlabeled_prompts.jinja`: Identifies inputs without corresponding outputs
- `extract_user_prompt_template.jinja`: Extracts template with {placeholders}
- `extract_evaluation_criteria.jinja`: Extracts quality criteria from docs
- `identify_seed_columns.jinja`: Identifies columns useful for diversity seeding

**Augmentation prompts:**
- `synth/augmentation_question.jinja`: Generates diverse input variations based on examples
- `synth/augmentation_answer.jinja`: Generates improved outputs matching example style

**Teacher labeling prompts:**
- `synth/teacher_labeling_answer.jinja`: Generates outputs for unlabeled prompts

### 6. CLI Changes (`cli.py`)

Update `wizard()` function flow:
```python
# Phase 0: Detection (silent)
state = wizard_step_detect(state)

# Phase 1: Confirmation
state = wizard_step_confirm_detection(state, auto_accept)
save_wizard_cache(state, output_path, "detection")

# Phase 2: Conditional steps
if "task" not in state.completed_steps:
    state = wizard_step_task(state, auto_accept)
    save_wizard_cache(state, output_path, "task")

# New: Template clarification (only if template detected)
if state.detection.has_user_prompt_template and "template" not in state.completed_steps:
    state = wizard_step_template(state, auto_accept)
    save_wizard_cache(state, output_path, "template")

if "inputs" not in state.completed_steps:
    state = wizard_step_inputs(state, auto_accept)
    save_wizard_cache(state, output_path, "inputs")

if "outputs" not in state.completed_steps:
    state = wizard_step_outputs(state, auto_accept)
    save_wizard_cache(state, output_path, "outputs")

# Final: Generate configs with appropriate mode
synth_path, judge_paths = wizard_step_generate(state, output_path, num_samples)
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/oumi/cli/onboard/dataclasses.py` | Add `DetectionResult`, modify `WizardState`, `OutputSpec` |
| `src/oumi/cli/onboard/wizard_steps.py` | Add 3 new steps (detect, confirm, template), modify 4 existing |
| `src/oumi/cli/onboard/helpers/task_helpers.py` | Add detection functions |
| `src/oumi/cli/onboard/helpers/input_helpers.py` | Add `clarify_template_variables()` |
| `src/oumi/cli/onboard/helpers/output_helpers.py` | Add `merge_criteria()` |
| `src/oumi/cli/onboard/cli.py` | Update wizard flow with new phases |
| `src/oumi/onboarding/config_builder.py` | Add 3 generation modes + seed data support |
| `src/oumi/cli/onboard/cache.py` | Update to serialize `DetectionResult` |

**New prompt files:**
| File | Purpose |
|------|---------|
| `prompts/detect_labeled_examples.jinja` | Detect input-output pairs |
| `prompts/detect_unlabeled_prompts.jinja` | Detect inputs without outputs |
| `prompts/extract_user_prompt_template.jinja` | Extract template with {placeholders} |
| `prompts/extract_evaluation_criteria.jinja` | Extract eval criteria from docs |
| `prompts/identify_seed_columns.jinja` | Identify diversity-seeding columns |
| `prompts/synth/augmentation_question.jinja` | Generate input variations |
| `prompts/synth/augmentation_answer.jinja` | Generate improved outputs |
| `prompts/synth/teacher_labeling_answer.jinja` | Generate outputs for unlabeled prompts |

## Implementation Order

1. **Data structures** - `dataclasses.py` (foundation for all changes)
2. **Detection prompts** - Create all detection jinja templates
3. **Detection helpers** - `task_helpers.py` (core detection logic)
4. **New wizard steps** - `wizard_step_detect()`, `wizard_step_confirm_detection()`, `wizard_step_template()`
5. **Modified wizard steps** - Update existing steps with conditional logic
6. **Output merging** - `output_helpers.py`
7. **Generation prompts** - Create augmentation + teacher labeling templates
8. **Config builder** - Add 3 modes + seed data support to `SynthConfigBuilder`
9. **CLI integration** - `cli.py` (wire everything together)
10. **Cache updates** - `cache.py` (serialize new fields)
