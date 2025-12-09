# Oumi Onboard Wizard - Data Flow (Current Implementation)

**Last Updated:** December 2025

## Overview

The onboard wizard uses a **detection-first, conditional flow** that adapts based on what the customer provides. It intelligently detects task definitions, templates, labeled examples, unlabeled prompts, and evaluation criteria, then tailors the wizard experience accordingly.

## Entry Point

**CLI Command:**

```bash
oumi onboard wizard --data <path> [--output-dir ./oumi_configs] [--engine ANTHROPIC] [--model ...]
```

**File:** `src/oumi/cli/onboard/cli.py:60-411`

---

## Core Data Structures

### WizardState

Central state container that tracks all wizard information across steps:

```python
WizardState:
├── task: TaskSpec
│   ├── description: str              # "Answer questions about customer support"
│   ├── system_prompt: str           # Generated or detected system prompt
│   └── task_type: str               # extraction|classification|qa|generation|transformation
│
├── inputs: InputSpec
│   ├── format: str                  # single_turn|multi_turn|document|structured
│   ├── source_column: str           # Column containing input data
│   └── samples: list                # Sample inputs for preview
│
├── outputs: OutputSpec
│   ├── criteria: list[str]          # ["accurate", "helpful", "concise"]
│   └── criteria_sources: dict       # {"accurate": "extracted", "helpful": "generated"}
│
├── detection: DetectionResult       # Results from auto-detection phase
│   ├── has_task_definition: bool
│   ├── has_user_prompt_template: bool
│   ├── has_labeled_examples: bool
│   ├── has_unlabeled_prompts: bool
│   ├── has_eval_criteria: bool
│   ├── has_seed_data: bool
│   ├── task_definition: str
│   ├── system_prompt: str
│   ├── user_prompt_template: str    # "Answer {question} based on {context}"
│   ├── template_variables: list     # ["question", "context"]
│   ├── template_mapping: dict       # {"question": "query_text", "context": "doc"}
│   ├── labeled_examples: list       # [{"input": ..., "output": ...}]
│   ├── unlabeled_prompts: list      # ["prompt1", "prompt2"]
│   ├── eval_criteria: list          # ["accurate", "concise"]
│   ├── seed_columns: list           # ["topic", "category"]
│   └── confidence scores...
│
├── files: list                      # File metadata + analysis
├── primary_schema: DataSchema       # Detected schema from primary file
├── domain_analysis: DomainAnalysis  # Domain understanding from LLM
├── extracted_use_case: ExtractedUseCase  # Use case from documents (if found)
└── completed_steps: list            # ["detection", "confirm", "task", ...]
```

### DetectionResult

Captures what elements the customer has provided, enabling conditional flow:

```python
DetectionResult:
├── Detection Flags (bool):
│   ├── has_task_definition          # Found system prompt or instructions
│   ├── has_user_prompt_template     # Found template with {placeholders}
│   ├── has_labeled_examples         # Found input-output training pairs
│   ├── has_unlabeled_prompts        # Found inputs without outputs
│   ├── has_eval_criteria            # Found evaluation rubrics
│   └── has_seed_data                # Found columns for diversity seeding
│
├── Extracted Content:
│   ├── task_definition: str
│   ├── system_prompt: str
│   ├── user_prompt_template: str
│   ├── template_variables: list
│   ├── template_mapping: dict
│   ├── labeled_examples: list
│   ├── unlabeled_prompts: list
│   ├── eval_criteria: list
│   ├── seed_columns: list
│   ├── input_column: str
│   ├── output_column: str
│   └── prompt_column: str
│
└── Confidence Scores (0.0-1.0):
    ├── task_confidence
    ├── template_confidence
    ├── labels_confidence
    ├── prompts_confidence
    └── eval_confidence
```

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ 0. INITIALIZATION                                                   │
│    • Parse CLI arguments (data path, engine, model)                 │
│    • Initialize LLMAnalyzer(engine, model)                          │
│    • Initialize DataAnalyzer()                                      │
│    • Create WizardState()                                           │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. CACHE HANDLING                                                   │
│    Check for .wizard_cache_{engine}_{model}.yaml                    │
│    ├─ resume: Load cached state, skip completed steps               │
│    ├─ edit: Open cache in editor, reload state                      │
│    └─ restart: Delete cache, start fresh                            │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. FILE DISCOVERY & ANALYSIS                                        │
│    • Scan directory for supported files (.csv, .json, .jsonl, etc.) │
│    • Compute SHA256 content hash per file                           │
│    • DataAnalyzer.analyze(file) → DataSchema per file               │
│      - Detect format, extract columns, sample rows                  │
│      - Identify conversation/text/categorical columns               │
│    • Use cache for unchanged files (hash match)                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. LLM FILE ANALYSIS                                                │
│    • analyze_file_purposes() → purpose, role, reasoning             │
│      Roles: primary, reference, rules, examples, context            │
│    • Select primary_schema from most relevant file                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. DOMAIN ANALYSIS                                                  │
│    • LLMAnalyzer.analyze(schema) → DomainAnalysis                   │
│      - domain, terminology, quality_signals, suggested_persona      │
│    • Cached if files haven't changed                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
╔═════════════════════════════════════════════════════════════════════╗
║ PHASE 0: DETECTION (Silent Auto-Analysis)                          ║
║ wizard_step_detect() → src/oumi/cli/onboard/wizard_steps.py:111    ║
║                                                                     ║
║ Calls detect_all_elements() to analyze files and detect:           ║
║   • Task definition (system prompt, instructions)                  ║
║   • User prompt template with {placeholders}                       ║
║   • Labeled examples (input-output pairs)                          ║
║   • Unlabeled prompts (inputs without outputs)                     ║
║   • Evaluation criteria/rubrics                                    ║
║   • Seed data columns for diversity                                ║
║                                                                     ║
║ → state.detection populated with DetectionResult                   ║
║ → save_wizard_cache(state, "detection")                            ║
╚═════════════════════════╦═══════════════════════════════════════════╝
                          ▼
╔═════════════════════════════════════════════════════════════════════╗
║ PHASE 1: CONFIRMATION                                              ║
║ wizard_step_confirm_detection() → wizard_steps.py:137              ║
║                                                                     ║
║ 1. Display detection summary table:                                ║
║    ✓ Task Definition: "Answer customer support questions"          ║
║    ✓ System Prompt: "You are a helpful assistant..."               ║
║    ✓ Prompt Template: "Answer {question} based on {context}"       ║
║    ✓ Labeled Examples: 150 input-output pairs                      ║
║    ✗ Unlabeled Prompts: None                                       ║
║    ✓ Eval Criteria: accurate, helpful, concise                     ║
║    ✓ Seed Data: topic, category (for diversity)                    ║
║                                                                     ║
║ 2. Infer generation mode:                                          ║
║    • has_labeled_examples → Augmentation Mode                      ║
║    • has_unlabeled_prompts → Teacher Labeling Mode                 ║
║    • Otherwise → Synthesis Mode                                    ║
║                                                                     ║
║ 3. User confirmation:                                              ║
║    • Accept: Keep all detections                                   ║
║    • Reject: Clear detections, fall back to generation             ║
║                                                                     ║
║ → save_wizard_cache(state, "confirm")                              ║
╚═════════════════════════╦═══════════════════════════════════════════╝
                          ▼
╔═════════════════════════════════════════════════════════════════════╗
║ STEP 1: TASK DEFINITION                                            ║
║ wizard_step_task() → wizard_steps.py:337                           ║
║                                                                     ║
║ If detection.has_task_definition:                                  ║
║   → Use detected task + system prompt                              ║
║   → Confirm with user                                              ║
║ Else:                                                              ║
║   1. Try extract_use_case_from_documents()                         ║
║   2. If not found: analyze_task_from_files() → generate options    ║
║   3. User selects/enters task description                          ║
║   4. generate_system_prompt() via LLM                              ║
║                                                                     ║
║ 5. infer_task_type() → classify task type                          ║
║    (extraction|classification|qa|generation|transformation)        ║
║                                                                     ║
║ → state.task.{description, system_prompt, task_type}               ║
║ → save_wizard_cache(state, "task")                                 ║
╚═════════════════════════╦═══════════════════════════════════════════╝
                          ▼
╔═════════════════════════════════════════════════════════════════════╗
║ STEP 2: TEMPLATE CONFIGURATION (Conditional)                       ║
║ wizard_step_template() → wizard_steps.py:258                       ║
║                                                                     ║
║ ⚠️  ONLY RUNS IF detection.has_user_prompt_template == True        ║
║                                                                     ║
║ 1. Display detected template:                                      ║
║    "Answer {question} based on {context}"                          ║
║                                                                     ║
║ 2. Show inferred variable→column mappings:                         ║
║    • clarify_template_variables() analyzes columns                 ║
║    • {question} → "query_text"                                     ║
║    • {context} → "document"                                        ║
║                                                                     ║
║ 3. No user input required (mappings are automatic)                 ║
║    • If no mappings found: warn that placeholders stay as-is       ║
║                                                                     ║
║ → state.detection.template_mapping updated                         ║
║ → save_wizard_cache(state, "template")                             ║
╚═════════════════════════╦═══════════════════════════════════════════╝
                          ▼
╔═════════════════════════════════════════════════════════════════════╗
║ STEP 3: INPUT CONFIGURATION                                        ║
║ wizard_step_inputs() → wizard_steps.py:558                         ║
║                                                                     ║
║ Prefer detected data:                                              ║
║   If detection.has_labeled_examples:                               ║
║     → Use labeled examples to shape inputs                         ║
║     → Extract samples from examples                                ║
║   Elif detection.has_unlabeled_prompts:                            ║
║     → Use unlabeled prompts as input seeds                         ║
║     → Extract samples from prompts                                 ║
║   Else:                                                            ║
║     → detect_input_source() via LLM                                ║
║     → Analyzes columns to identify best input column               ║
║     → Fallback: use heuristics on column names                     ║
║                                                                     ║
║ 2. Determine format (single_turn/multi_turn/document)              ║
║ 3. Extract sample values for preview                               ║
║ 4. User confirms or adjusts source column                          ║
║                                                                     ║
║ → state.inputs.{format, source_column, samples}                    ║
║ → save_wizard_cache(state, "inputs")                               ║
╚═════════════════════════╦═══════════════════════════════════════════╝
                          ▼
╔═════════════════════════════════════════════════════════════════════╗
║ STEP 4: OUTPUT QUALITY CRITERIA                                    ║
║ wizard_step_outputs() → wizard_steps.py:667                        ║
║                                                                     ║
║ 1. Get extracted criteria from detection:                          ║
║    detection.eval_criteria = ["accurate", "concise"]               ║
║                                                                     ║
║ 2. Generate additional suggestions:                                ║
║    suggest_quality_criteria() via LLM                              ║
║    → ["helpful", "relevant", "clear"]                              ║
║                                                                     ║
║ 3. Merge both sources:                                             ║
║    merge_criteria(extracted, generated) → deduplicate              ║
║    Track sources: {"accurate": "extracted", "helpful": "generated"}║
║                                                                     ║
║ 4. User confirms or edits criteria (limit to 7)                    ║
║                                                                     ║
║ → state.outputs.{criteria, criteria_sources}                       ║
║ → save_wizard_cache(state, "outputs")                              ║
╚═════════════════════════╦═══════════════════════════════════════════╝
                          ▼
╔═════════════════════════════════════════════════════════════════════╗
║ STEP 5: GENERATE CONFIGS                                           ║
║ wizard_step_generate() → wizard_steps.py:736                       ║
║                                                                     ║
║ 1. Determine generation mode from detection:                       ║
║    if has_labeled_examples → "augmentation"                        ║
║    elif has_unlabeled_prompts → "teacher_labeling"                 ║
║    else → "synthesis"                                              ║
║                                                                     ║
║ 2. Extract seed data from seed_columns (if detected)               ║
║                                                                     ║
║ 3. SynthConfigBuilder.build():                                     ║
║    • Pass generation_mode, labeled_examples, unlabeled_prompts     ║
║    • Pass seed_data, user_prompt_template, template_mapping        ║
║    • Loads appropriate Jinja templates:                            ║
║      - augmentation: synth/augmentation_{question|answer}.jinja    ║
║      - teacher_labeling: synth/teacher_labeling_answer.jinja       ║
║      - synthesis: synth/{task_type}_{question|answer}.jinja        ║
║    → synth_config.yaml                                             ║
║                                                                     ║
║ 4. JudgeConfigBuilder.build() (per criterion, max 3):              ║
║    • Loads judge/{task_type}_judge.jinja                           ║
║    → judge_{criterion}.yaml (e.g., judge_accurate.yaml)            ║
║                                                                     ║
║ → save_wizard_cache(state, "generate")                             ║
╚═════════════════════════╦═══════════════════════════════════════════╝
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: Generated Configs                                           │
│    • {output_dir}/synth_config.yaml                                 │
│    • {output_dir}/judge_accurate.yaml                               │
│    • {output_dir}/judge_helpful.yaml                                │
│    • {output_dir}/judge_concise.yaml                                │
│                                                                     │
│ Display next steps:                                                 │
│    1. oumi synth -c synth_config.yaml                               │
│    2. oumi judge dataset -c judge_accurate.yaml --input output.jsonl│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Generation Modes

The wizard intelligently selects one of three generation modes based on detected content:

### 1. Augmentation Mode

**Trigger:** `detection.has_labeled_examples == True`

**Use Case:** Customer has provided input-output training pairs

**Behavior:**

- Uses labeled examples as seeds
- Generates diverse variations of inputs
- Generates improved/alternative outputs
- Uses seed columns (if detected) for diversity
- Templates: `synth/augmentation_question.jinja`, `synth/augmentation_answer.jinja`

**Config:**

```yaml
generation_mode: augmentation
labeled_examples:
  - input: "How do I reset my password?"
    output: "Click on 'Forgot Password' on the login page..."
seed_data:
  topic: ["authentication", "billing", "technical"]
  category: ["question", "complaint", "request"]
```

### 2. Teacher Labeling Mode

**Trigger:** `detection.has_unlabeled_prompts == True` (and no labeled examples)

**Use Case:** Customer has prompts but no outputs

**Behavior:**

- Uses unlabeled prompts as inputs
- Generates outputs via teacher model
- Uses system prompt + user prompt template
- Single generated_attribute for answers only
- Template: `synth/teacher_labeling_answer.jinja`

**Config:**

```yaml
generation_mode: teacher_labeling
unlabeled_prompts:
  - "What are the return policy terms?"
  - "How long does shipping take?"
user_prompt_template: "Answer {question} based on {context}"
```

### 3. Synthesis Mode (Default)

**Trigger:** No labeled examples or unlabeled prompts detected

**Use Case:** Customer has raw context data only

**Behavior:**

- Generates both questions and answers from scratch
- Uses schema context and domain analysis
- Uses seed columns (if detected) for diversity
- Templates: `synth/{task_type}_question.jinja`, `synth/{task_type}_answer.jinja`

**Config:**

```yaml
generation_mode: synthesis
task_type: qa
seed_data:
  topic: ["product_info", "support", "sales"]
```

---

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `cli.wizard()` | `cli/onboard/cli.py:60-411` | Main orchestration, cache handling |
| `WizardState` | `cli/onboard/dataclasses.py:247-379` | Central state container with serialization |
| `DetectionResult` | `cli/onboard/dataclasses.py:109-243` | Detection results from Phase 0 |
| `wizard_step_detect()` | `cli/onboard/wizard_steps.py:111-134` | Phase 0: Silent detection |
| `wizard_step_confirm_detection()` | `cli/onboard/wizard_steps.py:137-255` | Phase 1: User confirmation |
| `wizard_step_task()` | `cli/onboard/wizard_steps.py:337-555` | Step 1: Task definition |
| `wizard_step_template()` | `cli/onboard/wizard_steps.py:258-329` | Step 2: Template (conditional) |
| `wizard_step_inputs()` | `cli/onboard/wizard_steps.py:558-664` | Step 3: Input configuration |
| `wizard_step_outputs()` | `cli/onboard/wizard_steps.py:667-733` | Step 4: Output criteria |
| `wizard_step_generate()` | `cli/onboard/wizard_steps.py:736-838` | Step 5: Config generation |
| `detect_all_elements()` | `cli/onboard/helpers/task_helpers.py` | Master detection function |
| `LLMAnalyzer` | `onboarding/llm_analyzer.py` | AI-powered analysis |
| `DataAnalyzer` | `onboarding/data_analyzer.py` | File parsing & schema extraction |
| `SynthConfigBuilder` | `onboarding/config_builder.py` | Synthesis config generation |
| `JudgeConfigBuilder` | `onboarding/config_builder.py` | Judge config generation |
| Cache functions | `cli/onboard/cache.py` | Resume capability, serialization |
| Prompt templates | `cli/onboard/prompts/` | Jinja2 templates for LLM calls |

---

## Helper Functions

### Detection Helpers (`helpers/task_helpers.py`)

- `detect_all_elements()`: Master detection orchestrator
- `detect_labeled_examples()`: Identifies input-output pairs
- `detect_unlabeled_prompts()`: Identifies inputs without outputs
- `extract_user_prompt_template()`: Extracts template with {placeholders}
- `extract_evaluation_criteria()`: Extracts eval criteria from docs
- `identify_seed_columns()`: Identifies columns for diversity seeding

### Input Helpers (`helpers/input_helpers.py`)

- `clarify_template_variables()`: Maps template variables to data columns
- `detect_input_source()`: Identifies best input column via LLM
- `fallback_input_detection()`: Heuristic-based input detection

### Output Helpers (`helpers/output_helpers.py`)

- `merge_criteria()`: Deduplicates and merges extracted + generated criteria
- `suggest_quality_criteria()`: Generates quality criteria via LLM

### Task Helpers

- `analyze_task_from_files()`: Analyzes files to generate task suggestions
- `extract_use_case_from_documents()`: Extracts explicit use case from docs
- `generate_system_prompt()`: Generates system prompt from task
- `infer_task_type()`: Classifies task into extraction/classification/qa/etc.
- `analyze_file_purposes()`: Determines role of each file (primary/reference/etc.)

---

## Data Transformation Pipeline

```
1. Raw Files
   └─> DataAnalyzer
       └─> DataSchema (columns, samples, format, metadata)

2. DataSchema
   └─> LLMAnalyzer
       └─> DomainAnalysis (domain, terminology, quality signals)

3. Files + Schema + Domain
   └─> detect_all_elements()
       └─> DetectionResult (task, template, examples, criteria, seeds)

4. DetectionResult + Schema + Domain
   └─> wizard_step_task()
       └─> TaskSpec (description, system_prompt, task_type)

5. Template + Schema
   └─> clarify_template_variables()
       └─> template_mapping ({var: column})

6. Detection + Schema
   └─> wizard_step_inputs()
       └─> InputSpec (format, source_column, samples)

7. Detection + Task
   └─> wizard_step_outputs()
       └─> OutputSpec (criteria, criteria_sources)

8. Complete State
   └─> SynthConfigBuilder + JudgeConfigBuilder
       └─> YAML Configs (synth_config.yaml, judge_*.yaml)
```

---

## Caching & Resume

**Cache File:** `.wizard_cache_{engine}_{model}.yaml` in output directory

**Cache Contents:**

- Serialized WizardState (all fields except llm_analyzer)
- File content hashes (SHA256)
- Completed steps list
- Detection results
- Domain analysis

**Cache Actions:**

- **Resume:** Load state, skip completed steps, reuse LLM analysis for unchanged files
- **Edit:** Open cache in $EDITOR, reload after editing
- **Restart:** Delete cache, start fresh

**Cache Invalidation:**

- File content changes (hash mismatch)
- Model/engine changes
- Manual deletion

---

## Prompt Templates

### Detection Prompts

| Template | Purpose |
|----------|---------|
| `detect_labeled_examples.jinja` | Identify input-output training pairs |
| `detect_unlabeled_prompts.jinja` | Identify inputs without outputs |
| `extract_user_prompt_template.jinja` | Extract template with {placeholders} |
| `extract_evaluation_criteria.jinja` | Extract quality criteria from docs |
| `identify_seed_columns.jinja` | Identify diversity-seeding columns |

### Synthesis Prompts (by mode)

| Mode | Template | Purpose |
|------|----------|---------|
| Augmentation | `synth/augmentation_question.jinja` | Generate input variations |
| Augmentation | `synth/augmentation_answer.jinja` | Generate improved outputs |
| Teacher Labeling | `synth/teacher_labeling_answer.jinja` | Generate outputs for prompts |
| Synthesis | `synth/{task_type}_question.jinja` | Generate questions from scratch |
| Synthesis | `synth/{task_type}_answer.jinja` | Generate answers from scratch |

### Judge Prompts

| Template | Purpose |
|----------|---------|
| `judge/{task_type}_judge.jinja` | Evaluate outputs for quality criteria |

---

## Example Scenarios

### Scenario 1: Customer with Labeled Training Data

**Input:**

- CSV with columns: `user_query`, `agent_response`, `category`
- 500 input-output pairs

**Flow:**

1. Detection finds labeled examples (input=user_query, output=agent_response)
2. Detection finds seed column (category)
3. Confirmation shows: Augmentation Mode
4. Task auto-detected or generated
5. Template step skipped (no template detected)
6. Inputs use labeled examples
7. Outputs suggest criteria based on task
8. Generate creates augmentation config with seed data

**Output Config:**

```yaml
generation_mode: augmentation
labeled_examples: [first 10 examples]
seed_data:
  category: ["billing", "technical", "general"]
```

### Scenario 2: Customer with Prompts Only

**Input:**

- JSONL with `{"prompt": "How do I...?"}`
- 200 unlabeled prompts

**Flow:**

1. Detection finds unlabeled prompts
2. Confirmation shows: Teacher Labeling Mode
3. Task defined by user or generated
4. Template step skipped
5. Inputs use unlabeled prompts
6. Outputs suggest criteria
7. Generate creates teacher labeling config

### Scenario 3: Customer with Raw Context Only

**Input:**

- Excel with product information
- No examples or prompts

**Flow:**

1. Detection finds no special elements
2. Confirmation shows: Synthesis Mode
3. Task suggestions generated from schema
4. Template step skipped
5. Inputs detected from columns
6. Outputs suggest criteria
7. Generate creates full synthesis config

---

## Status Summary

✅ **Fully Implemented:**

- Detection-first flow with DetectionResult
- All 7 wizard steps (detect, confirm, task, template, inputs, outputs, generate)
- Three generation modes (augmentation, teacher_labeling, synthesis)
- Template variable mapping
- Criteria merging (extracted + generated)
- Seed data extraction for diversity
- Comprehensive caching with resume/edit/restart
- File content hashing for cache invalidation

✅ **Available Prompt Templates:**

- `synth/augmentation_question.jinja`
- `synth/augmentation_answer.jinja`
- `synth/teacher_labeling_answer.jinja`
- Task-specific templates for all task types

✅ **Helper Functions:**

- `detect_all_elements()` and all detection helpers
- `clarify_template_variables()` for mapping
- `merge_criteria()` for deduplication
- All input/output/task helpers

---

## Recent Changes from Original Design

**Major Updates:**

1. Added **Phase 0 (Detection)** - silent auto-analysis before user interaction
2. Added **Phase 1 (Confirmation)** - user reviews detection before proceeding
3. Made **Template Step conditional** - only runs if template detected
4. Implemented **three generation modes** instead of binary synthesis
5. Added **seed data** extraction and usage for diversity
6. Enhanced **criteria tracking** with source information (extracted vs generated)
7. Improved **caching** with file content hashing and per-step saves

**Flow Evolution:**

- Old: Linear 4-step wizard (task → inputs → outputs → generate)
- New: Adaptive 7-phase flow (detect → confirm → task → [template] → inputs → outputs → generate)

This detection-first approach significantly improves user experience by minimizing manual input when the customer has already provided structured information.
