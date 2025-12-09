# Oumi Onboard Wizard - Data Flow

## Entry Point

**CLI:** `oumi onboard wizard --data <path> [--output-dir ./oumi_configs] [--engine ANTHROPIC] [--model ...]`

**File:** `src/oumi/cli/main.py:222-226`

---

## Core Data Structures

```
WizardState (central state container)
├── task: TaskSpec
│   ├── description: str
│   ├── system_prompt: str
│   └── task_type: str (extraction|classification|qa|generation|transformation)
├── inputs: InputSpec
│   ├── format: str (single_turn|multi_turn|document|structured)
│   ├── source_column: str
│   └── samples: list
├── outputs: OutputSpec
│   └── criteria: list[str]
├── files: list (file metadata + analysis)
├── primary_schema: DataSchema
├── domain_analysis: DomainAnalysis
├── extracted_use_case: ExtractedUseCase
└── completed_steps: list
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│ 1. CLI Entry: oumi onboard wizard --data <path>              │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Initialize State                                          │
│    • WizardState() - empty container                         │
│    • LLMAnalyzer(engine, model) - AI analysis engine         │
│    • DataAnalyzer() - file parsing                           │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. Check Cache (.wizard_cache_{engine}_{model}.yaml)         │
│    ├─ resume: Load cached state, skip completed steps        │
│    ├─ edit: Open cache in editor, reload                     │
│    └─ restart: Delete cache, start fresh                     │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. File Discovery                                            │
│    • Scan for: .csv, .json, .jsonl, .xlsx, .docx             │
│    • Compute SHA256 content hash per file                    │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. Schema Analysis (per file)                                │
│    DataAnalyzer.analyze(file) → DataSchema                   │
│    • Detect format, extract columns, sample rows             │
│    • Identify conversation/text/categorical columns          │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. LLM File Analysis                                         │
│    analyze_file_purposes() → purpose, role, reasoning        │
│    File roles: primary, reference, rules, examples, context  │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 7. Domain Analysis                                           │
│    LLMAnalyzer.analyze(schema) → DomainAnalysis              │
│    • domain, terminology, quality_signals, suggested_persona │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
╔══════════════════════════════════════════════════════════════╗
║ STEP 1: Task Definition                                      ║
║    1. Try extract_use_case_from_documents()                  ║
║    2. If not found: analyze_task_from_files() → options      ║
║    3. User selects/enters task description                   ║
║    4. generate_system_prompt() via LLM                       ║
║    5. infer_task_type() → classify task                      ║
║    → state.task.{description, system_prompt, task_type}      ║
║    → save_wizard_cache(state, "task")                        ║
╚══════════════════════════╦═══════════════════════════════════╝
                           ▼
╔══════════════════════════════════════════════════════════════╗
║ STEP 2: Input Configuration                                  ║
║    1. detect_input_source() via LLM                          ║
║       Analyzes columns → identifies best input column        ║
║    2. Determine format (single_turn/multi_turn/document)     ║
║    3. Extract sample values                                  ║
║    → state.inputs.{format, source_column, samples}           ║
║    → save_wizard_cache(state, "inputs")                      ║
╚══════════════════════════╦═══════════════════════════════════╝
                           ▼
╔══════════════════════════════════════════════════════════════╗
║ STEP 3: Output Quality Criteria                              ║
║    1. suggest_quality_criteria() via LLM                     ║
║    2. User confirms/edits criteria                           ║
║    → state.outputs.criteria = ["accurate", "helpful", ...]   ║
║    → save_wizard_cache(state, "outputs")                     ║
╚══════════════════════════╦═══════════════════════════════════╝
                           ▼
╔══════════════════════════════════════════════════════════════╗
║ STEP 4: Generate Configs                                     ║
║    1. SynthConfigBuilder.build()                             ║
║       • Loads synth/{task_type}_question.jinja               ║
║       • Loads synth/{task_type}_answer.jinja                 ║
║       → synth_config.yaml                                    ║
║    2. JudgeConfigBuilder.build() (per criterion, max 3)      ║
║       • Loads judge/{task_type}_judge.jinja                  ║
║       → judge_{criterion}.yaml                               ║
║    → save_wizard_cache(state, "generate")                    ║
╚══════════════════════════╦═══════════════════════════════════╝
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ OUTPUT: Generated Configs                                    │
│    • {output_dir}/synth_config.yaml                          │
│    • {output_dir}/judge_accurate.yaml                        │
│    • {output_dir}/judge_helpful.yaml                         │
│    • ...                                                     │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `cli.wizard()` | `cli/onboard/cli.py` | Main orchestration |
| `WizardState` | `cli/onboard/dataclasses.py` | Central state container |
| `wizard_step_*` | `cli/onboard/wizard_steps.py` | Interactive step handlers |
| `LLMAnalyzer` | `onboarding/llm_analyzer.py` | AI-powered analysis |
| `DataAnalyzer` | `onboarding/data_analyzer.py` | File parsing & schema extraction |
| `SynthConfigBuilder` | `onboarding/config_builder.py` | Synthesis config generation |
| `JudgeConfigBuilder` | `onboarding/config_builder.py` | Judge config generation |
| `cache.py` | `cli/onboard/cache.py` | Resume capability |
| `prompts/` | `cli/onboard/prompts/` | Jinja2 templates for LLM calls |

---

## Data Transformation Summary

1. **File → DataSchema**: DataAnalyzer extracts columns, samples, format
2. **DataSchema → DomainAnalysis**: LLM infers domain semantics
3. **Files + Domain → Task**: LLM generates task suggestions
4. **Task + Schema → System Prompt**: LLM creates system prompt
5. **Task → Task Type**: LLM classifies (extraction/classification/qa/generation/transformation)
6. **Schema + Task → Input Source**: LLM identifies input column
7. **Task → Quality Criteria**: LLM suggests evaluation criteria
8. **State → YAML Configs**: Config builders generate final outputs
