# Config Generation with oumi init

The `oumi init` command generates synthetic data pipeline configurations from natural language descriptions. Instead of manually writing complex YAML configs, you describe what you want to build and the AI creates both the synthesis (`synth`) and quality evaluation (`judge`) configurations for you.

## What You Can Build

- **Question-Answer datasets** - Trivia, reading comprehension, educational content
- **Data augmentation pipelines** - Paraphrases, variations, style transfers
- **Conversation datasets** - Customer service, multi-turn dialogues, roleplay
- **Information extraction** - Entity extraction, summarization, structured output
- **Code generation** - Documentation, test cases, code comments
- **Domain-specific data** - Medical, legal, financial, technical content

## How It Works

The `oumi init` command uses a conversational AI approach:

1. **Analyze** - You describe your task in plain English
2. **Clarify** - The AI asks follow-up questions to understand requirements
3. **Generate** - Complete synth and judge configs are created
4. **Validate** - Configs are verified for correctness
5. **Preview** - Test the pipeline with sample data before saving

## Quick Start

### Prerequisites

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY='your-key-here'
```

### Basic Usage

Generate configs for a simple task:

```bash
oumi init --task "Generate trivia questions about world geography"
```

The AI will:
1. Analyze your task description
2. Ask clarifying questions (difficulty levels, output format, etc.)
3. Generate both `synth_config.yaml` and `judge_config.yaml`
4. Show a preview and let you save or edit

### Loading Task from a File

For complex task descriptions or version control, use a file:

```bash
# Create a task file
cat > task.txt << 'EOF'
Generate reading comprehension questions from academic papers.
Include questions about methodology, findings, and implications.
Vary difficulty from undergraduate to expert level.
Each question should test critical thinking, not just recall.
EOF

# Use the task file
oumi init --task-file task.txt
```

This is especially useful for:
- Long, detailed task descriptions
- Tasks with multiple requirements
- Version controlling your task definitions
- Sharing task definitions across teams

### With Source Files

Use existing documents or datasets as input:

```bash
# From a document
oumi init --task "Create reading comprehension questions from this textbook" \
    --source textbook.pdf

# From a dataset
oumi init --task "Generate paraphrases of these support tickets" \
    --source tickets.jsonl

# Multiple sources
oumi init --task "Generate QA pairs from these documents" \
    --source chapter1.md --source chapter2.md

# Combine task file with sources
oumi init --task-file task.txt --source data.jsonl
```

### Non-Interactive Mode

For automation and scripting, skip the conversation:

```bash
# Auto-generate with reasonable defaults
oumi init --non-interactive --task "Generate customer service conversations for a bank"

# Preview without saving (outputs configs to stdout)
oumi init -N -t "Generate coding problems" --dry-run
```

## Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--task` | `-t` | Task description (required*) | - |
| `--task-file` | `-T` | Path to file containing task description (required*) | - |
| `--source` | `-s` | Source file(s) - can be repeated | None |
| `--output-dir` | `-o` | Directory to save configs | `./configs/` |
| `--output-format` | `-f` | Output format: conversation, instruction, raw | `conversation` |
| `--non-interactive` | `-N` | Skip conversation, use defaults | False |
| `--new` | `-n` | Start fresh (ignore existing session) | False |
| `--dry-run` | | Preview without saving | False |

*Either `--task` or `--task-file` is required (but not both)

## The Conversation Flow

In interactive mode, `oumi init` guides you through a structured conversation:

### Phase 1: Task Analysis

The AI analyzes your description and shows its understanding:

```
╭─────────────── Understanding ───────────────╮
│ {                                           │
│   "summary": "Generate geography trivia",   │
│   "task_type": "qa_generation",             │
│   "output_format": "conversation",          │
│   "key_requirements": [                     │
│     "factual accuracy",                     │
│     "varied difficulty"                     │
│   ],                                        │
│   "confidence": "medium"                    │
│ }                                           │
╰─────────────────────────────────────────────╯
```

### Phase 2: Clarification

If needed, the AI asks targeted questions:

```
What difficulty levels should the questions cover?
(This affects the sample distribution in the config)

  1. Easy - Basic facts everyone should know
  2. Easy/Medium/Hard - Full range of difficulty
  3. Only Hard - Expert-level questions

Select option: 2
```

### Phase 3: Config Generation

Once requirements are clear, configs are generated:

```
╭─────────────── synth_config.yaml ───────────────╮
│ # Generated by: oumi init --task                │
│ strategy: GENERAL                               │
│ num_samples: 100                                │
│ output_path: geography_qa.jsonl                 │
│                                                 │
│ strategy_params:                                │
│   sampled_attributes:                           │
│     - id: difficulty                            │
│       name: Difficulty Level                    │
│       ...                                       │
╰─────────────────────────────────────────────────╯
```

### Phase 4: Review & Edit

Before saving, you can:

- **save** - Write configs to disk
- **edit** - Request changes (e.g., "add more difficulty levels")
- **preview** - Run a small test to see sample output
- **cancel** - Discard and exit

```
What would you like to do? [save/edit/preview/cancel]: preview

Running preview with 3 samples...
╭─────────────── Sample 1/3 ───────────────╮
│ {                                        │
│   "question": "What is the capital...",  │
│   "answer": "Paris is the capital..."    │
│ }                                        │
╰──────────────────────────────────────────╯
```

## Session Management

`oumi init` automatically saves your progress:

### Auto-Resume

If you exit mid-conversation, your session is preserved:

```bash
# First run - exit after asking questions
oumi init --task "Generate medical QA" --output-dir ./medical/
# ... answer some questions, then exit ...

# Resume automatically
oumi init --output-dir ./medical/
╭─────────────── Resuming Session ───────────────╮
│ Task: Generate medical QA                       │
│ Phase: generation                               │
│ Sources: 0 file(s)                             │
│                                                │
│ Use --new to start a fresh session             │
╰─────────────────────────────────────────────────╯
```

### Start Fresh

To ignore an existing session:

```bash
oumi init --new --task "Start with a different task"
```

## Generated Configs

### Synth Config

The synthesis config defines how data is generated:

```yaml
# Generated by: oumi init --task
# Task: Generate trivia questions about world geography

strategy: GENERAL
num_samples: 100
output_path: trivia_questions.jsonl

strategy_params:
  sampled_attributes:
    - id: difficulty
      name: Difficulty Level
      description: How challenging the question should be
      possible_values:
        - id: easy
          name: Easy
          description: Basic facts most people know
        - id: hard
          name: Hard
          description: Obscure knowledge for experts

  generated_attributes:
    - id: question
      instruction_messages:
        - role: SYSTEM
          content: You are an expert trivia writer...
        - role: USER
          content: Generate a {difficulty.name} question about {topic}...

    - id: answer
      instruction_messages:
        - role: SYSTEM
          content: You are a knowledgeable assistant...
        - role: USER
          content: Answer this question accurately: {question}

  transformed_attributes:
    - id: conversation
      transformation_strategy:
        type: CHAT
        chat_transform:
          messages:
            - role: USER
              content: "{question}"
            - role: ASSISTANT
              content: "{answer}"

  passthrough_attributes:
    - conversation
    - difficulty

inference_config:
  model:
    model_name: claude-sonnet-4-20250514
  engine: ANTHROPIC
  generation:
    max_new_tokens: 2048
    temperature: 0.7
```

### Judge Config

The judge config defines quality evaluation criteria:

```yaml
# Generated by: oumi init --task
# Task: Evaluate trivia questions

judge_params:
  system_instruction: |
    You are evaluating the quality of trivia questions.
    Assess factual accuracy, question clarity, and answer completeness.

  prompt_template: |
    Question: {question}
    Answer: {answer}

    Evaluate this trivia QA pair. Is it high quality?

  response_format: JSON
  judgment_type: BOOL
  include_explanation: true

inference_config:
  model:
    model_name: claude-sonnet-4-20250514
  engine: ANTHROPIC
  generation:
    max_new_tokens: 1024
    temperature: 0.0
```

## Using Generated Configs

After saving, run your pipeline:

```bash
# Generate synthetic data
oumi synth -c configs/trivia_questions_synth.yaml

# Evaluate quality
oumi judge dataset -c configs/trivia_questions_judge.yaml \
    --input trivia_questions.jsonl
```

## Supported Source Types

### Documents

Text content that gets segmented for context:

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Requires PyPDF2 |
| Word | `.docx` | Requires python-docx |
| Markdown | `.md` | Plain text |
| Text | `.txt` | Plain text |

### Datasets

Structured data to augment or transform:

| Format | Extension | Notes |
|--------|-----------|-------|
| JSONL | `.jsonl` | One JSON object per line |
| JSON | `.json` | Array of objects |
| CSV | `.csv` | Comma-separated |
| TSV | `.tsv` | Tab-separated |
| Excel | `.xlsx`, `.xls` | Requires openpyxl |
| Parquet | `.parquet` | Binary columnar format |

## Best Practices

### Writing Good Task Descriptions

**Be specific about what you want:**

```bash
# Too vague
oumi init --task "Generate questions"

# Better - includes context and requirements
oumi init --task "Generate reading comprehension questions from academic papers. \
    Include questions about methodology, findings, and implications. \
    Vary difficulty from undergraduate to expert level."
```

**Include key requirements:**

```bash
oumi init --task "Generate customer service conversations. \
    The agent should be professional but friendly. \
    Include scenarios for billing inquiries, technical support, and complaints. \
    Conversations should be 3-5 turns each."
```

### Using Task Files

For complex tasks, use `--task-file` to keep descriptions maintainable:

```bash
# task.txt
cat > medical_qa_task.txt << 'EOF'
Generate medical QA pairs from clinical guidelines.

Requirements:
- Focus on diagnosis, treatment, and prevention
- Include both common and rare conditions
- Questions should be answerable from the provided guidelines
- Answers must cite specific guideline sections
- Maintain medical accuracy and appropriate terminology
- Target audience: medical students and residents

Output format:
- Question: Clear, specific clinical scenario
- Answer: Evidence-based response with guideline references
- Difficulty: Beginner, Intermediate, or Advanced
EOF

oumi init --task-file medical_qa_task.txt --source guidelines.pdf
```

Benefits:
- **Version control** - Track changes to task definitions in git
- **Reusability** - Use the same task file across multiple runs
- **Collaboration** - Share task files with teammates
- **Documentation** - Task file serves as project documentation

### Using Non-Interactive Mode Effectively

For automation, provide detailed task descriptions:

```bash
# The more detail, the better the results
oumi init --non-interactive --task "Generate Python coding problems with solutions. \
    Topics: data structures, algorithms, string manipulation. \
    Difficulty: easy to medium. \
    Each problem should have: problem statement, example input/output, solution code. \
    Output as instruction format."
```

### Iterating on Configs

Use the edit loop to refine:

```
What would you like to do? edit

Describe your changes: Add a 'medium' difficulty level between easy and hard

╭─────────────── Changes Applied ───────────────╮
│ Added medium difficulty level                  │
│ Modified: sampled_attributes                   │
╰────────────────────────────────────────────────╯
```

## Troubleshooting

### "ANTHROPIC_API_KEY environment variable not set"

Set your API key:

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

### "Source file not found"

Verify the file path exists:

```bash
ls -la path/to/your/file.jsonl
```

### "Config validation failed"

The generated config has an issue. Try:

1. Use `edit` to fix specific problems
2. Run with `--dry-run` to see configs without saving
3. Start fresh with `--new` and provide more detail

### "Generation failed"

API or network issues. Check:

1. Your API key is valid
2. You have API credits/quota
3. Network connectivity

### Low Quality Results

Improve your task description:

1. Be more specific about requirements
2. Provide example source files
3. Answer clarifying questions thoughtfully
4. Use the preview feature to test before committing

## Next Steps

- Learn about {doc}`synth config options </user_guides/synth>` for fine-tuning
- Explore {doc}`judge configuration </user_guides/judge/judge_config>` for custom evaluation
- See {doc}`inference engines </user_guides/infer/inference_engines>` for model options
