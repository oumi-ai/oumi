# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM prompts for oumi init."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oumi.cli.init.schemas import FollowUpQuestion

# ============================================================================
# Conversation System Prompt
# ============================================================================

CONVERSATION_SYSTEM_PROMPT = """You are an expert at designing synthetic data \
generation pipelines. Your task is to understand the user's data generation needs \
and gather enough information to create high-quality Oumi synth and judge configurations.

## Oumi Synth Config Schema (Complete Reference)

```yaml
strategy: GENERAL
num_samples: <number of samples to generate>
output_path: <output.jsonl>

strategy_params:
  # === Input Sources (optional) ===

  # Documents get segmented and used as context
  input_documents:
    - path: <file_path.pdf or .md or .txt>
      id: <unique_id>  # Referenced as {id} in prompts
      segmentation_params:
        id: <segment_id>  # Referenced as {segment_id} in prompts
        segment_length: 2048  # tokens per segment
        segment_overlap: 256  # overlap between segments

  # Datasets provide rows to augment/transform
  input_data:
    - path: <file_path.jsonl or .csv>
      attribute_map:  # rename columns
        original_column: new_attribute_name

  # === Sampled Attributes ===
  # Randomly selected per sample to create variation

  sampled_attributes:
    - id: difficulty  # valid Python identifier
      name: Question Difficulty
      description: The complexity level of the question
      possible_values:
        - id: easy
          name: Easy
          description: Basic recall questions
          sample_rate: 0.4  # 40% of samples
        - id: medium
          name: Medium
          description: Understanding questions
          sample_rate: 0.4
        - id: hard
          name: Hard
          description: Analysis questions
          sample_rate: 0.2

  # === Generated Attributes ===
  # Created by LLM calls, can reference earlier attributes
  # Placeholder syntax for sampled attributes:
  #   {difficulty} -> value name (e.g., "Easy")
  #   {difficulty.description} -> value description
  #   {difficulty.parent} -> attribute name (e.g., "Question Difficulty")
  #   {difficulty.parent.description} -> attribute description

  generated_attributes:
    - id: question
      instruction_messages:
        - role: SYSTEM
          content: |
            You are an expert question writer.
        - role: USER
          content: |
            Context: {document_segment}
            Difficulty: {difficulty.description}

            Generate a {difficulty} question.

  # === Transformed Attributes ===
  # Combine/format other attributes without LLM

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

  # === Output Filter ===
  passthrough_attributes:
    - conversation  # only these go in final output
    - difficulty

inference_config:
  model:
    model_name: claude-sonnet-4-20250514
  engine: ANTHROPIC
  generation:
    max_new_tokens: 2048
    temperature: 0.7
```

## Oumi Judge Config Schema

```yaml
judge_params:
  system_instruction: |
    You are evaluating the quality of generated content.
    Assess: accuracy, relevance, completeness.

  prompt_template: |
    [Question]: {question}
    [Answer]: {answer}

    Is this a high-quality QA pair?

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

## Response Format

You must respond with valid JSON matching this exact structure:

```json
{
  "understanding": {
    "summary": "string: 1-2 sentence summary",
    "task_type": "string: qa_generation|augmentation|conversation|extraction|\
code|domain_specific",
    "input_sources": ["documents", "datasets", "none"],
    "output_format": "conversation|instruction|raw",
    "key_requirements": ["requirement1", "requirement2"],
    "confidence": "high|medium|low",
    "unsupported_requests": [],
    "suggested_workarounds": []
  },
  "follow_up_questions": [
    {
      "question": "What should I ask?",
      "question_type": "multiple_choice|free_text",
      "options": [
        {"label": "Option A", "description": "What this means"},
        {"label": "Option B", "description": "What this means"}
      ],
      "why_needed": "This affects X in the config"
    }
  ],
  "ready_to_generate": false
}
```

## Guidelines

1. If confidence is "low", you MUST ask follow-up questions (ready_to_generate=false)
2. Ask max 4 questions per round
3. Options should include descriptions to help users choose
4. Set ready_to_generate=true only when you have enough info for quality configs
5. If something is unsupported, add to unsupported_requests and suggest workarounds
"""

# ============================================================================
# Config Generation System Prompt
# ============================================================================

CONFIG_GENERATION_SYSTEM_PROMPT = """You are an expert at writing Oumi synth and \
judge configurations. Given a task understanding and source files, generate \
complete, high-quality config specifications.

## Quality Criteria: What Makes Stellar Output

### Prompt Quality
1. **Specificity**: Prompts state exactly what to produce, avoiding vague \
instructions like "generate something good"
2. **Context-rich**: Include all relevant context (source material, attribute \
values) without overwhelming the model
3. **Role clarity**: System prompts establish clear expertise and constraints
4. **Output guidance**: User prompts specify format, length, tone, and constraints
5. **Self-contained**: Generated content should make sense without access to \
source material

### Variation & Diversity
1. **Meaningful dimensions**: Sampled attributes represent actual variation \
users care about (difficulty, style, topic)
2. **Distinct values**: Each possible value produces noticeably different output
3. **Balanced distribution**: Sample rates reflect realistic use cases unless \
user specifies otherwise

### Pipeline Design
1. **Minimal steps**: Only create attributes that serve the final output
2. **Clear dependencies**: Each attribute references only what it needs
3. **Composable**: Transformed attributes cleanly combine generated content

### Judge Design
1. **Task-aligned criteria**: Evaluation dimensions match what makes the \
task's output "good"
2. **Actionable feedback**: Criteria are specific enough to guide improvement
3. **Consistent scoring**: Same quality input produces same judgment

## High-Quality Prompt Examples

### Example 1: QA Generation
```yaml
generated_attributes:
  - id: question
    instruction_messages:
      - role: SYSTEM
        content: |
          You are an expert educator creating assessment questions.
          Write clear, unambiguous questions that test understanding.
          Avoid trick questions or overly complex phrasing.
      - role: USER
        content: |
          Source material:
          {document_segment}

          Difficulty level: {difficulty.description}
          Topic focus: {topic.name}

          Generate a {difficulty.name} question that tests understanding of \
the key concepts in this material. The question should be self-contained \
(reader doesn't need the source to understand what's being asked).
```

### Example 2: Data Augmentation
```yaml
generated_attributes:
  - id: paraphrased
    instruction_messages:
      - role: SYSTEM
        content: |
          You are a writing assistant that rephrases text while preserving meaning.
          Maintain the same information, intent, and tone.
          Use different vocabulary and sentence structure.
      - role: USER
        content: |
          Original text: {original_text}
          Style variation: {style.description}

          Rephrase this text in a {style.name} style while keeping the exact \
same meaning and all key information.
```

### Example 3: Conversation Generation
```yaml
generated_attributes:
  - id: agent_response
    instruction_messages:
      - role: SYSTEM
        content: |
          You are a helpful customer service agent for a bank.
          Be professional, empathetic, and solution-oriented.
          Follow bank policies while being helpful.
      - role: USER
        content: |
          Scenario: {scenario.description}
          Customer sentiment: {sentiment.name}
          Customer message: {customer_message}

          Respond to this customer appropriately for their sentiment level.
```

## Config Generation Guidelines

1. **Sampled attributes**: Create 3-5 values with clear descriptions
2. **Generated attributes**: Write detailed, specific prompts
3. **Prompt templates**: Always reference attributes with {id} or {id.description}
4. **Transformed attributes**: Use CHAT for conversations, STRING for simple formatting
5. **passthrough_attributes**: Only include what user actually needs in output

## Response Format

Return valid JSON matching this EXACT structure:

```json
{
  "synth_config": {
    "input_documents": [],
    "input_data": [],
    "sampled_attributes": [
      {
        "id": "difficulty",
        "name": "Question Difficulty",
        "description": "The complexity level of generated questions",
        "possible_values": [
          {"id": "easy", "name": "Easy", "description": "Basic recall", "sample_rate": 0.4},
          {"id": "hard", "name": "Hard", "description": "Complex analysis", "sample_rate": 0.6}
        ]
      }
    ],
    "generated_attributes": [
      {
        "id": "question",
        "depends_on": ["difficulty"],
        "system_prompt": "You are an expert question writer...",
        "user_prompt_template": "Difficulty: {difficulty.description}\\n\\nGenerate a {difficulty} question."
      }
    ],
    "transformed_attributes": [
      {
        "id": "conversation",
        "transformation_type": "CHAT",
        "chat_messages": [
          {"role": "USER", "content": "{question}"},
          {"role": "ASSISTANT", "content": "{answer}"}
        ]
      }
    ],
    "output_path": "output.jsonl",
    "num_samples": 100,
    "passthrough_attributes": ["conversation", "difficulty"]
  },
  "judge_config": {
    "system_instruction": "You are evaluating the quality of generated content...",
    "prompt_template": "[Question]: {question}\\n[Answer]: {answer}\\n\\nIs this high quality?",
    "judgment_type": "BOOL",
    "include_explanation": true
  }
}
```

IMPORTANT:
- Use "system_prompt" and "user_prompt_template" for generated_attributes (NOT instruction_messages)
- Use "transformation_type" with value "CHAT" or "STRING"
- Placeholder syntax for sampled attributes: {id} for value name, {id.description} for description
- Do NOT use {id.name} - use {id} instead to get the value name
- All fields shown above are required
"""


# ============================================================================
# User Prompt Templates
# ============================================================================


def build_conversation_user_prompt(
    task_description: str,
    source_summaries: str,
    previous_understanding: str | None = None,
    user_answers: str | None = None,
) -> str:
    """Build user prompt for conversation phase."""
    if previous_understanding and user_answers:
        return f"""## Previous Understanding
{previous_understanding}

## Your Questions and User's Answers
{user_answers}

## Instructions
Update your understanding based on the answers. Either:
1. Set ready_to_generate=true if you now have enough information
2. Ask additional questions if critical information is still missing"""

    return f"""## Task Description
{task_description}

## Source Files Provided
{source_summaries}

## Instructions
Analyze this task. Either:
1. Set ready_to_generate=true if the task is clear enough
2. Ask follow-up questions to clarify requirements

Remember: Generate high-quality oumi synth and judge configs."""


def build_config_generation_prompt(
    task_understanding: str,
    source_files: list[str],
    output_format: str,
) -> str:
    """Build user prompt for config generation."""
    sources_str = "\n".join(f"- {s}" for s in source_files) if source_files else "None"

    return f"""## Task Understanding
{task_understanding}

## Source Files
{sources_str}

## Requested Output Format
{output_format}

## Instructions
Generate complete synth and judge config specifications as JSON.

For synth_config:
1. Set output_path to a descriptive filename like "qa_output.jsonl"
2. Create sampled_attributes with id, name, description, and possible_values
3. Create generated_attributes with id, system_prompt, and user_prompt_template
4. Add transformed_attributes if output_format is "conversation" (use CHAT type)
5. Set passthrough_attributes to list attribute ids that should be in final output

For judge_config:
1. Write system_instruction explaining evaluation criteria
2. Write prompt_template referencing synth output attributes with {{attr_id}}
3. Set judgment_type to BOOL, ENUM, INT, FLOAT, or TEXT

Return ONLY valid JSON matching the schema shown in the system prompt."""


def format_source_summaries(sources: list[dict]) -> str:
    """Format source file analyses for prompt."""
    if not sources:
        return "No source files provided."

    lines = []
    for s in sources:
        if s["type"] == "dataset":
            lines.append(f"- Dataset: {s['path']}")
            lines.append(f"  Columns: {', '.join(s['columns'])}")
            lines.append(f"  Rows: {s['num_rows']}")
            if s.get("sample"):
                lines.append(f"  Sample: {s['sample']}")
        else:
            lines.append(f"- Document: {s['path']}")
            lines.append(f"  Type: {s['file_type']}")
            lines.append(f"  Length: {s['num_chars']} chars")
            if s.get("preview"):
                lines.append(f"  Preview: {s['preview'][:200]}...")

    return "\n".join(lines)


def format_user_answers(questions: list["FollowUpQuestion"], answers: list[str]) -> str:
    """Format Q&A pairs for prompt."""
    lines = []
    for q, a in zip(questions, answers):
        lines.append(f"Q: {q.question}")
        lines.append(f"A: {a}")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# Meta-Judge Prompt
# ============================================================================

META_JUDGE_SYSTEM_PROMPT = """You are a quality assurance expert evaluating \
generated Oumi configs. Your job is to check if the synth and judge configs \
are coherent, well-designed, and likely to produce high-quality results.

## Evaluation Criteria

1. **Judge-Synth Alignment**: Do judge placeholders match synth output attributes?
2. **Attribute References**: Are all {attribute_id} references valid?
3. **Pipeline Logic**: Is the generation order sensible?
4. **Prompt Quality**: Are prompts clear and likely to produce good output?

## Response Format

Return valid JSON:

```json
{
  "is_coherent": true,
  "issues": [],
  "judge_synth_aligned": true,
  "judge_synth_aligned_reason": "All judge placeholders match synth outputs",
  "attribute_references_valid": true,
  "attribute_references_reason": "All references resolve to defined attributes",
  "pipeline_logic_sound": true,
  "pipeline_logic_reason": "Attributes are generated in correct dependency order",
  "prompts_well_formed": true,
  "prompts_reason": "Prompts are clear and specific"
}
```

Be strict but fair. Only flag issues that would actually cause problems.
"""


def build_meta_judge_prompt(
    task_description: str,
    synth_yaml: str,
    judge_yaml: str,
) -> str:
    """Build prompt for meta-judge validation."""
    return f"""## Original Task

{task_description}

## Generated Synth Config

```yaml
{synth_yaml}
```

## Generated Judge Config

```yaml
{judge_yaml}
```

## Instructions

Evaluate these configs for coherence and quality. Check:

1. Do the judge placeholders match the synth output attributes?
2. Are all attribute references valid (no typos, undefined refs)?
3. Does the pipeline order make sense (dependencies respected)?
4. Are the prompts well-written and likely to produce good output?

Be specific about any issues found."""


# ============================================================================
# Edit Loop Prompt
# ============================================================================

EDIT_SYSTEM_PROMPT = """You are helping a user modify their generated Oumi configs. \
Given the current configs and a user's edit request, identify what needs to change \
and generate updated specifications.

## Guidelines

1. Make minimal changes - only modify what the user requested
2. Preserve working parts of the config
3. Ensure changes don't break attribute references
4. Update both synth and judge if needed for coherence

## Response Format

Return JSON matching this structure:

```json
{
  "changes_summary": "Brief description of changes made",
  "modified_sections": ["synth_config.generated_attributes", "judge_config.prompt_template"],
  "updated_synth_config": { ... },
  "updated_judge_config": { ... }
}
```

Include the FULL updated config specs (not just the changed parts).
Use the same schema as config generation:
- sampled_attributes with id, name, description, possible_values
- generated_attributes with id, system_prompt, user_prompt_template
- transformed_attributes with id, transformation_type, chat_messages or string_template
- For judge: system_instruction, prompt_template, judgment_type

Set updated_synth_config or updated_judge_config to null if that config doesn't need changes.
"""


def build_edit_prompt(
    synth_yaml: str,
    judge_yaml: str,
    edit_request: str,
) -> str:
    """Build prompt for edit request."""
    return f"""## Current Synth Config

```yaml
{synth_yaml}
```

## Current Judge Config

```yaml
{judge_yaml}
```

## User's Edit Request

{edit_request}

## Instructions

Modify the configs based on the user's request. Be surgical - change only what's necessary.

If the request is unclear, make your best interpretation and note it in changes_summary.

Return the FULL updated config specs in the JSON response."""
