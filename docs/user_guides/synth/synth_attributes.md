# Attribute Configuration

Attributes are the building blocks of data synthesis. They define what varies in your generated data and how content is created.

## Attribute Types

Oumi supports three types of attributes:

| Type | Description | Use Case |
|------|-------------|----------|
| **Sampled** | Randomly selected from predefined options | Categories, difficulty levels, topics |
| **Generated** | Created by LLM using instruction messages | Questions, answers, summaries |
| **Transformed** | Rule-based conversions of existing attributes | Format conversions, chat formatting |

## Sampled Attributes

Sampled attributes have values randomly selected from a predefined list. They're perfect for categorical variation in your data.

### Basic Configuration

```yaml
sampled_attributes:
  - id: difficulty
    name: Difficulty Level
    description: How challenging the question should be
    possible_values:
      - id: easy
        name: Easy
        description: Simple, straightforward questions
      - id: medium
        name: Medium
        description: Moderately challenging questions
      - id: hard
        name: Hard
        description: Complex, advanced questions
```

### Controlling Sample Rates

Use `sample_rate` to control the distribution of values:

```yaml
sampled_attributes:
  - id: difficulty
    possible_values:
      - id: easy
        name: Easy
        sample_rate: 0.4  # 40% of samples
      - id: medium
        name: Medium
        sample_rate: 0.4  # 40% of samples
      - id: hard
        name: Hard
        # No sample_rate = remaining 20%
```

### Multiple Attributes

Combine multiple sampled attributes for diverse data:

```yaml
sampled_attributes:
  - id: topic
    possible_values:
      - {id: geography, name: Geography}
      - {id: history, name: History}
      - {id: science, name: Science}
  - id: difficulty
    possible_values:
      - {id: easy, name: Easy}
      - {id: medium, name: Medium}
      - {id: hard, name: Hard}
  - id: style
    possible_values:
      - {id: formal, name: Formal}
      - {id: casual, name: Casual}
```

This creates combinations like "easy geography formal", "hard science casual", etc.

## Generated Attributes

Generated attributes are created by an LLM based on instruction messages. They can reference other attributes.

### Basic Configuration

```yaml
generated_attributes:
  - id: question
    instruction_messages:
      - role: SYSTEM
        content: "You are a teacher creating quiz questions."
      - role: USER
        content: "Create a {difficulty} {topic} question."
```

### Chaining Attributes

Generate attributes that depend on previous ones:

```yaml
generated_attributes:
  - id: question
    instruction_messages:
      - role: USER
        content: "Create a {difficulty} question about {topic}."
  - id: answer
    instruction_messages:
      - role: USER
        content: "Answer this question: {question}"
  - id: explanation
    instruction_messages:
      - role: USER
        content: "Explain why the answer to '{question}' is '{answer}'."
```

### Using Examples

Reference input examples in your prompts:

```yaml
input_examples:
  - examples:
    - example_question: "What is the capital of France?"
      example_answer: "Paris"

generated_attributes:
  - id: question
    instruction_messages:
      - role: SYSTEM
        content: "Generate questions like this: {example_question}"
      - role: USER
        content: "Create a new {topic} question."
```

### Postprocessing

Clean up generated text with postprocessing:

```yaml
generated_attributes:
  - id: summary
    instruction_messages:
      - role: USER
        content: "Summarize: {text}. Format: 'Summary: <summary>'"
    postprocessing_params:
      id: clean_summary
      cut_prefix: "Summary: "
      strip_whitespace: true
```

#### Postprocessing Options

| Parameter | Description |
|-----------|-------------|
| `cut_prefix` | Remove this prefix and everything before it |
| `cut_suffix` | Remove this suffix and everything after it |
| `regex` | Extract content matching regex pattern |
| `strip_whitespace` | Remove leading/trailing whitespace |
| `added_prefix` | Add this prefix to the result |
| `added_suffix` | Add this suffix to the result |
| `keep_original_text_attribute` | Keep original alongside cleaned version |

Full postprocessing example:

```yaml
postprocessing_params:
  id: cleaned_attribute
  keep_original_text_attribute: true
  cut_prefix: "Answer: "
  cut_suffix: "\n\n"
  regex: "\\*\\*(.+?)\\*\\*"  # Extract content between ** **
  strip_whitespace: true
  added_prefix: "Response: "
  added_suffix: "."
```

## Attribute Referencing

In instruction messages and transformations, reference attributes using `{attribute_id}` syntax:

| Syntax | Description |
|--------|-------------|
| `{attribute_id}` | The value/name of the attribute |
| `{attribute_id.description}` | The description of a sampled attribute value |
| `{attribute_id.parent}` | The parent name of a sampled attribute |
| `{attribute_id.parent.description}` | The parent description of a sampled attribute |

### Example

Given this sampled attribute:

```yaml
sampled_attributes:
  - id: difficulty
    name: Difficulty Level
    description: How hard the question is
    possible_values:
      - id: easy
        name: Easy
        description: Basic facts everyone should know
```

If `easy` is sampled:

- `{difficulty}` → "Easy"
- `{difficulty.description}` → "Basic facts everyone should know"
- `{difficulty.parent}` → "Difficulty Level"
- `{difficulty.parent.description}` → "How hard the question is"

## Transformed Attributes

Transformed attributes apply rule-based transformations to existing attributes. They don't use LLMs.

See {doc}`synth_transformations` for detailed transformation strategies.

### Quick Example

```yaml
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
```

## Combining Attribute Types

A typical synthesis config combines all three types:

```yaml
strategy_params:
  # 1. Sample categories
  sampled_attributes:
    - id: topic
      possible_values:
        - {id: math, name: Math}
        - {id: science, name: Science}

  # 2. Generate content based on samples
  generated_attributes:
    - id: question
      instruction_messages:
        - role: USER
          content: "Create a {topic} question."
    - id: answer
      instruction_messages:
        - role: USER
          content: "Answer: {question}"

  # 3. Transform into training format
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

  # 4. Specify what goes in output
  passthrough_attributes:
    - conversation
    - topic
```

## See Also

- {doc}`synth_config` - Complete configuration reference
- {doc}`synth_transformations` - Transformation strategies
