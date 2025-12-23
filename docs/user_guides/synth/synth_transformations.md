# Transformation Strategies

Transformed attributes apply rule-based conversions to existing attributes without using LLMs. They're useful for formatting, combining, and restructuring data.

## Overview

Oumi supports four transformation types:

| Type | Description | Output Type |
|------|-------------|-------------|
| `STRING` | Format string interpolation | string |
| `LIST` | Create a list from attributes | list |
| `DICT` | Create a dictionary from attributes | dict |
| `CHAT` | Format as chat conversation | list of messages |

## Sample Data

For the examples below, assume we have a data sample with these values:

```json
{
  "question": "What color is the sky?",
  "answer": "The sky is blue."
}
```

## String Transformation

Create formatted strings by interpolating attributes:

```yaml
transformed_attributes:
  - id: formatted_qa
    transformation_strategy:
      type: STRING
      string_transform: "Question: {question}\nAnswer: {answer}"
```

**Result:**

```json
{
  "formatted_qa": "Question: What color is the sky?\nAnswer: The sky is blue."
}
```

### Use Cases

- Creating prompts with specific formats
- Combining multiple fields into one
- Adding labels or prefixes

### Advanced Example

```yaml
transformed_attributes:
  - id: prompt
    transformation_strategy:
      type: STRING
      string_transform: |
        ### Instruction
        Answer the following question accurately.

        ### Question
        {question}

        ### Response
        {answer}
```

## List Transformation

Create a list from multiple attributes:

```yaml
transformed_attributes:
  - id: qa_list
    transformation_strategy:
      type: LIST
      list_transform:
        - "{question}"
        - "{answer}"
```

**Result:**

```json
{
  "qa_list": [
    "What color is the sky?",
    "The sky is blue."
  ]
}
```

### Use Cases

- Creating sequences for training
- Building input-output pairs
- Preparing data for specific model formats

### Advanced Example

```yaml
transformed_attributes:
  - id: multi_turn
    transformation_strategy:
      type: LIST
      list_transform:
        - "You are a helpful assistant."
        - "{question}"
        - "{answer}"
        - "Thank you for the answer!"
        - "You're welcome! Let me know if you have more questions."
```

## Dictionary Transformation

Create a dictionary with named fields:

```yaml
transformed_attributes:
  - id: qa_dict
    transformation_strategy:
      type: DICT
      dict_transform:
        query: "{question}"
        response: "{answer}"
```

**Result:**

```json
{
  "qa_dict": {
    "query": "What color is the sky?",
    "response": "The sky is blue."
  }
}
```

### Use Cases

- Restructuring data for different formats
- Creating nested structures
- Renaming fields

### Advanced Example

```yaml
transformed_attributes:
  - id: training_sample
    transformation_strategy:
      type: DICT
      dict_transform:
        instruction: "Answer the question."
        input: "{question}"
        output: "{answer}"
        metadata:
          topic: "{topic}"
          difficulty: "{difficulty}"
```

## Chat Transformation

Format data as a chat conversation with roles:

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

**Result:**

```json
{
  "conversation": {
    "messages": [
      {"role": "user", "content": "What color is the sky?"},
      {"role": "assistant", "content": "The sky is blue."}
    ]
  }
}
```

### Supported Roles

| Role | Description |
|------|-------------|
| `SYSTEM` | System prompt/instructions |
| `USER` | Human/user message |
| `ASSISTANT` | AI/model response |

### Multi-Turn Conversations

```yaml
transformed_attributes:
  - id: conversation
    transformation_strategy:
      type: CHAT
      chat_transform:
        messages:
          - role: SYSTEM
            content: "You are a helpful educational assistant."
          - role: USER
            content: "{question}"
          - role: ASSISTANT
            content: "{answer}"
          - role: USER
            content: "Can you explain that in more detail?"
          - role: ASSISTANT
            content: "{detailed_explanation}"
```

### With System Prompt

```yaml
transformed_attributes:
  - id: conversation
    transformation_strategy:
      type: CHAT
      chat_transform:
        messages:
          - role: SYSTEM
            content: "You are an expert in {topic}. Provide accurate, concise answers."
          - role: USER
            content: "{question}"
          - role: ASSISTANT
            content: "{answer}"
```

## Chaining Transformations

You can reference transformed attributes in other transformations:

```yaml
transformed_attributes:
  # First: Create a formatted QA string
  - id: qa_text
    transformation_strategy:
      type: STRING
      string_transform: "Q: {question}\nA: {answer}"

  # Second: Include the QA text in a conversation
  - id: conversation
    transformation_strategy:
      type: CHAT
      chat_transform:
        messages:
          - role: USER
            content: "Review this Q&A:\n{qa_text}"
          - role: ASSISTANT
            content: "{review}"
```

## Complete Example

Here's a complete synthesis config using multiple transformation types:

```yaml
strategy: GENERAL
num_samples: 50
output_path: training_data.jsonl

strategy_params:
  sampled_attributes:
    - id: topic
      possible_values:
        - {id: science, name: Science}
        - {id: history, name: History}

  generated_attributes:
    - id: question
      instruction_messages:
        - role: USER
          content: "Create a {topic} question."
    - id: answer
      instruction_messages:
        - role: USER
          content: "Answer: {question}"
    - id: explanation
      instruction_messages:
        - role: USER
          content: "Explain the answer to: {question}"

  transformed_attributes:
    # String: Create a summary
    - id: summary
      transformation_strategy:
        type: STRING
        string_transform: "[{topic}] {question} - {answer}"

    # Dict: Create Alpaca format
    - id: alpaca_format
      transformation_strategy:
        type: DICT
        dict_transform:
          instruction: "Answer the following {topic} question."
          input: "{question}"
          output: "{answer}"

    # Chat: Create conversation format
    - id: conversation
      transformation_strategy:
        type: CHAT
        chat_transform:
          messages:
            - role: SYSTEM
              content: "You are a knowledgeable assistant."
            - role: USER
              content: "{question}"
            - role: ASSISTANT
              content: "{answer}"
            - role: USER
              content: "Why is that the answer?"
            - role: ASSISTANT
              content: "{explanation}"

  passthrough_attributes:
    - conversation
    - alpaca_format
    - topic

inference_config:
  model:
    model_name: "gpt-4o-mini"
  engine: OPENAI
```

## Tips

1. **Use CHAT for SFT**: The `CHAT` transformation creates the format expected by Oumi's SFT training.

2. **Combine with postprocessing**: Generate raw content, then transform it into the final format.

3. **Keep transformations simple**: Complex logic is better handled in generated attributes with LLMs.

4. **Test incrementally**: Start with a few samples to verify transformations produce expected output.

## See Also

- {doc}`synth_config` - Complete configuration reference
- {doc}`synth_attributes` - Detailed attribute configuration
- {doc}`/resources/datasets/data_formats` - Oumi data formats
