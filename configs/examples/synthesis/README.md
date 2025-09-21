# Synthesis Examples

This directory contains example configurations for different data synthesis use cases using the `oumi synth` command. Each example demonstrates how to generate specific types of synthetic training data.

## Available Examples

### 1. Question-Answer Generation (`question_answer_synth.yaml`)

**Purpose**: Generate QA pairs from documents or contexts for training conversational models.

**What it does**: Creates geography quiz questions with varying difficulty levels (easy, medium, hard) across different topics (capitals, physical geography, countries, climate).

**Key features**:
- Uses example questions for few-shot learning
- Generates both questions and answers separately
- Includes difficulty and topic classification
- Produces 50 samples with balanced difficulty distribution

**Run with**:
```bash
oumi synth -c configs/examples/synthesis/question_answer_synth.yaml
```

<details>
<summary><strong>Example Output</strong></summary>

```json
{
  "difficulty": "easy",
  "topic": "climate",
  "cleaned_question": "Which climate has hot temperatures year-round and high levels of rainfall?",
  "cleaned_answer": "The climate zone characterized by hot temperatures year-round and high levels of rainfall, typically found near the Earth's equator, is the tropical rainforest climate, also known as the equatorial climate or tropical wet climate.",
  "conversation": {
    "conversation_id": "conversation-f6ffd3b5-2605-420a-be16-704772a17de8",
    "messages": [
      {
        "content": "Which climate has hot temperatures year-round and high levels of rainfall?",
        "role": "user"
      },
      {
        "content": "The climate zone characterized by hot temperatures year-round and high levels of rainfall, typically found near the Earth's equator, is the tropical rainforest climate, also known as the equatorial climate or tropical wet climate.",
        "role": "assistant"
      }
    ],
    "metadata": {}
  }
}
```

</details>

### 2. Data Augmentation (`data_augmentation_synth.yaml`)

**Purpose**: Create variations of existing datasets by sampling different attributes and regenerating content.

**What it does**: Takes existing instruction-response pairs and creates variations with different styles (professional, casual, educational) and complexity levels (simple, detailed, expert).

**Key features**:
- Uses input data from existing datasets
- Generates augmented versions with different styles and complexity
- Preserves original data for comparison
- Demonstrates combination sampling for specific style-complexity pairs

**Run with**:
```bash
oumi synth -c configs/examples/synthesis/data_augmentation_synth.yaml
```

<details>
<summary><strong>Example Output</strong></summary>

```json
{
  "original_instruction": "What's the weather like in Seattle today?",
  "original_input": "",
  "original_response": "I apologize, but I don't have access to real-time weather information for Seattle.",
  "augmentation_type": "rephrase",
  "phrasing_style": "formal",
  "specificity_level": "precise",
  "cleaned_instruction": "Please provide a comprehensive report on the current meteorological conditions in Seattle, Washington. I require specific details on temperature, precipitation, wind speed, and cloud cover for today's date. Kindly ensure the information is up-to-date and accurate as of the most recent weather update for the city.",
  "cleaned_response": "I apologize, but I don't have access to real-time or up-to-date meteorological data for Seattle, Washington. As an AI language model, I'm not able to provide current information on temperature, precipitation, wind speed, or cloud cover for today's date in Seattle. For the most accurate and recent weather conditions, I recommend checking a reliable weather service or meteorological website.",
  "conversation": {
    "conversation_id": "conversation-485db404-87ea-47c1-8b3e-60cc771da4fc",
    "messages": [
      {
        "content": "Please provide a comprehensive report on the current meteorological conditions in Seattle, Washington. I require specific details on temperature, precipitation, wind speed, and cloud cover for today's date. Kindly ensure the information is up-to-date and accurate as of the most recent weather update for the city.",
        "role": "user"},
      {
        "content": "I apologize, but I don't have access to real-time or up-to-date meteorological data for Seattle, Washington. As an AI language model, I'm not able to provide current information on temperature, precipitation, wind speed, or cloud cover for today's date in Seattle. For the most accurate and recent weather conditions, I recommend checking a reliable weather service or meteorological website.",
        "role": "assistant"}
    ],
    "metadata": {}
  }
}
```

</details>

### 3. Instruction Following (`instruction_following_synth.yaml`)

**Purpose**: Generate instruction-response pairs with varying complexity and domains.

**What it does**: Creates diverse task instructions across multiple domains (writing, analysis, coding, math, science, business) with different complexity levels and task formats.

**Key features**:
- Multi-domain instruction generation
- Varying complexity levels (beginner, intermediate, advanced)
- Different task formats (explain, create, analyze, solve, summarize)
- Balanced distribution with targeted combinations

**Run with**:
```bash
oumi synth -c configs/examples/synthesis/instruction_following_synth.yaml
```

<details>
<summary><strong>Example Output</strong></summary>

```json
{
  "domain": "writing",
  "complexity": "beginner",
  "task_format": "create",
  "cleaned_instruction": "Write a short story of 250-300 words about a child's first day of school. Include descriptions of the child's emotions, the classroom environment, and an interaction with a new classmate.",
  "cleaned_response": "Lily's heart raced as she clutched her mother's hand...already looking forward to tomorrow's adventures.",
  "conversation": {
    "conversation_id": "conversation-0670db9d-6147-4d71-bcbd-a05a7f8f8d1a",
    "messages": [
      {
        "content": "Write a short story of 250-300 words about a child's first day of school. Include descriptions of the child's emotions, the classroom environment, and an interaction with a new classmate.",
        "role": "user"
      },
      {
        "content": "Lily's heart raced as she clutched her mother's hand...already looking forward to tomorrow's adventures.",
        "role": "assistant"
      }
    ],
    "metadata": {}
  }
}
```

</details>

### 4. Conversation Agent (`conversation_synth.yaml`)

**Purpose**: Create multi-turn conversations by chaining generated responses. Includes examples of structured actions parseable for tool-use.

**What it does**: Generates realistic customer support conversations with different scenarios, customer personalities, and resolution outcomes.

**Key features**:
- Multi-turn conversation generation (4 messages)
- Different customer personalities (friendly, frustrated, confused, demanding, curious)
- Various support scenarios (account issues, billing, product questions, technical support, refunds)
- Converts to chat format for training
- Demonstrates conversation flow and natural progression

**Run with**:
```bash
oumi synth -c configs/examples/synthesis/conversation_synth.yaml
```

<details>
<summary><strong>Example Output</strong></summary>

```json
{
  "scenario": "account_issue",
  "customer_type": "concise",
  "customer_interaction": "cooperative",
  "cleaned_opener": "Can't log in. Need password reset ASAP. BrightTech account.",
  "cleaned_agent_response": "<ACTION>\ntype: CLARIFY\nparameters: { \"reason\": \"Need email address to proceed with escalation\" }\n</ACTION>\nI apologize for the inconvenience you're experiencing with logging into your BrightTech account. To assist you with a password reset, I'll need to escalate this to our account specialists. Could you please provide the email address associated with your BrightTech account?",
  "cleaned_followup": "jsmith@email.com. That's the one.",
  "cleaned_final_response": "<ACTION>\ntype: ESCALATE\nparameters: { \"account_email\": \"jsmith@email.com\", \"reason\": \"Password reset request\" }\n</ACTION>\nThank you for providing your email address. I've escalated your password reset request to our account specialists. They will process your request as soon as possible and send instructions to jsmith@email.com on how to reset your password. Please check your email inbox (including spam/junk folders) for further instructions. Is there anything else I can help you with regarding your BrightTech account?",
  "conversation": {
    "conversation_id": "conversation-8658569a-8024-48e5-831e-00f0fbe150a6",
    "messages": [
      {
        "content": "You are CareBot...",
        "role": "system"
      },
      {
        "content": "Can't log in. Need password reset ASAP. BrightTech account.",
        "role": "user"
      },
      {
        "content": "<ACTION>\ntype: CLARIFY\nparameters: { \"reason\": \"Need email address to proceed with escalation\" }\n</ACTION>\nI apologize for the inconvenience you're experiencing with logging into your BrightTech account. To assist you with a password reset, I'll need to escalate this to our account specialists. Could you please provide the email address associated with your BrightTech account?",
        "role": "assistant"
      },
      {
        "content": "jsmith@email.com. That's the one.",
        "role": "user"
      },
      {
        "content": "<ACTION>\ntype: ESCALATE\nparameters: { \"account_email\": \"jsmith@email.com\", \"reason\": \"Password reset request\" }\n</ACTION>\nThank you for providing your email address. I've escalated your password reset request to our account specialists. They will process your request as soon as possible and send instructions to jsmith@email.com on how to reset your password. Please check your email inbox (including spam/junk folders) for further instructions. Is there anything else I can help you with regarding your BrightTech account?",
        "role": "assistant"
      }
    ],
    "metadata": {}
  }
}
```

</details>

### 5. Domain-specific QA (`domain_qa_synth.yaml`)

**Purpose**: Generate domain-specific training data by conditioning on domain attributes.

**What it does**: Creates medical Q&A data across different medical specialties with appropriate context and complexity levels.

**Key features**:
- Medical specialty focus (cardiology, dermatology, pediatrics, neurology, orthopedics, endocrinology)
- Context-aware generation (patient education, diagnosis support, treatment guidance, prevention advice)
- Complexity levels for different audiences (basic, intermediate, professional)
- Includes medical terminology explanations
- Demonstrates domain-specific content generation

**Run with**:
```bash
oumi synth -c configs/examples/synthesis/domain_qa_synth.yaml
```

<details>
<summary><strong>Example Output</strong></summary>

```json
{
  "specialty": "dermatology",
  "context_type": "prevention_advice",
  "complexity_level": "basic",
  "cleaned_question": "What are three important steps you can take to protect your skin from sun damage and reduce your risk of skin cancer?",
  "cleaned_answer": "Here are three important steps you can take to protect your skin from sun damage...",
  "conversation": {
    "conversation_id": "conversation-ea1eccda-d7ba-4b34-86f9-52989aa11ae6",
    "messages": [
      {
        "content": "What are three important steps you can take to protect your skin from sun damage and reduce your risk of skin cancer?",
        "role": "user"
      },
      {
        "content": "Here are three important steps you can take to protect your skin from sun damage...",
        "role": "assistant"
      }
    ],
    "metadata": {}
  }
}
```

</details>

## Usage Tips

### Before Running

1. **Set up API access**: Most examples use Claude 3.5 Sonnet. Make sure you have:
   - Anthropic API key set in your environment (`ANTHROPIC_API_KEY`)
   - Or modify the `inference_config` to use a different model/engine

2. **Check output paths**: Examples save to files like `geography_qa_dataset.jsonl`. Modify `output_path` if needed.

3. **Adjust sample counts**: Start with smaller `num_samples` for testing, then scale up.

### Customization

- **Change the model**: Modify `inference_config.model.model_name` and `engine`
- **Adjust generation parameters**: Modify `temperature`, `max_new_tokens`, etc.
- **Add your own data**: Replace `input_examples` or add `input_data` paths
- **Modify attributes**: Change `sampled_attributes` and `generated_attributes` for your use case
- **Control distribution**: Use `sample_rate` and `combination_sampling` to control output distribution

### Common Modifications

```yaml
# Use a different model
inference_config:
  model:
    model_name: gpt-4o
  engine: OPENAI

# Add your own input data
strategy_params:
  input_data:
    - path: "path/to/your/data.jsonl"
      attribute_map:
        old_field: new_attribute

# Generate more samples
num_samples: 100

# Use different output format
output_path: my_custom_dataset.jsonl

# Increase workers for higher throughput
inference_config:
  max_workers: 100  # Increase for higher generation throughput based on your API limts
```

## Next Steps

After generating synthetic data:

1. **Review the output**: Check the generated samples for quality and relevance
2. **Use for training**: Include the dataset in your training configuration (see our [training guide](../../docs/user_guides/train/train.md) for more details)
3. **Iterate and improve**: Modify the synthesis config based on results
4. **Combine datasets**: Use multiple synthesis runs to create larger, more diverse datasets

For more information, see the [Data Synthesis Guide](../../docs/user_guides/synth.md).
