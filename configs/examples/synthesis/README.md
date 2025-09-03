# Synthesis Examples

This directory contains example configurations for different data synthesis use cases using the `oumi synth` command. Each example demonstrates how to generate specific types of synthetic training data.

## Available Examples

### 1. Question-Answer Generation (`question_answer_generation.yaml`)

**Purpose**: Generate QA pairs from documents or contexts for training conversational models.

**What it does**: Creates geography quiz questions with varying difficulty levels (easy, medium, hard) across different topics (capitals, physical geography, countries, climate).

**Key features**:
- Uses example questions for few-shot learning
- Generates both questions and answers separately
- Includes difficulty and topic classification
- Produces 50 samples with balanced difficulty distribution

**Run with**:
```bash
oumi synth -c configs/examples/synthesis/question_answer_generation.yaml
```

### 2. Data Augmentation (`data_augmentation.yaml`)

**Purpose**: Create variations of existing datasets by sampling different attributes and regenerating content.

**What it does**: Takes existing instruction-response pairs and creates variations with different styles (professional, casual, educational) and complexity levels (simple, detailed, expert).

**Key features**:
- Uses input data from existing datasets
- Generates augmented versions with different styles and complexity
- Preserves original data for comparison
- Demonstrates combination sampling for specific style-complexity pairs

**Run with**:
```bash
oumi synth -c configs/examples/synthesis/data_augmentation.yaml
```

### 3. Instruction Following (`instruction_following.yaml`)

**Purpose**: Generate instruction-response pairs with varying complexity and domains.

**What it does**: Creates diverse task instructions across multiple domains (writing, analysis, coding, math, science, business) with different complexity levels and task formats.

**Key features**:
- Multi-domain instruction generation
- Varying complexity levels (beginner, intermediate, advanced)
- Different task formats (explain, create, analyze, solve, summarize)
- Balanced distribution with targeted combinations

**Run with**:
```bash
oumi synth -c configs/examples/synthesis/instruction_following.yaml
```

### 4. Conversation Synthesis (`conversation_synthesis.yaml`)

**Purpose**: Create multi-turn conversations by chaining generated responses.

**What it does**: Generates realistic customer support conversations with different scenarios, customer personalities, and resolution outcomes.

**Key features**:
- Multi-turn conversation generation (4 messages)
- Different customer personalities (friendly, frustrated, confused, demanding, curious)
- Various support scenarios (account issues, billing, product questions, technical support, refunds)
- Converts to chat format for training
- Demonstrates conversation flow and natural progression

**Run with**:
```bash
oumi synth -c configs/examples/synthesis/conversation_synthesis.yaml
```

### 5. Domain Adaptation (`domain_adaptation.yaml`)

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
oumi synth -c configs/examples/synthesis/domain_adaptation.yaml
```

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
2. **Use for training**: Include the dataset in your training configuration
3. **Iterate and improve**: Modify the synthesis config based on results
4. **Combine datasets**: Use multiple synthesis runs to create larger, more diverse datasets

For more information, see the [Data Synthesis Guide](../../docs/user_guides/synth.md).
