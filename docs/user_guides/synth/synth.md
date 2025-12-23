# Data Synthesis

```{toctree}
:maxdepth: 2
:caption: Data Synthesis
:hidden:

synth_config
synth_attributes
synth_transformations
```

The `oumi synth` command enables you to generate synthetic datasets using large language models. Instead of manually creating training data, you can define rules and templates that automatically generate diverse, high-quality examples.

:::{versionadded} 0.5.0
Advanced synthesis features including new transformation strategies and attribute types.
:::

## What You Can Build

- **Question-Answer datasets** for training chatbots
- **Instruction-following datasets** with varied complexity levels
- **Domain-specific training data** (legal, medical, technical)
- **Conversation datasets** with different personas or styles
- **Data augmentation** to expand existing small datasets

## How It Works

The synthesis process follows three steps:

1. **Define attributes** - What varies in your data (topic, difficulty, style, etc.)
2. **Create templates** - How the AI should generate content using those attributes
3. **Generate samples** - The system creates many examples by combining different attribute values

## Quick Start

Let's create a simple question-answer dataset. Save this as `my_first_synth.yaml`:

```yaml
# Generate 10 geography questions
strategy: GENERAL
num_samples: 10
output_path: geography_qa.jsonl

strategy_params:
  # Give the AI an example to learn from
  input_examples:
    - examples:
      - example_question: "What is the capital of France?"

  # Define what should vary across examples
  sampled_attributes:
    - id: difficulty
      name: Difficulty Level
      description: How challenging the question should be
      possible_values:
        - id: easy
          name: Easy
          description: Basic facts everyone should know
        - id: hard
          name: Hard
          description: Detailed knowledge for experts

  # Tell the AI how to generate questions and answers
  generated_attributes:
    - id: question
      instruction_messages:
        - role: SYSTEM
          content: "You are a geography teacher creating quiz questions. Example: {example_question}"
        - role: USER
          content: "Create a {difficulty} geography question. Write the question only, not the answer."
    - id: answer
      instruction_messages:
        - role: SYSTEM
          content: "You are a helpful AI assistant."
        - role: USER
          content: "{question}"

# Configure which AI model to use
inference_config:
  model:
    model_name: claude-3-5-sonnet-20240620
  engine: ANTHROPIC
```

::::{dropdown} Alternative: Using a local model (no API key required)
If you don't have an API key, you can use a local model instead:

```yaml
inference_config:
  model:
    model_name: HuggingFaceTB/SmolLM2-1.7B-Instruct
  engine: NATIVE_TEXT
  generation:
    max_new_tokens: 256
    temperature: 0.7
```

Or use OpenAI-compatible APIs:

```yaml
inference_config:
  model:
    model_name: gpt-4o-mini
  engine: OPENAI
```
::::

Run it with:

```bash
oumi synth -c my_first_synth.yaml
```

**What happens:** The system will create 10 geography questions, some easy and some hard, saved to `geography_qa.jsonl`.

## Understanding the Results

After running synthesis, you'll see:

- A preview table showing the first few generated samples
- The total number of samples created
- Instructions for using the dataset in training

Each line in the output file contains one example:

```json
{"difficulty": "easy", "question": "What is the largest continent?", "answer": "Asia"}
{"difficulty": "hard", "question": "Which country has the most time zones?", "answer": "France"}
```

## Command Line Options

The `oumi synth` command supports these options:

- `--config`, `-c`: Path to synthesis configuration file (required)
- `--level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

You can also use CLI overrides to modify configuration parameters:

```bash
oumi synth -c config.yaml \
  --num_samples 50 \
  --inference_config.generation.temperature 0.5 \
  --strategy_params.sampled_attributes[0].possible_values[0].sample_rate 0.8
```

## Output Format

The synthesized dataset is saved as a JSONL file where each line contains a JSON object with the attributes in the config:

```json
{"difficulty": "easy", "topic": "geography", "question": "What is the capital of France?", "answer": "Paris"}
{"difficulty": "medium", "topic": "history", "question": "When did World War II end?", "answer": "World War II ended in 1945"}
```

After synthesis completes, you'll see a preview table and instructions on how to use the generated dataset for training:

```
Successfully synthesized 100 samples and saved to synthetic_qa_dataset.jsonl

To train a model, run: oumi train -c path/to/your/train/config.yaml

If you included a 'conversation' chat attribute in your config, update the
config to use your new dataset:
data:
  train:
    datasets:
      - dataset_name: "text_sft_jsonl"
        dataset_path: "synthetic_qa_dataset.jsonl"
```

## Best Practices

1. **Start Small**: Begin with a small `num_samples` to test your configuration
2. **Use Examples**: Provide good examples in `input_examples` for better generation quality
3. **Postprocess Outputs**: Use postprocessing to clean and format generated text
4. **Monitor Costs**: Be aware of API costs when using commercial models
5. **Validate Results**: Review generated samples before using for training
6. **Version Control**: Keep your synthesis configs in version control

## Common Use Cases

### Question-Answer Generation

Generate QA pairs from documents or contexts for training conversational models.

**Example**: See {gh}`configs/examples/synthesis/question_answer_synth.yaml` for a complete geography Q&A generation example.

### Data Augmentation

Create variations of existing datasets by sampling different attributes and regenerating content.

**Example**: See {gh}`configs/examples/synthesis/data_augmentation_synth.yaml` for an example that augments existing datasets with different styles and complexity levels.

### Instruction Following

Generate instruction-response pairs with varying complexity and domains.

**Example**: See {gh}`configs/examples/synthesis/instruction_following_synth.yaml` for a multi-domain instruction generation example covering writing, coding, analysis, and more.

### Conversation Synthesis

Create multi-turn conversations by chaining generated responses.

**Example**: See {gh}`configs/examples/synthesis/conversation_synth.yaml` for a customer support conversation generation example.

### Domain Adaptation

Generate domain-specific training data by conditioning on domain attributes.

**Example**: See {gh}`configs/examples/synthesis/domain_qa_synth.yaml` for a medical domain Q&A generation example with specialty-specific content.

## Troubleshooting

**Empty results**: Check that your instruction messages are well-formed and you have proper API access.

**Slow generation**: Increase `num_workers` or lower `politeness_policy` to improve throughput.

**Out of memory**: Use a smaller model or reduce `max_new_tokens` in generation config.

**Validation errors**: Ensure all attribute IDs are unique and required fields are not empty.

For more help, see the {doc}`/faq/troubleshooting` or report issues at https://github.com/oumi-ai/oumi/issues.

## See Also

- {doc}`synth_config` - Complete configuration reference
- {doc}`synth_attributes` - Detailed attribute configuration
- {doc}`synth_transformations` - Transformation strategies
- {doc}`/user_guides/train/train` - Training with synthesized data
- {doc}`/user_guides/infer/infer` - Inference engines for synthesis
- {doc}`/cli/commands` - CLI reference
