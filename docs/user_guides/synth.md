# Data Synthesis

The `oumi synth` command enables you to generate synthetic datasets using large language models. Instead of manually creating training data, you can define rules and templates that automatically generate diverse, high-quality examples.

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

## Your First Synthesis

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

## Next Steps: Building More Complex Datasets

Once you're comfortable with the basics, you can create more sophisticated datasets:

### Adding Multiple Attributes

Mix and match different properties (topic + difficulty + style):

```yaml
sampled_attributes:
  - id: topic
    possible_values: [{id: geography}, {id: history}, {id: science}]
  - id: difficulty
    possible_values: [{id: easy}, {id: medium}, {id: hard}]
  - id: style
    possible_values: [{id: formal}, {id: casual}, {id: academic}]
```

### Using Your Own Data

Feed in existing datasets or documents:

```yaml
input_data:
  - path: "my_existing_data.jsonl"
input_documents:
  - path: "textbook.pdf"
```

### Creating Conversations

Build multi-turn dialogues:

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

Ready to dive deeper? The sections below cover all available options in detail.

---

## Complete Configuration Reference

### Top-Level Parameters

- **`strategy`**: The synthesis strategy to use (currently only `GENERAL` is supported)
- **`num_samples`**: Number of synthetic samples to generate
- **`output_path`**: Path where the generated dataset will be saved (must end with `.jsonl`)
- **`strategy_params`**: Parameters specific to the synthesis strategy
- **`inference_config`**: Configuration for the model used in generation

### Strategy Parameters

The `strategy_params` section defines the core synthesis logic:

#### Input Sources

You can provide data from multiple sources:

**`input_data`**: Existing datasets to sample from

```yaml
input_data:
  - path: "hf:dataset_name"  # HuggingFace dataset
    hf_split: train
  - path: "/path/to/local/data.jsonl"  # Local file
    attribute_map:
      old_column_name: new_attribute_name
```

**`input_documents`**: Documents to segment and use in synthesis

```yaml
input_documents:
  - path: "/path/to/document.pdf"
    id: my_doc
    segmentation_params:
      id: doc_segment
      segment_length: 2048
      segment_overlap: 200
```

**`input_examples`**: Inline examples for few-shot learning

```yaml
input_examples:
  - examples:
    - attribute1: "value1"
      attribute2: "value2"
    - attribute1: "value3"
      attribute2: "value4"
```

#### Attribute Types

**Sampled Attributes**: Randomly selected values from predefined options

```yaml
sampled_attributes:
  - id: difficulty
    name: Difficulty Level
    description: How challenging the question should be
    possible_values:
      - id: easy
        name: Easy
        description: Simple, straightforward questions
        sample_rate: 0.4  # 40% of samples
      - id: medium
        name: Medium
        description: Moderately challenging questions
        sample_rate: 0.4  # 40% of samples
      - id: hard
        name: Hard
        description: Complex, advanced questions
        # No sample_rate specified = 20% (remaining)
```

**Generated Attributes**: Created by LLM using instruction messages

```yaml
generated_attributes:
  - id: summary
    instruction_messages:
      - role: SYSTEM
        content: "You are a helpful summarization assistant."
      - role: USER
        content: "Summarize this text: {input_text}. Format your result as 'Summary: <summary>'"
    postprocessing_params:
      id: clean_summary
      cut_prefix: "Summary: "
      strip_whitespace: true
```

**Transformed Attributes**: Rule-based transformations of existing attributes

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

#### Advanced Features

**Combination Sampling**: Control probability of specific attribute combinations

```yaml
combination_sampling:
  - combination:
      difficulty: hard
      topic: science
    sample_rate: 0.1  # 10% of samples will have hard science questions
```

**Passthrough Attributes**: Specify which attributes to include in final output

```yaml
passthrough_attributes:
  - question
  - answer
  - difficulty
  - topic
```

## Attribute Referencing

In instruction messages and transformations, you can reference attributes using `{attribute_id}` syntax:

- `{attribute_id}`: The value/name of the attribute
- `{attribute_id.description}`: The description of a sampled attribute value
- `{attribute_id.parent}`: The parent name of a sampled attribute
- `{attribute_id.parent.description}`: The parent description of a sampled attribute

## Postprocessing

Generated attributes can be postprocessed to clean up the output:

```yaml
postprocessing_params:
  id: cleaned_attribute
  keep_original_text_attribute: true  # Keep original alongside cleaned version
  cut_prefix: "Answer: "  # Remove this prefix and everything before it
  cut_suffix: "\n\n"      # Remove this suffix and everything after it
  regex: "\\*\\*(.+?)\\*\\*"  # Extract content between ** **
  strip_whitespace: true  # Remove leading/trailing whitespace
  added_prefix: "Response: "  # Add this prefix
  added_suffix: "."       # Add this suffix
```

## Transformation Strategies

For the following examples, let's assume we have a data sample with the following values.

```json
{
  "question": "What color is the sky?",
  "answer": "The sky is blue."
}
```

### String Transformation

```yaml
transformed_attributes:
  - id: example_string_attribute
    transformation_strategy:
      type: STRING
      string_transform: "Question: {question}\nAnswer: {answer}"
```

Example Result:

```
{
  "example_string_attribute": "Question: What color is the sky?\nAnswer: The sky is blue."
}
```

### List Transformation

```yaml
transformed_attributes:
  - id: example_list_attribute
    transformation_strategy:
      type: LIST
      list_transform:
        - "{question}"
        - "{answer}"
```

Example Result:

```json
{
  "example_list_attribute": [
    "What color is the sky?",
    "The sky is blue.",
  ]
}
```

### Dictionary Transformation

```yaml
transformed_attributes:
  - id: example_dict_attribute
    transformation_strategy:
      type: DICT
      dict_transform:
        question: "{question}"
        answer: "{answer}"
```

Example Result:

```
{
  "example_list_attribute": {
    "question": "What color is the sky?",
    "answer": "The sky is blue.",
  }
}
```

### Chat Transformation

```yaml
transformed_attributes:
  - id: string_attribute
    transformation_strategy:
      type: CHAT
      chat_transform:
        messages:
          - role: USER
            content: "{question}"
          - role: ASSISTANT
            content: "{answer}"
```

## Document Segmentation

When using documents, you can segment them for processing:

```yaml
input_documents:
  - path: "/path/to/document.pdf"
    id: research_paper
    segmentation_params:
      id: paper_segment
      segmentation_strategy: TOKENS
      tokenizer: "openai-community/gpt2"
      segment_length: 1024
      segment_overlap: 128
      keep_original_text: true
```

## Inference Configuration

Configure the model and generation parameters:

```yaml
inference_config:
  model:
    model_name: "claude-3-5-sonnet-20240620"
  engine: ANTHROPIC
  generation:
    max_new_tokens: 1024
    temperature: 0.7
    top_p: 0.9
  remote_params:
    num_workers: 5
    politeness_policy: 60  # Delay between requests in seconds
```

### Supported Engines

- `ANTHROPIC`: Claude models (requires API key)
- `OPENAI`: OpenAI models (requires API key)
- `VLLM`: Local vLLM inference server
- `NATIVE_TEXT`: Local HuggingFace transformers
- And many more (see {doc}`/user_guides/infer/inference_engines`)

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

For more help, see the [FAQ](../faq/troubleshooting.md) or report issues at https://github.com/oumi-ai/oumi/issues.
