# Synthesis Configuration Reference

This page provides a complete reference for all synthesis configuration options.

## Top-Level Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy` | string | The synthesis strategy to use (currently only `GENERAL` is supported) |
| `num_samples` | int | Number of synthetic samples to generate |
| `output_path` | string | Path where the generated dataset will be saved (must end with `.jsonl`) |
| `strategy_params` | object | Parameters specific to the synthesis strategy |
| `inference_config` | object | Configuration for the model used in generation |

## Strategy Parameters

The `strategy_params` section defines the core synthesis logic.

### Input Sources

You can provide data from multiple sources:

#### input_data

Existing datasets to sample from:

```yaml
input_data:
  - path: "hf:dataset_name"  # HuggingFace dataset
    hf_split: train
  - path: "/path/to/local/data.jsonl"  # Local file
    attribute_map:
      old_column_name: new_attribute_name
```

#### input_documents

Documents to segment and use in synthesis:

```yaml
input_documents:
  - path: "/path/to/document.pdf"
    id: my_doc
    segmentation_params:
      id: doc_segment
      segment_length: 2048
      segment_overlap: 200
```

#### input_examples

Inline examples for few-shot learning:

```yaml
input_examples:
  - examples:
    - attribute1: "value1"
      attribute2: "value2"
    - attribute1: "value3"
      attribute2: "value4"
```

### Advanced Features

#### Combination Sampling

Control probability of specific attribute combinations:

```yaml
combination_sampling:
  - combination:
      difficulty: hard
      topic: science
    sample_rate: 0.1  # 10% of samples will have hard science questions
```

#### Passthrough Attributes

Specify which attributes to include in final output:

```yaml
passthrough_attributes:
  - question
  - answer
  - difficulty
  - topic
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

| Parameter | Description |
|-----------|-------------|
| `segmentation_strategy` | How to split documents (`TOKENS`, `CHARACTERS`, `SENTENCES`) |
| `tokenizer` | Tokenizer to use for token-based segmentation |
| `segment_length` | Maximum length of each segment |
| `segment_overlap` | Overlap between consecutive segments |
| `keep_original_text` | Whether to preserve original text alongside segments |

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

| Engine | Description | Requirements |
|--------|-------------|--------------|
| `ANTHROPIC` | Claude models | `ANTHROPIC_API_KEY` |
| `OPENAI` | OpenAI models | `OPENAI_API_KEY` |
| `VLLM` | Local vLLM inference server | vLLM installation |
| `SGLANG` | Local SGLang inference server | SGLang installation |
| `NATIVE_TEXT` | Local HuggingFace transformers | Model weights |
| `TOGETHER` | Together AI hosted models | `TOGETHER_API_KEY` |
| `PARASAIL` | Parasail hosted models | `PARASAIL_API_KEY` |

For more inference engine options, see {doc}`/user_guides/infer/inference_engines`.

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 1024 | Maximum tokens to generate |
| `temperature` | 0.7 | Sampling temperature (0 = deterministic) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | - | Top-k sampling (optional) |
| `stop_sequences` | - | Sequences that stop generation |

### Remote Parameters

For API-based engines:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_workers` | 1 | Number of concurrent API requests |
| `politeness_policy` | 0 | Delay between requests (seconds) |
| `retry_attempts` | 3 | Number of retry attempts on failure |
| `timeout` | 60 | Request timeout (seconds) |

## Complete Example

```yaml
strategy: GENERAL
num_samples: 100
output_path: synthetic_dataset.jsonl

strategy_params:
  input_examples:
    - examples:
      - example_question: "What is the capital of France?"
        example_answer: "Paris"

  sampled_attributes:
    - id: topic
      name: Topic
      description: The subject area of the question
      possible_values:
        - id: geography
          name: Geography
          description: Questions about places and locations
          sample_rate: 0.4
        - id: history
          name: History
          description: Questions about historical events
          sample_rate: 0.3
        - id: science
          name: Science
          description: Questions about scientific concepts
          sample_rate: 0.3

    - id: difficulty
      name: Difficulty Level
      description: How challenging the question should be
      possible_values:
        - id: easy
          name: Easy
          description: Basic facts
        - id: hard
          name: Hard
          description: Advanced knowledge

  generated_attributes:
    - id: question
      instruction_messages:
        - role: SYSTEM
          content: "Generate educational questions."
        - role: USER
          content: "Create a {difficulty} {topic} question."
    - id: answer
      instruction_messages:
        - role: SYSTEM
          content: "You are a helpful assistant."
        - role: USER
          content: "{question}"

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
    - topic
    - difficulty

inference_config:
  model:
    model_name: "gpt-4o-mini"
  engine: OPENAI
  generation:
    max_new_tokens: 512
    temperature: 0.8
  remote_params:
    num_workers: 3
    politeness_policy: 1
```

## See Also

- {doc}`synth_attributes` - Detailed attribute configuration
- {doc}`synth_transformations` - Transformation strategies
