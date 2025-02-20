# Inference CLI

## Overview

The Oumi CLI provides a simple interface for running inference tasks. The main command is `oumi infer`,
which supports both interactive chat and batch processing modes. The interactive mode lets you send text inputs
directly from your terminal, while the batch mode lets you submit a jsonl file of conversations for batch processing.

To use the CLI you need an {py:obj}`~oumi.core.configs.InferenceConfig`. This config
will specify which model and inference engine you're using, as well as any relevant
inference-time variables - see {doc}`/user_guides/infer/configuration` for more details.

```{seealso}
Check out our [Infer CLI definition](/cli/commands.md#inference) to see the full list of command line options.
```

## Basic Usage

```bash
# Interactive chat
oumi infer -i -c config.yaml

# Process input file
oumi infer -c config.yaml --input_path input.jsonl --output_path output.jsonl
```

## Command Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `-c`, `--config` | Configuration file path | Required | `-c config.yaml` |
| `-i`, `--interactive` | Enable interactive mode | False | `-i` |
| `--input_path` | Input JSONL file path | None | `--input_path data.jsonl` |
| `--output_path` | Output JSONL file path | None | `-output_path results.jsonl` |
| `--model.device_map` | GPU device(s) | "cuda" | `--model.device_map "cuda:0"` |
| `--model.model_name` | Model name | None | `--model.model_name "HuggingFaceTB/SmolLM2-135M-Instruct"` |
| `--generation.seed` | Random seed | None | `--seed 42` |
| `--log-level` | Logging level | INFO | `--log-level DEBUG` |

## Configuration File

Example `config.yaml`:

```yaml
model:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  model_kwargs:
    device_map: "auto"
    torch_dtype: "float16"

generation:
  max_new_tokens: 100
  temperature: 0.7
  top_p: 0.9
  batch_size: 1

engine: "VLLM"
```

## Common Usage Patterns

### Interactive Chat

```bash
# Basic chat
oumi infer -i -c configs/chat.yaml

# Chat with specific GPU
oumi infer -i -c configs/chat.yaml --model.device_map cuda:0
```

### Batch Processing

```bash
# Process dataset
oumi infer -c configs/batch.yaml \
  --input_path dataset.jsonl \
  --output_path results.jsonl \
  --generation.batch_size 32
```

### Multi-GPU Inference

```bash
# Use specific GPUs
oumi infer -c configs/multi_gpu.yaml \
  --model.device_map "cuda:0,cuda:1"

# Tensor parallel inference
oumi infer -c configs/multi_gpu.yaml \
  --model.model_kwargs.tensor_parallel_size 4
```

## Input/Output Formats

### Input JSONL

```json
{"messages": [{"role": "user", "content": "Hello!"}]}
{"messages": [{"role": "user", "content": "How are you?"}]}
```

### Output JSONL

```json
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi!"}]}
{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm good!"}]}
```

## See Also

- {doc}`configuration` for config file options
- {doc}`common_workflows` for usage examples
