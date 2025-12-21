# Oumi Simplified Python API

This document describes the simplified API layer for Oumi that reduces boilerplate while maintaining backward compatibility.

## Overview

The simplified API provides one-liner interfaces for the most common operations:
- **Inference**: `chat()` for quick model interactions
- **Training**: `train()` with sensible defaults
- **Evaluation**: `evaluate()` with task lists
- **Judging**: `judge()` with predefined criteria

## Quick Start

```python
from oumi import chat, train, evaluate
from oumi.judge import judge

# Chat with any model (provider auto-detected)
response = chat("gpt-4o", "What is machine learning?")

# Train a model with one line
train("meta-llama/Llama-3.1-8B", "tatsu-lab/alpaca")

# Evaluate on benchmarks
results = evaluate("meta-llama/Llama-3.1-8B", tasks=["mmlu", "hellaswag"])

# Judge model outputs
results = judge("gpt-4o", dataset, criteria="truthfulness")
```

## API Reference

### `chat()` - Simple Inference

```python
from oumi import chat

# Basic usage
response = chat("gpt-4o", "Hello!")

# With parameters
response = chat("claude-3-opus", "Explain AI", temperature=0.7, max_tokens=500)

# Multi-turn with dict messages (OpenAI format)
response = chat("gpt-4o", messages=[
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
])

# Continue a conversation
conv = chat("gpt-4o", "Hi!", return_conversation=True)
response = chat("gpt-4o", "Tell me more", conversation=conv)

# From YAML config
response = chat("my_config.yaml", "Hello")

# Explicit provider prefix
response = chat("anthropic/claude-3-opus", "Hello")
```

**Provider Auto-Detection:**
- `gpt-4o`, `gpt-4-turbo`, `o1-preview` → OpenAI
- `claude-3-opus`, `claude-3.5-sonnet` → Anthropic
- `gemini-pro`, `gemini-1.5-flash` → Google
- `meta-llama/Llama-3.1-8B` → vLLM (HuggingFace models)
- Explicit: `openai/gpt-4o`, `anthropic/claude-3`, `together/model`

### `train()` - Simple Training

```python
from oumi import train

# SFT training (default)
train("meta-llama/Llama-3.1-8B", "tatsu-lab/alpaca")

# DPO training
train("meta-llama/Llama-3.1-8B", "my-preference-dataset", method="dpo")

# With custom parameters
train(
    "meta-llama/Llama-3.1-8B",
    "tatsu-lab/alpaca",
    method="sft",
    epochs=5,
    batch_size=8,
    learning_rate=1e-5,
    use_peft=True,
    lora_r=32,
)

# From YAML config (full control)
train("training_config.yaml")

# From TrainingConfig object
from oumi.core.configs import TrainingConfig
train(TrainingConfig(...))
```

**Supported Training Methods:**
- `sft` - Supervised Fine-Tuning (default)
- `dpo` - Direct Preference Optimization
- `kto` - Kahneman-Tversky Optimization
- `grpo` - Group Relative Policy Optimization
- `gkd` - Generalized Knowledge Distillation
- `gold` - GOLD Training

### `evaluate()` - Simple Evaluation

```python
from oumi import evaluate

# Basic evaluation
results = evaluate("meta-llama/Llama-3.1-8B", tasks=["mmlu"])

# Multiple tasks with sampling
results = evaluate(
    "gpt-4o",
    tasks=["mmlu", "hellaswag", "arc_easy"],
    num_samples=100,
    batch_size=16,
)

# From YAML config
results = evaluate("eval_config.yaml")
```

### `judge()` - LLM-as-Judge

```python
from oumi.judge import judge

# Dataset format
dataset = [
    {"request": "What is 2+2?", "response": "4"},
    {"request": "Capital of France?", "response": "Paris"},
]

# Predefined criteria
results = judge("gpt-4o", dataset, criteria="truthfulness")
results = judge("gpt-4o", dataset, criteria="helpfulness")
results = judge("gpt-4o", dataset, criteria="safety")
results = judge("gpt-4o", dataset, criteria="relevance")
results = judge("gpt-4o", dataset, criteria="coherence")

# Custom prompt template
results = judge(
    "claude-3-opus",
    dataset,
    prompt_template="Is this response accurate? Q: {request} A: {response}",
    judgment_type="bool",
)

# From config file
results = judge("judge_config.yaml", dataset)
```

**Predefined Criteria:**
| Criteria | Output Type | Description |
|----------|-------------|-------------|
| `truthfulness` | bool | Is the response factually accurate? |
| `helpfulness` | int (1-5) | How helpful is the response? |
| `safety` | bool | Is the response safe and appropriate? |
| `relevance` | bool | Does the response address the question? |
| `coherence` | bool | Is the response logical and well-structured? |

## TrainingConfig Factory Methods

For intermediate users who want more control than the simple API but less boilerplate than full configs:

```python
from oumi.core.configs import TrainingConfig

# Create SFT config
config = TrainingConfig.for_sft(
    model="meta-llama/Llama-3.1-8B",
    dataset="tatsu-lab/alpaca",
    epochs=3,
    batch_size=4,
)

# Create DPO config
config = TrainingConfig.for_dpo(
    model="meta-llama/Llama-3.1-8B",
    dataset="my-preference-dataset",
    learning_rate=1e-5,
)

# Create GRPO config
config = TrainingConfig.for_grpo(
    model="meta-llama/Llama-3.1-8B",
    dataset="my-grpo-dataset",
)

# Generic factory method
config = TrainingConfig.for_method(
    method="kto",
    model="meta-llama/Llama-3.1-8B",
    dataset="my-dataset",
    use_peft=True,
    lora_r=32,
    lora_alpha=64,
)
```

## Backward Compatibility

All existing code continues to work unchanged:

```python
# These still work exactly as before
from oumi import train, evaluate, infer
from oumi.core.configs import TrainingConfig, EvaluationConfig, InferenceConfig

config = TrainingConfig(...)
train(config)

config = EvaluationConfig(...)
evaluate(config)

config = InferenceConfig(...)
infer(config, inputs=["Hello"])
```

## Implementation Details

### Files Created/Modified

| File | Description |
|------|-------------|
| `src/oumi/utils/provider_detection.py` | Provider auto-detection logic |
| `src/oumi/infer.py` | Added `chat()` with engine caching |
| `src/oumi/train.py` | Added simple mode overloads |
| `src/oumi/evaluate.py` | Added simple mode overloads |
| `src/oumi/judge.py` | Added `judge()` with criteria templates |
| `src/oumi/core/configs/training_config.py` | Added factory methods |
| `src/oumi/__init__.py` | Updated exports |

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Engine caching | Enabled by default | Avoid re-initialization overhead |
| Default training method | SFT | 80% use case |
| Default PEFT | Enabled (LoRA) | Memory efficient |
| Provider detection | Hybrid (prefix + auto) | Explicit when needed, convenient otherwise |
| Error messages | User-friendly suggestions | Help users fix common mistakes |

### Tests

- `tests/unit/test_provider_detection.py` - 17 tests for provider detection
- `tests/unit/test_simple_api.py` - 24 tests for simplified API functions
