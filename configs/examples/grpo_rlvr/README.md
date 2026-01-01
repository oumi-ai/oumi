# Rubrics as Rewards (RaR) Implementation

This directory contains an implementation of the **Rubrics as Rewards (RaR)** methodology from the paper:

> **"Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains"**
> Anisha Gunjal et al. (Scale AI)
> arXiv:2507.17746

## Overview

RaR is a framework that uses structured, checklist-style rubrics as interpretable reward signals for on-policy RL training with GRPO. It enables training language models in domains lacking single ground-truth answers (e.g., medical reasoning, scientific explanation).

### Key Features

- **Weighted Rubrics**: Each criterion has an importance level (Essential, Important, Optional, Pitfall)
- **Pitfall Support**: Negative weights penalize responses that hit common mistakes
- **LLM-as-Judge**: Uses GPT-4o-mini to evaluate responses against rubrics
- **Judge Panels**: Multiple judges for robust evaluation (optional)

## Quick Start

### 1. Run a Small Test

```bash
# Requires: OPENAI_API_KEY environment variable
oumi train -c configs/examples/grpo_rlvr/train_rar_medicine_small.yaml
```

### 2. Explore the Dataset

```bash
python scripts/rar_demo.py --dataset medicine
```

### 3. Generate Rubrics for Custom Data

```bash
python scripts/rar_rubric_generator.py --prompt "What causes Type 2 diabetes?"
```

## Datasets

The implementation uses the official RaR datasets from HuggingFace:

| Dataset | Size | Domain | HuggingFace |
|---------|------|--------|-------------|
| RaR-Medicine | 22.4k | Medical reasoning | [anisha2102/RaR-Medicine](https://huggingface.co/datasets/anisha2102/RaR-Medicine) |
| RaR-Science | 22.9k | Scientific problems | [anisha2102/RaR-Science](https://huggingface.co/datasets/anisha2102/RaR-Science) |

### Rubric Structure

Each sample contains 7-17 rubrics with this structure:

```json
{
  "title": "Identify Most Sensitive Modality",
  "description": "Essential Criteria: Identifies non-contrast helical CT scan as the most sensitive modality for ureteric stones.",
  "weight": 5
}
```

### Weight Categories

| Category | Weight | Meaning |
|----------|--------|---------|
| Essential | 5 | Core requirements (must be satisfied) |
| Important | 3-4 | Significant points (should be included) |
| Optional | 1-2 | Helpful additions |
| Pitfall | -1 to -2 | Common mistakes to avoid |

## Training Configs

### Full Paper Reproduction

```bash
# Requires 8x H100 GPUs or equivalent (~40GB+ VRAM per GPU)
oumi train -c configs/examples/grpo_rlvr/train_rar_medicine.yaml
```

**Hyperparameters (from paper):**

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-7B-Instruct |
| num_generations (k) | 16 |
| Batch size | 96 |
| Learning rate | 5×10⁻⁶ |
| Temperature | 1.0 |
| Training steps | 300 |
| Warmup | 10% |
| Context length | 3584 |

### Smaller Test Configuration

```bash
# Single 24GB GPU
oumi train -c configs/examples/grpo_rlvr/train_rar_medicine_small.yaml
```

Uses Qwen2.5-1.5B with k=4 generations for faster iteration.

### Science Domain

```bash
oumi train -c configs/examples/grpo_rlvr/train_rar_science.yaml
```

## Components

### 1. Dataset Loaders

```python
from oumi.datasets import RaRMedicineDataset, RaRScienceDataset

# Load from HuggingFace
dataset = RaRMedicineDataset(split="train")
sample = dataset[0]

print(sample["prompt"])   # The question
print(sample["rubrics"])  # List of weighted rubric dicts
```

### 2. Reward Function

The `rubric_reward` function evaluates completions against rubrics:

```python
from oumi.datasets.grpo.rewards.rubric_reward import rubric_reward

rewards = rubric_reward(
    completions=completions,
    prompts=prompts,
    rubrics=rubrics,
)
```

By default, `rubric_reward` evaluates each rubric independently (one judge call per
rubric) and aggregates the results. To have a single judge evaluate all rubrics in
one call, pass `group_rubrics=True`:

```python
rewards = rubric_reward(
    completions=completions,
    prompts=prompts,
    rubrics=rubrics,
    group_rubrics=True,
)
```

**Pitfall Handling:**
- Pitfalls are phrased positively (e.g., "Avoids hallucination")
- Score=1 means pitfall was avoided (good) → positive contribution
- Score=0 means pitfall was hit (bad) → negative contribution

### 3. Rubric Generation

Generate rubrics for your own data:

```bash
# Single prompt
python scripts/rar_rubric_generator.py \
    --prompt "Explain photosynthesis" \
    --reference "Photosynthesis is the process by which plants..."

# Batch processing
python scripts/rar_rubric_generator.py \
    --input questions.jsonl \
    --output questions_with_rubrics.jsonl \
    --model gpt-4o
```

## File Structure

```
configs/examples/grpo_rlvr/
├── README.md                      # This file
├── train_rar_medicine.yaml        # Full paper reproduction (medicine)
├── train_rar_medicine_small.yaml  # Small test config (medicine)
├── train_rar_science.yaml         # Science domain config
├── train.yaml                     # Original simple rubric config
├── train_weighted.yaml            # Weighted rubric config
└── sample_data*.jsonl             # Sample data files

src/oumi/datasets/grpo/
├── rar_dataset.py                 # RaR dataset loaders
├── rlvr_rubric.py                 # Generic rubric dataset
└── rewards/
    └── rubric_reward.py           # Rubric-based reward function

scripts/
├── rar_demo.py                    # Demo script
└── rar_rubric_generator.py        # Rubric generation pipeline
```

## Environment Setup

```bash
# Required
export OPENAI_API_KEY="your-api-key"

# Optional: WandB logging
wandb login
```

## Reproducing Paper Results

The paper reports improvements on:
- **HealthBench**: Up to 31% relative improvement over LLM-as-judge baselines
- **GPQA-Diamond**: Up to 7% relative improvement

To reproduce:

1. Train on RaR-Medicine with full config
2. Evaluate on HealthBench using the trained checkpoint
3. Compare against baseline (direct Likert-based rewards)

## References

- **Paper**: [arXiv:2507.17746](https://arxiv.org/abs/2507.17746)
- **HuggingFace**: [Paper Page](https://huggingface.co/papers/2507.17746)
- **Scale AI Blog**: [Rubrics as Rewards](https://scale.com/blog/rubrics-as-rewards)
- **Datasets**:
  - [RaR-Medicine](https://huggingface.co/datasets/anisha2102/RaR-Medicine)
  - [RaR-Science](https://huggingface.co/datasets/anisha2102/RaR-Science)
