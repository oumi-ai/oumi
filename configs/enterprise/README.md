# Enterprise SFT Training Pipeline

This directory contains configurations for enterprise SFT experiments to establish hardened training configurations for small post-trained models.

## Quick Start

### 1. Prepare Datasets

```bash
# Prepare all datasets (banking77, pubmedqa, tatqa, nl2sql)
python scripts/enterprise/prepare_datasets.py --all --output-dir data/enterprise

# Or prepare individual datasets
python scripts/enterprise/prepare_datasets.py --task banking77 --output-dir data/enterprise
python scripts/enterprise/prepare_datasets.py --task pubmedqa --output-dir data/enterprise
python scripts/enterprise/prepare_datasets.py --task tatqa --output-dir data/enterprise
python scripts/enterprise/prepare_datasets.py --task nl2sql --output-dir data/enterprise
```

### 2. Run Baseline Evaluations

```bash
# Evaluate baseline model on task
oumi evaluate -c configs/enterprise/evaluation/task_banking77.yaml \
  --model.model_name "Qwen/Qwen3-4B-Instruct"

oumi evaluate -c configs/enterprise/evaluation/task_tatqa.yaml \
  --model.model_name "Qwen/Qwen3-4B-Instruct"

# Evaluate control metrics (IFEval + SimpleSafetyTests)
oumi evaluate -c configs/enterprise/evaluation/control_evals.yaml \
  --model.model_name "Qwen/Qwen3-4B-Instruct"
```

### 3. Run Hyperparameter Tuning

```bash
# Pilot tuning experiments (2 models x 2 tasks)
oumi tune -c configs/enterprise/tuning/qwen3_4b_banking77.yaml
oumi tune -c configs/enterprise/tuning/qwen3_4b_tatqa.yaml
oumi tune -c configs/enterprise/tuning/llama31_8b_banking77.yaml
oumi tune -c configs/enterprise/tuning/llama31_8b_tatqa.yaml
```

### 4. Evaluate Fine-tuned Models

```bash
# Evaluate best checkpoint on task
oumi evaluate -c configs/enterprise/evaluation/task_banking77.yaml \
  --model.model_name "output/enterprise/tuning/qwen3_4b_banking77/trial_0"

# Verify control metrics maintained
oumi evaluate -c configs/enterprise/evaluation/control_evals.yaml \
  --model.model_name "output/enterprise/tuning/qwen3_4b_banking77/trial_0"
```

## Directory Structure

```
configs/enterprise/
├── README.md                           # This file
├── tuning/
│   ├── qwen3_4b_banking77.yaml        # Qwen3-4B on Banking77
│   ├── qwen3_4b_tatqa.yaml            # Qwen3-4B on TAT-QA
│   ├── llama31_8b_banking77.yaml      # Llama-3.1-8B on Banking77
│   └── llama31_8b_tatqa.yaml          # Llama-3.1-8B on TAT-QA
└── evaluation/
    ├── control_evals.yaml             # IFEval + SimpleSafetyTests
    ├── task_banking77.yaml            # Banking77 evaluation
    └── task_tatqa.yaml                # TAT-QA evaluation

data/enterprise/
├── banking77/
│   ├── train.jsonl                    # 10,003 samples
│   ├── test.jsonl                     # 3,080 samples
│   └── labels.json                    # 77 class labels
├── pubmedqa/
│   ├── train.jsonl                    # 900 samples
│   └── test.jsonl                     # 100 samples
├── tatqa/
│   ├── train.jsonl                    # 13,196 samples
│   └── test.jsonl                     # 1,639 samples
└── nl2sql/
    ├── train.jsonl                    # 587 samples
    └── test.jsonl                     # 66 samples

src/oumi/evaluation/registry/enterprise/
├── __init__.py
├── classification.py                  # enterprise_banking77, enterprise_pubmedqa
├── tatqa.py                          # enterprise_tatqa
├── nl2sql.py                         # enterprise_nl2sql
└── simple_safety_tests.py            # simple_safety_tests
```

## Models

| Model | Config | Notes |
|-------|--------|-------|
| Qwen3-4B-Instruct | `configs/recipes/qwen3/sft/4b_full/train.yaml` | Full fine-tuning |
| Llama-3.1-8B-Instruct | `configs/recipes/llama3_1/sft/8b_full/train.yaml` | Full fine-tuning |
| Llama-3.2-1B-Instruct | `configs/recipes/llama3_2/sft/1b_full/train.yaml` | Full fine-tuning |
| Gemma-3-4B-it | `configs/recipes/gemma3/sft/4b_full/train.yaml` | Substituting Gemma 2 9B |

## Tasks

| Task | Dataset | Metric | Description |
|------|---------|--------|-------------|
| banking77 | `legacy-datasets/banking77` | Accuracy | 77-class customer query classification |
| pubmedqa | `qiaojin/PubMedQA` | Accuracy | 3-class medical QA (yes/no/maybe) |
| tatqa | `next-tat/TAT-QA` | Exact Match / F1 | Tabular + textual QA |
| nl2sql | `NovaSky-AI/SkyRL-SQL-653-data` | Edit Distance | Text-to-SQL |

## Hyperparameter Search Space

The tuning configs search over:

- Learning rate: [1e-5, 2e-5, 5e-5]
- Epochs: [1, 3]
- Batch size: [2, 4]

Fixed parameters:

- Gradient accumulation: 4
- Optimizer: adamw_torch_fused
- LR scheduler: cosine
- Warmup ratio: 0.03
- Weight decay: 0.01

## Constraints

Post-SFT models should:

1. **IFEval**: Stay within 5% of baseline accuracy
2. **SimpleSafetyTests**: Maintain >90% safe response rate

## Reproduction Log

Commands run to set up this pipeline:

```bash
# 1. Create directories
mkdir -p scripts/enterprise
mkdir -p src/oumi/evaluation/registry/enterprise
mkdir -p configs/enterprise/tuning
mkdir -p configs/enterprise/evaluation
mkdir -p data/enterprise

# 2. Prepare datasets
python scripts/enterprise/prepare_datasets.py --all --output-dir data/enterprise

# Dataset preparation output:
# - Banking77: 10,003 train, 3,080 test (77 classes)
# - PubMedQA: 900 train, 100 test (3 classes)
# - TAT-QA: 13,196 train, 1,639 test
# - NL2SQL: 587 train, 66 test
```
