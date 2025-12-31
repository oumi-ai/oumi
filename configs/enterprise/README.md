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
├── test_smollm2_train.yaml            # Test config for training
├── tuning/
│   ├── test_smollm2_banking77.yaml   # Test config for tuning
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

## Testing the Pipeline

Before running full experiments, test with SmolLM2-135M:

```bash
# Test training (quick sanity check)
oumi train -c configs/enterprise/test_smollm2_train.yaml

# Test hyperparameter tuning (2 trials)
oumi tune -c configs/enterprise/tuning/test_smollm2_banking77.yaml

# Test evaluation (10 samples)
oumi evaluate -c configs/enterprise/evaluation/task_banking77.yaml \
  --model.model_name "HuggingFaceTB/SmolLM2-135M-Instruct" \
  --model.model_max_length 2048 \
  --tasks.0.num_samples 10 \
  --enable_wandb false
```

## Key Configuration Details

The tuning configs use `return_conversations: true` with `return_conversations_format: dict`
to let TRL's SFT trainer handle tokenization and label creation:

```yaml
data:
  train:
    datasets:
      - dataset_name: text_sft
        dataset_path: data/enterprise/banking77/train.jsonl
        dataset_kwargs:
          return_conversations: true
          return_conversations_format: dict
```

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

# 3. Test with SmolLM2 (validated on 2025-12-19)
oumi train -c configs/enterprise/test_smollm2_train.yaml  # Success
oumi tune -c configs/enterprise/tuning/test_smollm2_banking77.yaml  # Success (2 trials)
oumi evaluate -c configs/enterprise/evaluation/task_banking77.yaml \
  --model.model_name "HuggingFaceTB/SmolLM2-135M-Instruct" \
  --model.model_max_length 2048 \
  --tasks.0.num_samples 10 \
  --enable_wandb false  # Success
```


---
---
<br><br><br>


## Worklog
The workflows above were generated by Claude Code based on [its plan](https://docs.google.com/document/d/1OjG1xSnC4sBuQKG-jnlMRYwTYpk87AuKKe0obxcs49c/edit?tab=t.0), generated from Tim's [Training Planning for Enterprise Launch](https://docs.google.com/document/d/1Sq5ogeUlI8VomdJ5QO1g1ea3JwU-yaXBpHAYgA9qMiI/edit?tab=t.pqdeirkbycvi#heading=h.306lxpnc5lb0) doc. Oussama's original branch here: https://github.com/oumi-ai/oumi/compare/oelachqar/tunestuff (starting point for the current experiment workflows).

Here we harden and the workflows for building task-specific benchmarks, and then run many training experiments to identify good default training configs and hyperparam ranges for supported student models on the enterprise platform.

This section contains running notes as we progress through experiments.


### Finalize use case datasets and benchmarks

#### 1. Dataprep
Generate formatted train/test/val splits under `data/enterprise/` for each use case.

```sh
cd /Users/tim/workspace/oumi
conda activate oumi

python scripts/enterprise/prepare_datasets.py \
  --output-dir data/enterprise \
  --all
# Preparing 4 task(s): ['banking77', 'pubmedqa', 'tatqa', 'nl2sql']
# ...
# Banking77: 9903 train, 100 val, 3080 test, 77 classes
# PubMedQA: 800 train, 100 val, 100 test
# TAT-QA: 13096 train, 100 val, 1639 test
# NL2SQL: 503 train, 50 val, 100 test
# Dataset preparation complete!
```

#### 2. Baseline evals (local testing to finalize datasets and metrics)
Evaluate a small baseline model on a subset of each task dataset to verify things work mechanically. Task evals use `--model.torch_dtype_str float32` for local testing with `SmolLM2-135M-Instruct` on Mac.

See each task's config for default dataset path and other relevant model and inf params.

Output bundles will be written to the `output_dir` specified in each task's config (e.g. `output_dir: "output/enterprise/evaluation/control"`), unless overridden at the command line. Individual predictions .jsonl files are included in the output bundles along with metrics.

##### banking77
```sh
oumi evaluate -c configs/enterprise/evaluation/task_banking77.yaml \
  --model.model_name "HuggingFaceTB/SmolLM2-135M-Instruct" \
  --model.model_max_length 2048 \
  --tasks.0.num_samples 20 \
  --generation.batch_size 1 \
  --enable_wandb false \
  --model.torch_dtype_str float32
# ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┓
# ┃ Benchmark            ┃ Metric      ┃ Score ┃ Std Error ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━┩
# │ enterprise_banking77 │ Accuracy    │ 0.00% │ -         │
# ├──────────────────────┼─────────────┼───────┼───────────┤
# │ enterprise_banking77 │ Num Correct │ 0     │ -         │
# ├──────────────────────┼─────────────┼───────┼───────────┤
# │ enterprise_banking77 │ Num Total   │ 20    │ -         │
# └──────────────────────┴─────────────┴───────┴───────────┘
```


##### pubmedqa
```sh
oumi evaluate -c configs/enterprise/evaluation/task_pubmedqa.yaml \
  --model.model_name "HuggingFaceTB/SmolLM2-135M-Instruct" \
  --model.model_max_length 2048 \
  --tasks.0.num_samples 20 \
  --generation.batch_size 1 \
  --enable_wandb false \
  --model.torch_dtype_str float32
# ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
# ┃ Benchmark           ┃ Metric      ┃ Score  ┃ Std Error ┃
# ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
# │ enterprise_pubmedqa │ Accuracy    │ 45.00% │ -         │
# ├─────────────────────┼─────────────┼────────┼───────────┤
# │ enterprise_pubmedqa │ Num Correct │ 9      │ -         │
# ├─────────────────────┼─────────────┼────────┼───────────┤
# │ enterprise_pubmedqa │ Num Total   │ 20     │ -         │
# ├─────────────────────┼─────────────┼────────┼───────────┤
# │ enterprise_pubmedqa │ Micro F1    │ 45.00% │ -         │
# └─────────────────────┴─────────────┴────────┴───────────┘
```


##### tat-qa
```sh
oumi evaluate -c configs/enterprise/evaluation/task_tatqa.yaml \
  --model.model_name "HuggingFaceTB/SmolLM2-135M-Instruct" \
  --model.model_max_length 2048 \
  --tasks.0.num_samples 20 \
  --generation.batch_size 1 \
  --enable_wandb false \
  --model.torch_dtype_str float32
# ┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
# ┃ Benchmark        ┃ Metric          ┃ Score  ┃ Std Error ┃
# ┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
# │ enterprise_tatqa │ Exact Match     │ 0.00%  │ -         │
# ├──────────────────┼─────────────────┼────────┼───────────┤
# │ enterprise_tatqa │ F1              │ 10.25% │ -         │
# ├──────────────────┼─────────────────┼────────┼───────────┤
# │ enterprise_tatqa │ Boxed Rate      │ 15.00% │ -         │
# ├──────────────────┼─────────────────┼────────┼───────────┤
# │ enterprise_tatqa │ Num Exact Match │ 0      │ -         │
# ├──────────────────┼─────────────────┼────────┼───────────┤
# │ enterprise_tatqa │ Num Boxed       │ 3      │ -         │
# ├──────────────────┼─────────────────┼────────┼───────────┤
# │ enterprise_tatqa │ Num Total       │ 20     │ -         │
# └──────────────────┴─────────────────┴────────┴───────────┘
```


##### nl2sql
```sh
# NB mac unified memory system goes crazy on this task
# can disable gpu for mac testing with CUDA_VISIBLE_DEVICES="" and --model.device_map "cpu"
oumi evaluate -c configs/enterprise/evaluation/task_nl2sql.yaml \
  --model.model_name "HuggingFaceTB/SmolLM2-135M-Instruct" \
  --model.device_map "mps" \
  --model.model_max_length 4096 \
  --model.torch_dtype_str "float32" \
  --tasks.0.num_samples 3 \
  --generation.batch_size 1 \
  --generation.max_new_tokens 128 \
  --enable_wandb false
# ┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
# ┃ Benchmark         ┃ Metric          ┃ Score  ┃ Std Error ┃
# ┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
# │ enterprise_nl2sql │ Edit Similarity │ 26.61% │ -         │
# ├───────────────────┼─────────────────┼────────┼───────────┤
# │ enterprise_nl2sql │ Edit Distance   │ 73.39% │ -         │
# ├───────────────────┼─────────────────┼────────┼───────────┤
# │ enterprise_nl2sql │ Exact Match     │ 0.00%  │ -         │
# ├───────────────────┼─────────────────┼────────┼───────────┤
# │ enterprise_nl2sql │ Num Exact Match │ 0      │ -         │
# ├───────────────────┼─────────────────┼────────┼───────────┤
# │ enterprise_nl2sql │ Num Total       │ 3      │ -         │
# └───────────────────┴─────────────────┴────────┴───────────┘
```


##### Control evals (IFEval, SimpleSafetyTests)
```sh
# (NB uses lm_harness, takes much more time and memory than task evals)
oumi evaluate -c configs/enterprise/evaluation/control_evals.yaml \
  --model.model_name "HuggingFaceTB/SmolLM2-135M-Instruct" \
  --model.model_max_length 2048 \
  --tasks.0.num_samples 20 \
  --tasks.1.num_samples 10 \
  --generation.batch_size 1 \
  --enable_wandb false \
  --model.torch_dtype_str float32
# ┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
# ┃ Benchmark          ┃ Metric                  ┃ Score  ┃ Std Error ┃
# ┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
# │ leaderboard_ifeval │ Prompt Level Strict Acc │ 25.00% │ ±9.93%    │
# ├────────────────────┼─────────────────────────┼────────┼───────────┤
# │ leaderboard_ifeval │ Inst Level Strict Acc   │ 36.67% │ ±N/A      │
# ├────────────────────┼─────────────────────────┼────────┼───────────┤
# │ leaderboard_ifeval │ Prompt Level Loose Acc  │ 20.00% │ ±9.18%    │
# ├────────────────────┼─────────────────────────┼────────┼───────────┤
# │ leaderboard_ifeval │ Inst Level Loose Acc    │ 33.33% │ ±N/A      │
# └────────────────────┴─────────────────────────┴────────┴───────────┘
# ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
# ┃ Benchmark           ┃ Metric      ┃ Score  ┃ Std Error ┃
# ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
# │ simple_safety_tests │ Safe Rate   │ 20.00% │ -         │
# ├─────────────────────┼─────────────┼────────┼───────────┤
# │ simple_safety_tests │ Unsafe Rate │ 80.00% │ -         │
# ├─────────────────────┼─────────────┼────────┼───────────┤
# │ simple_safety_tests │ Num Safe    │ 2      │ -         │
# ├─────────────────────┼─────────────┼────────┼───────────┤
# │ simple_safety_tests │ Num Unsafe  │ 8      │ -         │
# ├─────────────────────┼─────────────┼────────┼───────────┤
# │ simple_safety_tests │ Num Total   │ 10     │ -         │
# └─────────────────────┴─────────────┴────────┴───────────┘
```


### Baseline eval on enterprise students
Evaluate small student models that we will offer hardened configs for at enterprise launch.

As a starting point we will focus on these, adding additional base models as time permits:

- Qwen3-4B-Instruct
- Llama-3.1-8B-Instruct
- Llama-3.2-1B-Instruct
- Gemma-3-4B-it

Now we move the workflows to Exun so we can execute real runs on GPU nodes...



### Hyperparam sweep experiments
Run SFT experiments and evaluate resulting ckpts to identify good training config defaults for each base model -- we want values that strike a good balance across the four target use cases.

### Collate results and finalized proposed training configs
Analyze results and propose final default training configs for each student.
