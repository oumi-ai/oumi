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

> For the latest notes on this workstream, see: https://docs.google.com/document/d/12prclQ1jiRUeQ_YNRhw554-IsqcjwS2xGmkMl-t8Q7M

> Tracking experiment results here: https://docs.google.com/spreadsheets/d/1LmFGfJRtNp3hpP9dwD7FJ1uN7Z0I8cKX-Z1wZ5ZaW2I


The workflows above were generated by Claude Code based on [its plan](https://docs.google.com/document/d/1OjG1xSnC4sBuQKG-jnlMRYwTYpk87AuKKe0obxcs49c/edit?tab=t.0), generated from Tim's [Training Planning for Enterprise Launch](https://docs.google.com/document/d/1Sq5ogeUlI8VomdJ5QO1g1ea3JwU-yaXBpHAYgA9qMiI/edit?tab=t.pqdeirkbycvi#heading=h.306lxpnc5lb0) doc. Oussama's original branch here: https://github.com/oumi-ai/oumi/compare/oelachqar/tunestuff (starting point for the current experiment workflows).

Here we harden and the workflows for building task-specific benchmarks, and then run many training experiments to identify good default training configs and hyperparam ranges for supported student models on the enterprise platform.

This section contains informal running notes as we progress through experiments.


### Finalize use case datasets and benchmarks

#### 1. Dataprep
Generate formatted train/test/val splits under `data/enterprise/` for each use case.

```sh
cd /Users/tim/workspace/oumi
conda activate oumi  # not needed on k8s pod

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

- Qwen3-4B-Instruct-2507
- Llama-3.1-8B-Instruct
- Llama-3.2-1B-Instruct
- Gemma-3-4B-it

Now we move the workflows to Exun so we can execute real runs on GPU nodes...

#### Pod setup
We can run jobs as k8s batch jobs or just get a GPU sleeper and trigger them manually. We'll take the latter route until we figure out a better automation. 

See instructions on getting a GPU sleeper pod in the k8s cluster here: https://docs.google.com/document/d/1z27fs3GCyqntW4VwEPVWeye33tr-lZsD6uiIZG_VJlI/edit?tab=t.r926fer6ffr#bookmark=id.qokkh6d7c2ek

Create the pod and exec into it, then set up datasets and start triggering jobs with `run_all_evals.sh`.

First set up the environment and install oumi from this branch:

```sh
# ad hoc dev setup script for this project
export BASEDIR="/data/tim/code/oumi"
cd $BASEDIR

./scripts/enterprise/gpu-pod-setup.sh \
  --hf-token <REDACTED> \
  --wandb-token <REDACTED>

source /data/tim/.bashrc
```

Now generate, format, split use case datasets:

```sh
python scripts/enterprise/prepare_datasets.py \
  --output-dir $DATASET_DIR \
  --all
```

#### Note on inference engine
We ran a parity test for all baseline evals for native inference engine versus vLLM, and found that metrics differ only marginally across the two engines. Since vLLM is at least an order of magnitude faster, vLLM will be the SOP for eval jobs. Note that vLLM handles batching automatically, so there is no need to specify batch_size in eval configs.

> NB: We will sometimes hit OOM on Gemma or Llama3-8b when running evals in quick succession, causing the job and the pod to die. This can be mitigated to a degree (not guaranteed) by forcing cleanup of GPU memory with `pkill -9 -f vllm` after one job completes and then waiting a few minutes before restarting. A better long-term solution would probably be moving to REMOTE_VLLM inference engine (one persistent server per model which we can run multiple evals against).

Note also that even with `temperature: 0.0` or `temperature: 0.001` in all eval configs, we are getting more nondeterminism than expected in both native inference engine and vLLM (detected via observing variation across repeated runs). It does appear that the sampling params are being picked up, based on the `generation_params.json` included in the eval results bundles. Some nondeterminism is unavoidable even without sampling.


#### Execute baseline evals
Trigger all baseline evals like below. These will run serially with config values set in the corresponding task config under `evaluation/task_*.yaml` unless overridden in the CLI commands in `run_all_evals.sh`. Results will get dumped to timestamped directories per base model under the directory passed to `run_all_evals.sh` as `--output-dir`.

```sh
tmn baselines-vllm
cd /data/tim/code/oumi

DATASET_DIR=/data/tim/code/oumi/data/enterprise
OUTPUT_BASEDIR=/data/tim/evals/ent/baselines

# NB if the pod dies with OOM (137), get a new one and sleep between each set of evals
./scripts/enterprise/run_all_evals.sh \
  --model-name "Qwen/Qwen3-4B-Instruct-2507" \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR &&
./scripts/enterprise/run_all_evals.sh \
  --model-name "meta-llama/Llama-3.2-1B-Instruct" \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR &&
./scripts/enterprise/run_all_evals.sh \
  --model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR &&
./scripts/enterprise/run_all_evals.sh \
  --model-name "google/gemma-3-4b-it" \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR
```

#### Collate eval results
Run `collate_eval_results.py` to produce a flat csv table that gathers all available results, then `plot_eval_results.py` to create a basic visualization of the results. We'll transfer results bundles back to local first so we can inspect individual predictions with convenient desktop tools:

```sh
# transfer results bundles for collation and plotting
BASELINE_EVALS_SRC=$NAMESPACE/<POD>:/data/tim/code/oumi/output/enterprise/evals-vllm3
BASELINE_EVALS_DEST=~/Downloads/evals-vllm3
kubectl cp $BASELINE_EVALS_SRC $BASELINE_EVALS_DEST

# collate results
cd /Users/tim/workspace/oumi
python scripts/enterprise/collate_eval_results.py \
  --run-dirs $BASELINE_EVALS_DEST/* \
  --output $BASELINE_EVALS_DEST/collated/results.csv \
  --json $BASELINE_EVALS_DEST/collated/results.json

# produce a plot for each benchmark and available model
python scripts/enterprise/plot_eval_results.py \
  --results-json $BASELINE_EVALS_DEST/collated/results.json \
  --output $BASELINE_EVALS_DEST/collated/results-plot.png
```

Key metrics:

| Model | Banking77 | PubMedQA | PubMedQA | TAT-QA | TAT-QA | NL2SQL | IFEval | Safety |
|-------|-----------|----------|----------|--------|--------|--------|--------|--------|
|       | Accuracy | Accuracy | Macro F1 | EM | Boxed% | EditSim | Prompt Strict | Safe% |
| **Qwen3-4B-Instruct-2507** | 53.0% | 52.0% | 45.2% | 25.4% | 86.6% | 54.4% | 83.0% | 89.0% |
| **Llama-3.2-1B-Instruct** | 2.3% | 49.0% | 40.2% | 13.9% | 72.8% | 36.3% | 50.5% | 76.0% |
| **Llama-3.1-8B-Instruct** | 38.1% | 70.0% | 50.5% | 27.0% | 88.5% | 51.7% | 73.9% | 94.0% |
| **Gemma-3-4B-it** | 60.8% | 49.0% | 39.3% | 40.0% | 95.4% | 50.0% | 72.8% | 94.0% |


<br>

---
---
---
<br><br><br>

### SFT workflows

#### Example SFT workflow
Example e2e SFT training --> eval --> collate workflow, with base model Qwen3-4B-Instruct-2507 and pubmedqa task.

```sh
cd $BASEDIR
source /data/tim/.bashrc
wandb login

TRAINING_CONFIG=$BASEDIR/configs/enterprise/training/qwen3_4b_instruct_2507_train_full.yaml
TRAIN_DATASET=$BASEDIR/data/enterprise/pubmedqa/train.jsonl
VAL_DATASET=$BASEDIR/data/enterprise/pubmedqa/val.jsonl

RUN_NAME=qwen3-4b-inst-2507-pubmedqa-test
OUTPUT_DIR=/data/tim/checkpoints/$RUN_NAME

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
    -c $TRAINING_CONFIG \
    --data.train.datasets.0.dataset_path=$TRAIN_DATASET \
    --data.validation.datasets.0.dataset_path=$VAL_DATASET \
    --training.run_name=$RUN_NAME \
    --training.output_dir=$OUTPUT_DIR
```

See the job in wandb at: https://wandb.ai/lefft-oumi/huggingface/runs/16wytcrf.

Once complete, evaluate the final checkpoint against all tasks (supply the output path as model_name):

```sh
EVAL_OUTPUT_BASEDIR=/data/tim/evals/ent/ft

./scripts/enterprise/run_all_evals.sh \
  --model-name $OUTPUT_DIR \
  --data-dir $DATASET_DIR \
  --output-dir $EVAL_OUTPUT_BASEDIR
```

Then copy the results bundles to local for collation and analysis:

```sh
FT_TEST_EVALS_SRC=$NAMESPACE/<POD>:/data/tim/evals/ent/ft/20260109_224544_qwen3-4b-inst-2507-pubmedqa-test
FT_TEST_EVALS_DEST=~/Downloads/20260109_224544_qwen3-4b-inst-2507-pubmedqa-test

kubectl cp $FT_TEST_EVALS_SRC $FT_TEST_EVALS_DEST

# manually create a new dir with just baselines and this run
BASELINES_PLUS_TEST=/Users/tim/Downloads/ent-eval-baselines-plus-qwen-test

python scripts/enterprise/collate_eval_results.py \
  --run-dirs $BASELINES_PLUS_TEST/* \
  --output $BASELINES_PLUS_TEST/collated/results.csv \
  --json $BASELINES_PLUS_TEST/collated/results.json

# produce a plot for each benchmark and available model
python scripts/enterprise/plot_eval_results.py \
  --results-json $BASELINES_PLUS_TEST/collated/results.json \
  --output $BASELINES_PLUS_TEST/collated/results-plot.png
```

Results are as expected: boost in task performance, with varying degree of regression on other benchmarks:

| Model | Banking77 | PubMedQA | PubMedQA | TAT-QA | TAT-QA | NL2SQL | IFEval | Safety |
|-------|-----------|----------|----------|--------|--------|--------|--------|--------|
|       | Accuracy | Accuracy | Macro F1 | EM | Boxed% | EditSim | Prompt Strict | Safe% |
| **Qwen3-4B-Instruct-2507** | 53.0% | 52.0% | 45.2% | 25.4% | 86.6% | 54.4% | 83.0% | 89.0% |
| **Llama-3.2-1B-Instruct** | 2.3% | 49.0% | 40.2% | 13.9% | 72.8% | 36.3% | 50.5% | 76.0% |
| **Llama-3.1-8B-Instruct** | 38.1% | 70.0% | 50.5% | 27.0% | 88.5% | 51.7% | 73.9% | 94.0% |
| **Gemma-3-4B-it** | 60.8% | 49.0% | 39.3% | 40.0% | 95.4% | 50.0% | 72.8% | 94.0% |
| **Qwen3-4B + PubMedQA FT** | 22.5% | **74.0%** | **63.5%** | 0.0% | 0.0% | 54.4% | 67.3% | 86.0% |


#### Example tuning workflow
Demonstrate the `oumi tune` workflow, again using the pubmedqa use case. Here we use SmolLM2-135M-Instruct as base so we can execute many jobs quickly.

Note that we made some changes to oumi core to enable custom eval functions -- this way we get a full pubmedqa eval for every completed training run.

```sh
# needed for Optuna
pip install -e ".[tune]"

TUNE_TEST_OUTDIR=/data/tim/tune/ent/test_smollm2_pubmedqa-custom

oumi tune -c configs/enterprise/tuning/test_smollm2_pubmedqa.yaml \
  --tuning.output_dir $TUNE_TEST_OUTDIR
```

See the job in wandb at: https://wandb.ai/lefft-oumi/huggingface/runs/pwmfom3a.

You can retrieve eval results from the output dir (only includes task-specific evals):

```sh
cat $TUNE_TEST_OUTDIR/trial_0/custom_eval/custom_20260109_230642/task_result.json \
  | jq '.results.enterprise_pubmedqa.accuracy'
# 0.5
```

See also `$TUNE_TEST_OUTDIR/trials_results.csv` for a comparison of training metrics across trials.

You can also run more in-depth evals on individual trials if desired, using `run_all_evals.sh` as above.





### Initial config tuning experiments
Run SFT experiments and evaluate resulting ckpts to identify good training config defaults for each base model -- we want values that strike a good balance across the target use cases. At this stage we want to collect signals from multiple evals, so we launch hand-selected configs manually and evaluating the resulting models against our suite of benchmarks (four tasks and two control benchmarks).

Eventually we will want to do larger-scale systematic grid searches over several key config values for each base model and use case, to empirically identify optimal training configs.

But for now we manually select several configurations across multiple tasks, and settle on defaults that work reasonably well. Optimization will come after we have established reasonable defaults for a range of base models. 

#### `Gemma-3-4B-it`
We ran 12 total SFT jobs on top of `gemma-3-4B-it` targeting tat-qa (13k train records) and pubmedqa (800 train records), varying (not systematically) max LR, number of epochs, and weight decay. Only results from selected configurations are shown here.

The config at `configs/enterprise/training/gemma3_4b_it_train_full.yaml` works well for both tat-qa and pubmedqa -- though there is room to optimize it for both use cases. It was established by starting from `configs/recipes/gemma3/sft/4b_full/train.yaml`, and first adjusting values as needed to get jobs to run successfully, and then iteratively refining hyperparameters by monitoring loss curves in wandb and downstream eval metrics for each run.

Here are two workflows that use the selected training config (all paths on PVC `pvc-local-vp6ww`):

```sh
### tat-qa training (selected default config)
BASEDIR=/data/tim/code/oumi
TRAINING_CONFIG=$BASEDIR/configs/enterprise/training/gemma3_4b_it_train_full.yaml
TRAIN_DATASET=$BASEDIR/data/enterprise/tatqa/train.jsonl
VAL_DATASET=$BASEDIR/data/enterprise/tatqa/val.jsonl

RUN_NAME=gemma3-4b-it-tatqa-6-3
OUTPUT_DIR=/data/tim/checkpoints/$RUN_NAME

# NB: tatqa-6-2 uses 2e-5 1 ep, tatqa-6-3 uses 1.5e-5, 2 eps

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
    -c $TRAINING_CONFIG \
    --data.train.datasets.0.dataset_path=$TRAIN_DATASET \
    --data.validation.datasets.0.dataset_path=$VAL_DATASET \
    --training.run_name=$RUN_NAME \
    --training.output_dir=$OUTPUT_DIR

### tat-qa evaluation
DATASET_DIR=/data/tim/code/oumi/data/enterprise
OUTPUT_BASEDIR=/data/tim/evals/ent/ft
GEMMA_TATQA6=/data/tim/checkpoints/gemma3-4b-it-tatqa-6-3

# NB: must copy preprocessor_config.json from HF repo (not saved as part of trained model bundle)
# without this, vLLM evals will fail with "Can't load image processor for '/data/tim/checkpoints/gemma3-4b-it-tatqa-6'..."
huggingface-cli download google/gemma-3-4b-it preprocessor_config.json \
    --local-dir $GEMMA_TATQA6/

./scripts/enterprise/run_all_evals.sh \
  --model-name $GEMMA_TATQA6 \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR
```

```sh
### pubmedqa training 
BASEDIR=/data/tim/code/oumi
TRAINING_CONFIG=$BASEDIR/configs/enterprise/training/gemma3_4b_it_train_full.yaml
TRAIN_DATASET=$BASEDIR/data/enterprise/pubmedqa/train.jsonl
VAL_DATASET=$BASEDIR/data/enterprise/pubmedqa/val.jsonl

RUN_NAME=gemma3-4b-it-pubmedqa-2-4
OUTPUT_DIR=/data/tim/checkpoints/$RUN_NAME

# NB: this is used in pubmedqa-2-3:
# --training.num_train_epochs 3 \
# but in pubmedqa-2-4, we set LR 1.5e-5 and epochs 2

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
    -c $TRAINING_CONFIG \
    --data.train.datasets.0.dataset_path=$TRAIN_DATASET \
    --data.validation.datasets.0.dataset_path=$VAL_DATASET \
    --training.run_name=$RUN_NAME \
    --training.output_dir=$OUTPUT_DIR

### pubmedqa evaluation
DATASET_DIR=/data/tim/code/oumi/data/enterprise
OUTPUT_BASEDIR=/data/tim/evals/ent/ft
GEMMA_PMQA2=/data/tim/checkpoints/gemma3-4b-it-pubmedqa-2-4
huggingface-cli download google/gemma-3-4b-it preprocessor_config.json \
    --local-dir $GEMMA_PMQA2/

./scripts/enterprise/run_all_evals.sh \
  --model-name $GEMMA_PMQA2 \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR
```

Copy the results out of the cluster for collation, plotting, and analysis:

```sh
# [local]
FT_TEST_EVALS_SRC=$NAMESPACE/<POD>:/data/tim/evals/ent/ft
FT_TEST_EVALS_DEST=~/Downloads/20260109-ft

# NB also copy over baselines, and remove bundles that are not of interest before plotting
kubectl cp $FT_TEST_EVALS_SRC $FT_TEST_EVALS_DEST

python scripts/enterprise/collate_eval_results.py \
  --run-dirs $FT_TEST_EVALS_DEST/* \
  --output $FT_TEST_EVALS_DEST/collated/results.csv \
  --json $FT_TEST_EVALS_DEST/collated/results.json

# produce a plot for each benchmark and available model
python scripts/enterprise/plot_eval_results.py \
  --results-json $FT_TEST_EVALS_DEST/collated/results.json \
  --output $FT_TEST_EVALS_DEST/collated/results-plot.png
```

Final results:

| Model | Banking77 | PubMedQA | PubMedQA | TAT-QA | TAT-QA | NL2SQL | IFEval | Safety |
|-------|-----------|----------|----------|--------|--------|--------|--------|--------|
|       | Accuracy | Accuracy | Macro F1 | EM | Boxed% | EditSim | Prompt Strict | Safe% |
| **Gemma-3-4B-it** (baseline) | 61.0% | 49.0% | 39.2% | 39.7% | 95.3% | 50.2% | 73.4% | 95.0% |
| **Gemma3-4B + TAT-QA FT** | 52.4% | 49.0% | 27.0% | **46.6%** | **100%** | 46.3% | 68.0% | 93.0% |
| **Gemma3-4B + PubMedQA FT** | 57.9% | **67.0%** | **46.8%** | 0.7% | 5.6% | 45.2% | 71.2% | 96.0% |

High-level observations:

- tat-qa:
  - +6.9 EM improvement (39.7% --> 46.6%), similar boost in F1
  - boxed rate at 100% from 95% (showing the model follows response format instructions)
  - moderate regression on other benchnmarks

- pubmedqa:
  - +18 accuracy improvement (49% --> 67%), +8 on macro F1 (.39 --> .47)
  - moderate regression on other benchmarks
  - ignores instruction to use `\boxed{}` in tat-qa, leading to performance collapse

Follow-up -- ent uses completions-only training by default (not the case for oumi). We found that adding completion-only settings for `gemma3-4b-it` does not work as well for both tasks out of the box. The default config has been updated to use LR 1.5e-5 and train for 2 epochs.

Updated results table:

| Model | LR | Epochs | Compl. Only | Banking77 | PubMedQA | PubMedQA | TAT-QA | TAT-QA | NL2SQL | IFEval | Safety |
|-------|-----|--------|-------------|-----------|----------|----------|--------|--------|--------|--------|--------|
|       |     |        |             | Accuracy | Accuracy | Macro F1 | EM | Boxed% | EditSim | Prompt Strict | Safe% |
| **Gemma-3-4B-it** (baseline) | — | — | — | 61.0% | 49.0% | 39.2% | 39.7% | 95.3% | 50.2% | 73.4% | 95.0% |
| **Gemma3-4B + TAT-QA FT** | 2e-5 | 1 | No | 52.4% | 49.0% | 27.0% | 46.6% | 100% | 46.3% | 68.0% | 93.0% |
| **Gemma3-4B + PubMedQA FT** | 2e-5 | 1 | No | 57.9% | 67.0% | 46.8% | 0.7% | 5.6% | 45.2% | 71.2% | 96.0% |
| **Gemma3-4B + TAT-QA FT v2** | 1.5e-5 | 2 | Yes | 62.0% | 51.0% | 39.9% | **54.9%** | **100%** | 49.0% | **76.0%** | **97.0%** |
| **Gemma3-4B + PubMedQA FT v2** | 1.5e-5 | 2 | Yes | 54.5% | **82.0%** | **58.4%** | 0.0% | 0.4% | 50.0% | 72.3% | 96.0% |

Results are much stronger with completions-only.


#### `Ministral-3-3B-Instruct-2512`

NB: These models were only released 12/2025, and our current version of HF transformers does not support minimstral3. Upgrading to the most recent transformers breaks imports in both the oumi codebase (`ImportError: cannot import name 'SpecialTokensMixin' from 'transformers'`) and dependencies (`ImportError: cannot import name 'HybridCache' from 'transformers'`). We will revisit this once our transformers version has been upgraded.


#### `Llama-3.2-1B-Instruct`
Prepare a default config and verify that it produces gains on target tasks without catastrophic regression on general capabilities or other tasks. If unsuccessful, run additional SFT jobs as needed to find settings that work reasonably well across multiple tasks (here pubmedqa: small 800-record train set, simple classification task; and tat-qa: open-ended data analysis task, 13k train set). 

```sh
BASEDIR=/data/tim/code/oumi

# pubmedqa
TRAINING_CONFIG=$BASEDIR/configs/enterprise/training/llama32_1b_instruct_train_full.yaml
TRAIN_DATASET=$BASEDIR/data/enterprise/pubmedqa/train.jsonl
VAL_DATASET=$BASEDIR/data/enterprise/pubmedqa/val.jsonl

RUN_NAME=llama32_1b_instruct_train_full-pubmedqa
OUTPUT_DIR=/data/tim/checkpoints/$RUN_NAME

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
    -c $TRAINING_CONFIG \
    --data.train.datasets.0.dataset_path=$TRAIN_DATASET \
    --data.validation.datasets.0.dataset_path=$VAL_DATASET \
    --training.run_name=$RUN_NAME \
    --training.output_dir=$OUTPUT_DIR

# tatqa
TRAINING_CONFIG=$BASEDIR/configs/enterprise/training/llama32_1b_instruct_train_full.yaml
TRAIN_DATASET=$BASEDIR/data/enterprise/tatqa/train.jsonl
VAL_DATASET=$BASEDIR/data/enterprise/tatqa/val.jsonl

RUN_NAME=llama32_1b_instruct_train_full-tatqa
OUTPUT_DIR=/data/tim/checkpoints/$RUN_NAME

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
    -c $TRAINING_CONFIG \
    --data.train.datasets.0.dataset_path=$TRAIN_DATASET \
    --data.validation.datasets.0.dataset_path=$VAL_DATASET \
    --training.run_name=$RUN_NAME \
    --training.output_dir=$OUTPUT_DIR
```

Run full evals against the final ckpts:

```sh
cd /data/tim/code/oumi
DATASET_DIR=/data/tim/code/oumi/data/enterprise
OUTPUT_BASEDIR=/data/tim/evals/ent/llama32-ft

# pubmedqa
./scripts/enterprise/run_all_evals.sh \
  --model-name /data/tim/checkpoints/llama32_1b_instruct_train_full-pubmedqa \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR

# tatqa
./scripts/enterprise/run_all_evals.sh \
  --model-name /data/tim/checkpoints/llama32_1b_instruct_train_full-tatqa \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR

# baseline
./scripts/enterprise/run_all_evals.sh \
  --model-name "meta-llama/Llama-3.2-1B-Instruct" \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR
```

Collate results and inspect:

```sh
FT_TEST_EVALS_SRC=$NAMESPACE/<POD>:/data/tim/evals/ent/llama32-ft
FT_TEST_EVALS_DEST=~/Downloads/llama32-ft

kubectl cp $FT_TEST_EVALS_SRC $FT_TEST_EVALS_DEST

python scripts/enterprise/collate_eval_results.py \
  --run-dirs $FT_TEST_EVALS_DEST/* \
  --output $FT_TEST_EVALS_DEST/collated/results.csv \
  --json $FT_TEST_EVALS_DEST/collated/results.json

python scripts/enterprise/plot_eval_results.py \
  --results-json $FT_TEST_EVALS_DEST/collated/results.json \
  --output $FT_TEST_EVALS_DEST/collated/results-plot.png
```

Results:

| Model | PubMedQA Acc | TAT-QA F1 | Banking77 Acc | NL2SQL EM | IFEval Strict | Safety Rate |
|-------|--------------|-----------|---------------|-----------|---------------|-------------|
| Baseline (1B) | 0.47 | 0.205 | 0.024 | 0.00 | 0.499 | 0.73 |
|  PubMedQA | **0.69** | 0.00 | 0.032 | 0.00 | 0.368 | 0.82 |
| TAT-QA | 0.45 | **0.526** | 0.013 | 0.00 | 0.516 | 0.83 |

Notes:
- PubMedQA FT: .47 --> .69 task accuracy, tat-qa tanks
- TAT-QA FT: .205 --> 0.526 task F1, minor regression on PubMedQA
- mixed impact on control benchmarks


#### `Llama-3.2-3B-Instruct`
Reduce LR from 2e-5 to 1e-5 for 3b. Also try increasing GAS (and hence EBS) for stability. Other than that, we'll validate that the tested 1b config settings also work for 3b, and consider that the default if so. If not, run tweaks to identify a balance.

```sh
BASEDIR=/data/tim/code/oumi

# pubmedqa
TRAINING_CONFIG=$BASEDIR/configs/enterprise/training/llama32_3b_instruct_train_full.yaml
TRAIN_DATASET=$BASEDIR/data/enterprise/pubmedqa/train.jsonl
VAL_DATASET=$BASEDIR/data/enterprise/pubmedqa/val.jsonl

RUN_NAME=llama32_3b_instruct-pubmedqa
OUTPUT_DIR=/data/tim/checkpoints/$RUN_NAME

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
    -c $TRAINING_CONFIG \
    --data.train.datasets.0.dataset_path=$TRAIN_DATASET \
    --data.validation.datasets.0.dataset_path=$VAL_DATASET \
    --training.run_name=$RUN_NAME \
    --training.output_dir=$OUTPUT_DIR
```

```sh
# tatqa
TRAINING_CONFIG=$BASEDIR/configs/enterprise/training/llama32_3b_instruct_train_full.yaml
TRAIN_DATASET=$BASEDIR/data/enterprise/tatqa/train.jsonl
VAL_DATASET=$BASEDIR/data/enterprise/tatqa/val.jsonl

RUN_NAME=llama32_3b_instruct-tatqa
OUTPUT_DIR=/data/tim/checkpoints/$RUN_NAME

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
    -c $TRAINING_CONFIG \
    --data.train.datasets.0.dataset_path=$TRAIN_DATASET \
    --data.validation.datasets.0.dataset_path=$VAL_DATASET \
    --training.run_name=$RUN_NAME \
    --training.output_dir=$OUTPUT_DIR
```

Also ran tat-qa variants with GAS 4 and 16. Eval:

```sh
cd /data/tim/code/oumi
DATASET_DIR=/data/tim/code/oumi/data/enterprise
OUTPUT_BASEDIR=/data/tim/evals/ent/llama32-3b-ft

# pubmedqa
./scripts/enterprise/run_all_evals.sh \
  --model-name /data/tim/checkpoints/llama32_3b_instruct-pubmedqa \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR

# tatqa gas 2
./scripts/enterprise/run_all_evals.sh \
  --model-name /data/tim/checkpoints/llama32_3b_instruct-tatqa \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR

# tatqa gas 4
./scripts/enterprise/run_all_evals.sh \
  --model-name /data/tim/checkpoints/llama32_3b_instruct-tatqa-gas4 \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR

# tatqa gas 16
./scripts/enterprise/run_all_evals.sh \
  --model-name /data/tim/checkpoints/llama32_3b_instruct-tatqa-gas16 \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR

# baseline
./scripts/enterprise/run_all_evals.sh \
  --model-name "meta-llama/Llama-3.2-3B-Instruct" \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR
```

Collate and plot results:

```sh
EVALS_SRC=$NAMESPACE/<POD>:/data/tim/evals/ent/llama32-3b-ft
EVALS_DEST=~/Downloads/llama32-3b-ft
kubectl cp $EVALS_SRC $EVALS_DEST

python scripts/enterprise/collate_eval_results.py \
  --run-dirs $EVALS_DEST/* \
  --output $EVALS_DEST/collated/results.csv \
  --json $EVALS_DEST/collated/results.json

python scripts/enterprise/plot_eval_results.py \
  --results-json $EVALS_DEST/collated/results.json \
  --output $EVALS_DEST/collated/results-plot.png
```

| Model | PubMedQA Acc | TAT-QA F1 | Banking77 Acc | NL2SQL EM | IFEval Strict | Safety Rate |
|-------|--------------|-----------|---------------|-----------|---------------|-------------|
| Baseline (3B) | 0.69 | 0.379 | 0.275 | 0.00 | 0.689 | 0.85 |
| PubMedQA | **0.77** | 0.389 | 0.286 | 0.00 | 0.665 | 0.84 |
| TAT-QA (gas02) | 0.61 | **0.625** | 0.275 | 0.01 | 0.702 | 0.84 |
| TAT-QA (gas04) | 0.61 | 0.608 | 0.278 | 0.01 | 0.706 | 0.84 |
| TAT-QA (gas16) | 0.63 | 0.589 | 0.271 | 0.00 | 0.686 | 0.78 |

Notes:
- PubMedQA FT: .69 --> 0.77 task accuracy, no or marginal regression elsewhere
- TAT-QA FT (GAS 2): .379 --> 0.625 task F1, minor no or marginal regression elsewhere
- 3b baseline much stronger than 1b, less catastrophic forgetting during FT
- Best tat-qa config uses GAS 2 (smaller EBS than alternatives)


#### `Llama-3.1-8B-Instruct`
Start with a lower LR of 5e-6 for 8b, but compare with 1e-5. 

Note that enabling liger kernel will cause the job to fail with `AttributeError: 'CausalLMOutputWithPast' object has no attribute 'token_accuracy' `. 

```sh
BASEDIR=/data/tim/code/oumi

# pubmedqa
TRAINING_CONFIG=$BASEDIR/configs/enterprise/training/llama31_8b_instruct_train_full.yaml
TRAIN_DATASET=$BASEDIR/data/enterprise/pubmedqa/train.jsonl
VAL_DATASET=$BASEDIR/data/enterprise/pubmedqa/val.jsonl

RUN_NAME=llama31_8b_instruct-pubmedqa
OUTPUT_DIR=/data/tim/checkpoints/$RUN_NAME

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
    -c $TRAINING_CONFIG \
    --data.train.datasets.0.dataset_path=$TRAIN_DATASET \
    --data.validation.datasets.0.dataset_path=$VAL_DATASET \
    --training.run_name=$RUN_NAME \
    --training.output_dir=$OUTPUT_DIR
```

```sh
# tatqa
TRAINING_CONFIG=$BASEDIR/configs/enterprise/training/llama31_8b_instruct_train_full.yaml
TRAIN_DATASET=$BASEDIR/data/enterprise/tatqa/train.jsonl
VAL_DATASET=$BASEDIR/data/enterprise/tatqa/val.jsonl

RUN_NAME=llama31_8b_instruct-tatqa-1e5
OUTPUT_DIR=/data/tim/checkpoints/$RUN_NAME

oumi distributed torchrun --nproc_per_node=8 --master-port=9011 -m oumi train \
    -c $TRAINING_CONFIG \
    --data.train.datasets.0.dataset_path=$TRAIN_DATASET \
    --data.validation.datasets.0.dataset_path=$VAL_DATASET \
    --training.run_name=$RUN_NAME \
    --training.output_dir=$OUTPUT_DIR
```

Run full evals against the final ckpts:

```sh
cd /data/tim/code/oumi
DATASET_DIR=/data/tim/code/oumi/data/enterprise
OUTPUT_BASEDIR=/data/tim/evals/ent/llama31-8b-ft

./scripts/enterprise/run_all_evals.sh \
  --model-name /data/tim/checkpoints/llama31_8b_instruct-pubmedqa \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR

./scripts/enterprise/run_all_evals.sh \
  --model-name /data/tim/checkpoints/llama31_8b_instruct-pubmedqa-1e5 \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR

./scripts/enterprise/run_all_evals.sh \
  --model-name /data/tim/checkpoints/llama31_8b_instruct-tatqa \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR

./scripts/enterprise/run_all_evals.sh \
  --model-name /data/tim/checkpoints/llama31_8b_instruct-tatqa-1e5 \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR

./scripts/enterprise/run_all_evals.sh \
  --model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --data-dir $DATASET_DIR \
  --output-dir $OUTPUT_BASEDIR
```

Collate and plot results:

```sh
EVALS_SRC=$NAMESPACE/<POD>:/data/tim/evals/ent/llama31-8b-ft
EVALS_DEST=~/Downloads/llama31-8b-ft
kubectl cp $EVALS_SRC $EVALS_DEST

python scripts/enterprise/collate_eval_results.py \
  --run-dirs $EVALS_DEST/* \
  --output $EVALS_DEST/collated/results.csv \
  --json $EVALS_DEST/collated/results.json

python scripts/enterprise/plot_eval_results.py \
  --results-json $EVALS_DEST/collated/results.json \
  --output $EVALS_DEST/collated/results-plot.png
```

Results:

| Model | PubMedQA Acc | TAT-QA F1 | Banking77 Acc | NL2SQL EM | IFEval Strict | Safety Rate |
|-------|--------------|-----------|---------------|-----------|---------------|-------------|
| Baseline (8B) | 0.78 | 0.341 | 0.386 | 0.00 | 0.752 | 0.93 |
| PubMedQA (lr=5e-6) | **0.82** | 0.415 | 0.420 | 0.00 | 0.747 | 0.88 |
| PubMedQA (lr=1e-5) | **0.82** | 0.392 | 0.420 | 0.00 | 0.651 | 0.74 |
| TAT-QA (lr=5e-6) | 0.65 | 0.683 | 0.420 | 0.00 | 0.750 | 0.71 |
| TAT-QA (lr=1e-5) | 0.59 | **0.705** | 0.376 | 0.00 | 0.747 | 0.79 |

Notes:
- PubMedQA: both LRs get 0.82 acc, but 5e-6 has better macro F1 (.66 versus .58) and control metrics
- TAT-QA: 1e-5 slightly better than 5e-6 on target task (F1 .705 versus .683, EM .64 versus .61) but regresses more on most other tasks
- Going with max LR 5e-6 as default to strike a balance


#### `SmolLM2-1.7B-Instruct`
Small but capable model from HuggingFace. Uses ChatML format (`<|im_start|>` tokens). Starting with the preset config which has LR 2e-5, GAS 16, 3 epochs. Note: with GAS=16 and batch=1 on 8 GPUs, EBS=128, which means only ~6 steps/epoch on PubMedQA. Consider reducing GAS to 4 for more granular updates.

Use the new runner `run_experiment.sh` to orchestrate all the steps:

- run SFT
- evaluate FT'ed ckpt
- run baseline evals if `--with-baseline` is set
- collate + plot results (includes FT'ed model evals, plus most recent matching baseline evals)

This greatly reduces the manual work needed to run many jobs. Just launch all at once in a tmux session and come back later. So far it has been very reliable.

```sh
# no movement on initial smollm configs for pubmedqa
./scripts/enterprise/run_experiment.sh --model smollm2_1.7b --task pubmedqa --with-baseline

# try a few tweaks
./scripts/enterprise/run_experiment.sh --model smollm2_1.7b --task pubmedqa --suffix "-ebs32"
./scripts/enterprise/run_experiment.sh --model smollm2_1.7b --task pubmedqa --suffix "-3ep-8e6"

./scripts/enterprise/run_experiment.sh --model smollm2_1.7b --task tatqa
./scripts/enterprise/run_experiment.sh --model smollm2_1.7b --task tatqa --suffix "-3ep-8e6"

./scripts/enterprise/run_experiment.sh --model smollm2_1.7b --task banking77
./scripts/enterprise/run_experiment.sh --model smollm2_1.7b --task banking77 --suffix "-3ep-8e6"

# no movement on nl2sql, unsurprising for such a small base
./scripts/enterprise/run_experiment.sh --model smollm2_1.7b --task nl2sql
./scripts/enterprise/run_experiment.sh --model smollm2_1.7b --task nl2sql --suffix "-3ep-8e6"

# still nothing solid for pubmedqa
# try bumping LR much higher, only 2 eps, and gas 2 (for ebs 32)
./scripts/enterprise/run_experiment.sh --model smollm2_1.7b --task pubmedqa --suffix "-2ep-2gas-7e5"
./scripts/enterprise/run_experiment.sh --model smollm2_1.7b --task tatqa --suffix "-2ep-2gas-7e5"
./scripts/enterprise/run_experiment.sh --model smollm2_1.7b --task banking77 --suffix "-2ep-2gas-7e5"
./scripts/enterprise/run_experiment.sh --model smollm2_1.7b --task nl2sql --suffix "-2ep-2gas-7e5"

# okay cool this works best for all use cases

# copy results to local for inspection
kubectl cp $NAMESPACE/$POD:/data/tim/evals/ent/smollm2_1.7b-ft ~/Downloads/smollm2_1.7b-ft
kubectl cp $NAMESPACE/$POD:/data/tim/evals/ent/baselines/20260121_021024_SmolLM2-1_7B-Instruct ~/Downloads/smollm2_1.7b-ft/SmolLM2-1_7B-Instruct
```

Results:

| Model | PubMedQA Acc | TAT-QA F1 | Banking77 Acc | NL2SQL EM | IFEval Strict | Safety Rate |
|-------|--------------|-----------|---------------|-----------|---------------|-------------|
| Baseline (1.7B) | 0.71 | 0.101 | 0.048 | 0.00 | 0.471 | 0.73 |
| PubMedQA (default) | 0.66 | 0.099 | 0.049 | 0.00 | 0.495 | 0.72 |
| PubMedQA (ebs32) | 0.68 | 0.090 | 0.045 | 0.00 | 0.499 | 0.72 |
| PubMedQA (3ep-8e6) | 0.63 | 0.102 | 0.046 | 0.00 | 0.486 | 0.72 |
| PubMedQA (2ep-2gas-7e5) | **0.76** | 0.002 | 0.046 | 0.00 | 0.569 | 0.53 |
| TAT-QA (default) | 0.57 | 0.421 | 0.056 | 0.00 | 0.521 | 0.60 |
| TAT-QA (3ep-8e6) | 0.62 | 0.350 | 0.050 | 0.00 | 0.494 | 0.70 |
| TAT-QA (2ep-2gas-7e5) | 0.56 | **0.498** | 0.055 | 0.00 | 0.540 | 0.57 |
| Banking77 (default) | 0.71 | 0.071 | 0.667 | 0.00 | 0.499 | 0.75 |
| Banking77 (3ep-8e6) | 0.71 | 0.088 | 0.453 | 0.00 | 0.486 | 0.69 |
| Banking77 (2ep-2gas-7e5) | 0.65 | 0.006 | **0.909** | 0.00 | 0.510 | 0.52 |
| NL2SQL (default) | 0.70 | 0.067 | 0.044 | 0.00 | 0.471 | 0.80 |
| NL2SQL (3ep-8e6) | 0.72 | 0.095 | 0.049 | 0.00 | 0.473 | 0.70 |
| NL2SQL (2ep-2gas-7e5) | 0.70 | 0.002 | 0.046 | 0.00 | 0.492 | 0.64 |


#### `Qwen3-4B-Instruct-2507`
This config is already tuned and widely used. Verify that it produces reasonable results:

```sh
./scripts/enterprise/run_experiment.sh --model qwen3_4b --task pubmedqa --with-baseline
./scripts/enterprise/run_experiment.sh --model qwen3_4b --task tatqa
./scripts/enterprise/run_experiment.sh --model qwen3_4b --task banking77
./scripts/enterprise/run_experiment.sh --model qwen3_4b --task nl2sql

# transfer results to inspect locally
kubectl cp $NAMESPACE/$POD:/data/tim/evals/ent/qwen3_4b-ft ~/Downloads/qwen3_4b-ft
kubectl cp $NAMESPACE/$POD:/data/tim/evals/ent/baselines/20260121_182543_Qwen3-4B-Instruct-2507 ~/Downloads/qwen3_4b-ft/20260121_182543_Qwen3-4B-Instruct-2507
```

Results look good with existing defaults:

| Model | PubMedQA Acc | TAT-QA F1 | Banking77 Acc | NL2SQL EM | IFEval Strict | Safety Rate |
|-------|--------------|-----------|---------------|-----------|---------------|-------------|
| Baseline (4B) | 0.61 | 0.368 | 0.537 | 0.01 | 0.819 | 0.91 |
| PubMedQA | **0.81** | 0.000 | 0.222 | 0.02 | 0.665 | 0.91 |
| TAT-QA | 0.54 | **0.745** | 0.476 | 0.00 | 0.787 | 0.90 |
| Banking77 | 0.66 | 0.001 | **0.930** | 0.00 | 0.634 | 0.93 |
| NL2SQL | 0.61 | 0.001 | 0.427 | 0.02 | 0.540 | 0.88 |

**Observations:**
- Strong baseline: 4B model already has excellent IFEval (0.819) and safety (0.91)
- Solid task performance gains: pubmedqa +0.20, tat-qa F1 +0.38, banking77 +0.39, nl2sql +.06 (+.01 EM)
- tat-qa tanks when FT'ing for other tasks
- nl2sql is still tough (best is .02 EM), makes sense at 4b scale

Overall these settings work fine.


#### `Qwen3-8B`
Qwen3-8B is a "hybrid reasoning model", meaning that was trained to use special tokens to separate reasoning from solution, but that this behavior can be disabled by passing `enable_thinking=False` to `apply_chat_template()`, which will cause the chat template to render without an instruction to use the thinking tokens.

None of our SFT use cases use reasoning-style data, but we can still train on top of this model with non-reasoning data and get reasonable results. We should also synthesize some reasoning traces to create a reasoning-style version of nl2sql (this is the hardest use case and would likely benefit most from reasoning-style training).

First we establish whether existing defaults for Qwen3-8B FFT work well enough for these use cases. We will treat reasoning-style data for nl2sql as a P1 after hardened configs are established for other base models.

```sh
# with current ent defaults (minor edits for logging and disabling intermediate ckpts)
./scripts/enterprise/run_experiment.sh --model qwen3_8b --task pubmedqa --with-baseline
./scripts/enterprise/run_experiment.sh --model qwen3_8b --task tatqa
./scripts/enterprise/run_experiment.sh --model qwen3_8b --task banking77
./scripts/enterprise/run_experiment.sh --model qwen3_8b --task nl2sql

# transfer results to inspect locally
kubectl cp $NAMESPACE/$POD:/data/tim/evals/ent/qwen3_8b-ft ~/Downloads/qwen3_8b-ft
kubectl cp $NAMESPACE/$POD:/data/tim/evals/ent/baselines/20260121_210308_Qwen3-8B ~/Downloads/qwen3_8b-ft/20260121_210308_Qwen3-8B
```

| Model | PubMedQA Acc | TAT-QA F1 | Banking77 Acc | NL2SQL EM | IFEval Strict | Safety Rate |
|-------|--------------|-----------|---------------|-----------|---------------|-------------|
| Baseline (8B) | 0.11 | 0.009 | 0.156 | 0.00 | 0.338 | 0.97 |
| PubMedQA | **0.79** | 0.072 | 0.506 | 0.00 | 0.627 | 0.91 |
| TAT-QA | 0.42 | **0.773** | 0.526 | 0.00 | 0.617 | 0.92 |
| Banking77 | 0.18 | 0.025 | **0.936** | 0.00 | 0.464 | 0.92 |
| NL2SQL | 0.51 | 0.133 | 0.472 | 0.00 | 0.579 | 0.94 |

**Observations:**
- Baseline performs poorly across tasks, likely due to `<think>` token in outputs confusing eval parsing
- Strong task-specific gains from SFT: PubMedQA +0.68, TAT-QA F1 +0.76, Banking77 +0.78
- NL2SQL exact match stays at 0.00 - base model's `<think>` tokens may still affect SQL generation
- SFT improves IFEval (~0.3 --> 0.6), likely by training out the thinking behavior
- Safety remains high (0.91-0.97) across all configurations


8b will need some more work due to the reasoning behavior -- for default SFT config, we need to figure out how to disable reasoning from an oumi train config (need to someow pass `enable_thinking=False` to `apply_chat_template()`).


### Create master results file to track results across experiments [WiP]
Produce a big csv table that we import into this results tracker doc: https://docs.google.com/spreadsheets/d/1LmFGfJRtNp3hpP9dwD7FJ1uN7Z0I8cKX-Z1wZ5ZaW2I

Only setting this up starting with smollm2, so earlier runs don't have hyperparams and metrics in one place.

We need to do some messy backfilling to infer hyperparam settings for older runs (they're not saved along with the ckpts).

```sh
# backfill results from previous base models (one-time throwaway work)
# TODO clean this up before pushing:
python scripts/enterprise/build_master_results.py \
    --evals-dir /data/tim/evals/ent \
    --output /data/tim/evals/ent/master_results-backfill.csv

# TODO clean this up before pushing:
python configs/enterprise/ent-expts/append_results.py \
  --results /data/tim/evals/ent/smollm2_1.7b-ft/collated/results.csv \
  --checkpoints /data/tim/checkpoints \
  --master /data/tim/evals/enterprise_results_master.csv \
  --notes "SmolLM2 1.7B first run"
```