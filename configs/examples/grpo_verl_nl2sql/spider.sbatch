#!/bin/bash
#SBATCH --job-name=nl2sql_spider
# node-0 only: /workspace is per-node, and the Spider DBs + HF cache + the rps
# staging dir all live on node-0. node-1 has neither.
#SBATCH --nodelist=node-0
#SBATCH --no-requeue
# Whole node so the cgroup can reach any GPU: a non-SLURM process squats on one
# card, and SLURM would otherwise hand us that one. We pin a free card below.
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=nl2sql_spider_%j.out

# NOTE: never `set -x` here — it would echo WANDB_API_KEY into the job log.
set -uo pipefail

# --- run size: edit these two lines to switch smoke <-> full ------------------
STEPS="${STEPS:-30}"
TRAIN_LIMIT="${TRAIN_LIMIT:-1024}"
VAL_LIMIT="${VAL_LIMIT:-256}"
# -----------------------------------------------------------------------------

unset ROCR_VISIBLE_DEVICES
source /workspace/aniruddh/nl2sqlvenv/bin/activate
# prepend venv site-packages so Ray workers (system python) pick up trl<0.20
# (verl 0.7.1 imports trl.AutoModelForCausalLMWithValueHead, gone in trl>=1.0)
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:/workspace/aniruddh/nl2sqlvenv/lib/python3.12/site-packages"
export HF_HOME=/workspace/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_USE_V1=0
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export OUMI_RAY_NUM_CPUS=16   # image misreports ~190 CPUs -> ray.init hangs

# wandb: key read from a 600 file, never printed.
export WANDB_API_KEY="$(cat /workspace/aniruddh/wandb.key)"
export WANDB_PROJECT=oumi-nl2sql-verl
export WANDB_DIR=/workspace/aniruddh/wandb

cd "$SLURM_SUBMIT_DIR"

# Pin the first physically-idle card. SLURM tracks only its own allocations, so a
# non-SLURM squatter is invisible to it; without this we can land on a full GPU.
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
FREE_GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | awk -F', ' '$2 < 1000 {print $1; exit}')"
if [[ -z "$FREE_GPU" ]]; then echo "FATAL: no idle GPU on $(hostname)"; exit 1; fi
export CUDA_VISIBLE_DEVICES="$FREE_GPU"

echo "host=$(hostname) cwd=$(pwd) python=$(which python)"
echo "steps=$STEPS train_limit=$TRAIN_LIMIT val_limit=$VAL_LIMIT gpu=$FREE_GPU"

# Build Spider tool-agent rows (stdlib only, fast).
python scripts/datasets/build_spider_tool_agent.py \
  --spider-root /workspace/spider_raw/spider_data \
  --db-root /workspace/spider/database \
  --out-dir /workspace/aniruddh/spider_data \
  --train-limit "$TRAIN_LIMIT" --val-limit "$VAL_LIMIT"
echo "rows: train=$(wc -l < /workspace/aniruddh/spider_data/train.jsonl) val=$(wc -l < /workspace/aniruddh/spider_data/val.jsonl)"

python -c "import verl,oumi; print('verl',verl.__version__,'oumi ok')"

oumi train -c configs/examples/grpo_verl_nl2sql/spider_train.yaml \
  --training.max_steps "$STEPS"
echo "EXIT=$?"
