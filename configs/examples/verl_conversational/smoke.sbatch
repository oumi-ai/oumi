#!/bin/bash
#SBATCH --job-name=verl-conv-smoke
#SBATCH --nodelist=node-0
#SBATCH --no-requeue
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=200G
#SBATCH --time=00:40:00
#SBATCH --output=/workspace/aniruddh/verl-conv/smoke_%j.out

# Do not set --cpus-per-task; Ray misdetects the cgroup CPU limit on this cluster.
cd /workspace/aniruddh/verl-conv || exit 1
unset ROCR_VISIBLE_DEVICES
source /workspace/aniruddh/nl2sqlvenv/bin/activate
# Make this checkout and the virtualenv available to Ray workers.
export PYTHONPATH="/workspace/aniruddh/verl-conv/src:/workspace/aniruddh/nl2sqlvenv/lib/python3.12/site-packages"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_USE_V1=0
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export OUMI_RAY_NUM_CPUS=16
export TOKENIZERS_PARALLELISM=false
# Ray reward workers may use a different working directory.
export OUMI_JUDGE_CONFIG_PATH=/workspace/aniruddh/verl-conv/configs/examples/verl_conversational/judge.yaml
export OPENAI_API_KEY="$(cat /workspace/aniruddh/verl-conv/openai.key)"

echo "[smoke] host=$(hostname) python=$(which python) $(date -u +%H:%M:%S)"
python -c "import verl,oumi; print('verl',verl.__version__,'oumi',oumi.__file__)"
oumi train -c configs/examples/verl_conversational/train.yaml
echo "OUMI_TRAIN_EXIT=$?"
