#!/bin/bash
#SBATCH --job-name=verl-conv-smoke
#SBATCH --nodelist=node-0
#SBATCH --no-requeue
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/workspace/aniruddh/verl-usersim/smoke_%j.out

# Do not set --cpus-per-task; Ray misdetects the cgroup CPU limit on this cluster.
cd /workspace/aniruddh/verl-usersim || exit 1
unset ROCR_VISIBLE_DEVICES
# This cluster is shared with non-SLURM workloads, so SLURM can hand out a GPU that
# already has ~70GB in use and vLLM dies at startup. Claim the node and pick a device
# that is actually free.
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', *' '$2 < 1000 {print $1; exit}')
if [ -z "$FREE_GPU" ]; then echo "[smoke] no free GPU on $(hostname)"; exit 1; fi
export CUDA_VISIBLE_DEVICES="$FREE_GPU"
echo "[smoke] using GPU $FREE_GPU"
source /workspace/aniruddh/nl2sqlvenv/bin/activate
# Make this checkout and the virtualenv available to Ray workers.
export PYTHONPATH="/workspace/aniruddh/verl-usersim/src:/workspace/aniruddh/nl2sqlvenv/lib/python3.12/site-packages"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_USE_V1=0
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export OUMI_RAY_NUM_CPUS=16
export TOKENIZERS_PARALLELISM=false
# Ray reward workers may use a different working directory.
export OUMI_JUDGE_CONFIG_PATH=/workspace/aniruddh/verl-usersim/configs/examples/verl_conversational/judge.yaml
export OPENAI_API_KEY="$(cat /workspace/aniruddh/verl-usersim/openai.key)"

echo "[smoke] host=$(hostname) python=$(which python) $(date -u +%H:%M:%S)"
python -c "import verl,oumi; print('verl',verl.__version__,'oumi',oumi.__file__)"
python -m oumi train -c configs/examples/verl_conversational/train.yaml
echo "OUMI_TRAIN_EXIT=$?"
