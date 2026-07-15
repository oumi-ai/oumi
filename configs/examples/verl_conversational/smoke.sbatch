#!/bin/bash
#SBATCH --job-name=verl-conv-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --output=/workspace/aniruddh/verl-conv/smoke_%j.out

cd /workspace/aniruddh/verl-conv || exit 1
echo "[smoke] starting $(date -u +%H:%M:%S)"

# Shadow the platform's system-wide oumi with this branch's checkout.
export PYTHONPATH="$PWD/src:$PYTHONPATH"
# User-sim + judge are OpenAI engines; key shipped from the laptop's ~/.zshrc.
export OPENAI_API_KEY="$(cat /workspace/aniruddh/verl-conv/openai.key)"
# RunPod SLURM gotchas.
unset ROCR_VISIBLE_DEVICES
# Policy model (Llama-3.2-1B-Instruct) is pre-cached; stay offline for HF.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python3 -c "import verl,oumi,sys; print('verl',verl.__version__,'oumi',oumi.__file__)" 2>&1 | tail -1
python3 -m oumi train -c configs/examples/verl_conversational/train.yaml
echo "OUMI_TRAIN_EXIT=$?"
