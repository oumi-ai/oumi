#!/bin/bash

# Ad hoc setup script for GPU pod
# Usage: ./scripts/enterprise/gpu-pod-setup.sh --hf-token <token> --wandb-token <token>

set -e

# Parse arguments
HF_TOKEN=""
WANDB_TOKEN=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --hf-token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --wandb-token)
      WANDB_TOKEN="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --hf-token <huggingface_token> --wandb-token <wandb_api_key>"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --hf-token <huggingface_token> --wandb-token <wandb_api_key>"
      exit 1
      ;;
  esac
done

if [ -z "$HF_TOKEN" ]; then
  echo "Error: --hf-token is required"
  echo "Usage: $0 --hf-token <huggingface_token> --wandb-token <wandb_api_key>"
  exit 1
fi

if [ -z "$WANDB_TOKEN" ]; then
  echo "Error: --wandb-token is required"
  echo "Usage: $0 --hf-token <huggingface_token> --wandb-token <wandb_api_key>"
  exit 1
fi

echo "=============================================="
echo "GPU Pod Setup Script"
echo "=============================================="

# Clone and setup oumi (don't do this if using copy synced from local)
# echo "[1/6] Setting up oumi repository..."
# mkdir -p /data/tim/code && cd /data/tim/code
# if [ ! -d "oumi" ]; then
#   git clone https://github.com/oumi-ai/oumi.git
# fi

echo "Setting up local copy of oumi repo..."
cd /data/tim/code/oumi
# git checkout lefft/ent-train-expts
pip install -e .

# Extra dependencies for lm-harness
echo "Installing lm-harness dependencies..."
pip install langdetect immutabledict

# Without this, oumi tune jobs will die with:
# deepspeed.ops.op_builder.builder.MissingCUDAException: CUDA_HOME does not exist, unable to compile CUDA op(s)
pip uninstall deepspeed -y

# QoL utilities
echo "Installing system utilities..."
apt update
apt install -y jq tmux wget build-essential
pip install visidata

# Install yq
echo "Installing yq..."
mkdir -p /data/tim/bin
wget -q https://github.com/mikefarah/yq/releases/download/v4.50.1/yq_linux_amd64 -O /data/tim/bin/yq
chmod +x /data/tim/bin/yq

# HuggingFace login
echo "Logging into HuggingFace..."
huggingface-cli login --token "$HF_TOKEN"

# Setup bashrc if it doesn't already exist
if [ ! -f /data/tim/.bashrc ]; then
  echo "Creating .bashrc..."
  cat > /data/tim/.bashrc << EOF
# GPU pod QoL aliases
alias tmn="tmux new -s"
alias tma="tmux attach -t"
alias tml="tmux ls"
export DATASET_DIR=/data/tim/code/oumi/data/enterprise
export PATH="\$PATH:/data/tim/bin"
export HF_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="$WANDB_TOKEN"
wandb login
EOF
else
  echo "Skipping .bashrc setup (already exists)"
fi


python -c "import torch; print(f'\n\n\nAvailable GPUs: {torch.cuda.device_count()}\n')"

echo "=============================================="
echo "Setup complete!"
echo "=============================================="
