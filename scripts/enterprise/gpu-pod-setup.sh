#!/bin/bash

# Ad hoc setup script for GPU pod
# Usage: ./scripts/enterprise/gpu-pod-setup.sh --hf-token <your_token>

set -e

# get HF token as an arg
HF_TOKEN=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --hf-token)
      HF_TOKEN="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --hf-token <huggingface_token>"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --hf-token <huggingface_token>"
      exit 1
      ;;
  esac
done

if [ -z "$HF_TOKEN" ]; then
  echo "Error: --hf-token is required"
  echo "Usage: $0 --hf-token <huggingface_token>"
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

echo "[1/6] Setting up local copy of oumi repo..."
cd /data/tim/code/oumi
# git checkout lefft/ent-train-expts
pip install -e .

# Extra dependencies for lm-harness
echo "[2/6] Installing lm-harness dependencies..."
pip install langdetect immutabledict

# QoL utilities
echo "[3/6] Installing system utilities..."
apt update
apt install -y jq tmux wget build-essential

# Install yq
echo "[4/6] Installing yq..."
mkdir -p /data/tim/bin
wget -q https://github.com/mikefarah/yq/releases/download/v4.50.1/yq_linux_amd64 -O /data/tim/bin/yq
chmod +x /data/tim/bin/yq

# Setup shell aliases and env vars
echo "[5/6] Setting up shell environment, source /data/tim/.bashrc to once complete..."
cat >> /data/tim/.bashrc << 'EOF'
# GPU pod QoL aliases
alias tmn="tmux new -s"
alias tma="tmux attach -t"
alias tml="tmux ls"
export DATASET_DIR=/data/tim/code/oumi/data/enterprise
export PATH="$PATH:/data/tim/bin"
EOF


# HuggingFace login
echo "[6/6] Logging into HuggingFace..."
huggingface-cli login --token "$HF_TOKEN"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
