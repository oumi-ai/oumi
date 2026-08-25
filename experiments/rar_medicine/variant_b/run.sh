#!/usr/bin/env bash
# Launch RaR-Medicine GRPO training with the Variant B gpt-5-mini judge reward.
set -euo pipefail

VARIANT_B_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The judge needs the OpenAI API key; fail fast instead of mid-rollout.
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the gpt-5-mini judge}"

# Makes the oumi driver import rar_medicine_grpo.py so the dataset and reward
# registrations exist before the config is resolved.
export OUMI_EXTRA_DEPS_FILE="${VARIANT_B_DIR}/oumi_extra_deps.txt"

# verl's reward-loop Ray actors import the reward as `pkg://rar_medicine_grpo`;
# they inherit this environment, so the module dir must be on PYTHONPATH.
export PYTHONPATH="${VARIANT_B_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec oumi train -c "${VARIANT_B_DIR}/train_verl.yaml" "$@"
