#!/usr/bin/env bash
# Launch RaR-Medicine GRPO training with the Variant B gpt-5-mini judge reward.
#
# Pins the run to GPUs 0-3 (the yaml's trainer.n_gpus_per_node is 4 to match),
# takes the judge's OpenAI key from the repo-root .env, and writes the full
# driver + Ray worker output to logs/<run_name>.log at the repo root.
#
# Safe to re-run: verl auto-resumes from the newest checkpoint under the
# yaml's output_dir. The previous attempt's log is kept, not overwritten.
set -euo pipefail

VARIANT_B_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${VARIANT_B_DIR}/../../.." && pwd)"

# GPUs. Ray sizes its GPU pool from CUDA_VISIBLE_DEVICES at ray.init(), so this
# must be exported before `oumi train` starts and must agree with
# trainer.n_gpus_per_node in the yaml. Override on the command line to move the
# run: CUDA_VISIBLE_DEVICES=4,5,6,7 bash run.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Pre-flight: refuse to start on a GPU that already has memory in use. A hung
# previous attempt (e.g. vLLM died at start and the trainer sat waiting) keeps
# ~13.6 GiB of FSDP shards per GPU; another job's vLLM can hold 70+ GiB. Either
# way the budget in GPU_MEMORY.md no longer holds and the run OOMs at the first
# weight sync. FORCE=1 skips the check.
if [[ -z "${FORCE:-}" ]]; then
  busy=""
  while IFS=, read -r idx used; do
    used="${used// /}"
    (( used > 1024 )) && busy="${busy} GPU${idx// /}=${used}MiB"
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES}")
  if [[ -n "${busy}" ]]; then
    echo "refusing to launch: memory already in use on${busy}" >&2
    echo "  (stale run? check: pgrep -af 'oumi train|VLLM::EngineCore|vLLMHttpServer'; FORCE=1 to override)" >&2
    exit 1
  fi
fi

# Judge key. The repo .env stores it as INTERNAL_OPENAI_API_KEY; the reward
# reads OPENAI_API_KEY. Copy that one variable across rather than sourcing the
# whole file: everything exported here is inherited by every Ray worker, and
# .env also holds billing/database credentials that have no business there.
# An OPENAI_API_KEY already in the environment wins.
if [[ -z "${OPENAI_API_KEY:-}" && -f "${REPO_ROOT}/.env" ]]; then
  _key="$(sed -n 's/^INTERNAL_OPENAI_API_KEY=//p' "${REPO_ROOT}/.env" | head -n1)"
  _key="${_key%\"}"; _key="${_key#\"}"; _key="${_key%\'}"; _key="${_key#\'}"
  [[ -n "${_key}" ]] && export OPENAI_API_KEY="${_key}"
  unset _key
fi
# Fail fast instead of mid-rollout.
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY, or add INTERNAL_OPENAI_API_KEY to ${REPO_ROOT}/.env}"

# Makes the oumi driver import rar_medicine_grpo.py so the dataset and reward
# registrations exist before the config is resolved.
export OUMI_EXTRA_DEPS_FILE="${VARIANT_B_DIR}/oumi_extra_deps.txt"

# verl's reward-loop Ray actors import the reward as `pkg://rar_medicine_grpo`;
# they inherit this environment, so the module dir must be on PYTHONPATH.
export PYTHONPATH="${VARIANT_B_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Logging. Ray forwards worker stdout/stderr to the driver, so one redirect
# captures the trainer, the FSDP workers and the vLLM servers.
LOG_DIR="${REPO_ROOT}/logs"
# RUN_NAME only names the log file; the yaml's training.run_name (or a
# --training.run_name override passed through "$@") names the wandb run.
RUN_NAME="${RUN_NAME:-medqa_gemma4-e2b-it_fullft}"
LOG="${LOG_DIR}/${RUN_NAME}.log"
mkdir -p "${LOG_DIR}"
# Keep the previous attempt's log as <name>.<timestamp>.log.
[[ -s "${LOG}" ]] && mv "${LOG}" "${LOG%.log}.$(date +%Y%m%d-%H%M%S).log"

echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "logging to ${LOG}"
exec oumi train -c "${VARIANT_B_DIR}/train_verl.yaml" "$@" >"${LOG}" 2>&1
