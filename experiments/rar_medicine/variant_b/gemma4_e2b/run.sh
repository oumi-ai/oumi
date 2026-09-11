#!/usr/bin/env bash
# Launch RaR-Medicine GRPO training (gemma-4-E2B-it, full FT) with the
# Variant B gpt-4.1-mini judge reward.
#
# After the 2026-09-10 merge with main, the dataset (anisha2102/RaR-Medicine),
# the reward (rar_medicine_verl) and the gemma-4 FSDP rank-buffer sync all ship
# inside oumi, so no extra-deps module or external_lib patch is needed. The
# gemma-4 cache-less forward fix comes from transformers==5.5.2.
#
# Pins the run to GPUs 0-3 (the yaml's trainer.n_gpus_per_node is 4 to match),
# takes the judge's OpenAI key from the repo-root .env, and writes the full
# driver + Ray worker output to logs/<run_name>.log at the repo root.
#
# Safe to re-run: verl auto-resumes from the newest checkpoint under the
# yaml's output_dir. The previous attempt's log is kept, not overwritten.
#
# Usage (from any directory, inside the `oumi` conda env):
#   bash experiments/rar_medicine/variant_b/gemma4_e2b/run.sh
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash run.sh        # other GPUs
#   bash run.sh --training.max_steps 10             # extra oumi overrides
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUN_DIR}/../../../.." && pwd)"

# Environment sanity: oumi, verl, vllm and flash-attn must be importable.
if ! python -c "import oumi, verl, vllm, flash_attn" 2>/dev/null; then
  echo "oumi/verl/vllm/flash_attn not importable from $(command -v python)." >&2
  echo "  run: conda activate oumi   (and pip install flash-attn --no-build-isolation)" >&2
  exit 1
fi
tf_ver="$(python -c 'import transformers; print(transformers.__version__)')"
if [[ "${tf_ver}" != "5.5.2" ]]; then
  echo "transformers==${tf_ver}; 5.5.2 is required for correct gemma-4 cache-less forwards." >&2
  echo "  run: pip install transformers==5.5.2   (FORCE_TF=1 to override)" >&2
  [[ -z "${FORCE_TF:-}" ]] && exit 1
fi

# GPUs. Ray sizes its GPU pool from CUDA_VISIBLE_DEVICES at ray.init(), so this
# must be exported before `oumi train` starts and must agree with
# trainer.n_gpus_per_node in the yaml.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Pre-flight: refuse to start on a GPU that already has memory in use (a hung
# previous attempt keeps ~13.6 GiB of FSDP shards per GPU; another job's vLLM
# can hold 70+ GiB, and the run OOMs at the first weight sync). FORCE=1 skips.
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

# Judge key. Copy only OPENAI_API_KEY out of the repo .env rather than sourcing
# the whole file: everything exported here is inherited by every Ray worker,
# and .env also holds unrelated credentials. A key already in the env wins.
if [[ -z "${OPENAI_API_KEY:-}" && -f "${REPO_ROOT}/.env" ]]; then
  _key="$(sed -n 's/^OPENAI_API_KEY=//p' "${REPO_ROOT}/.env" | head -n1)"
  _key="${_key%\"}"; _key="${_key#\"}"; _key="${_key%\'}"; _key="${_key#\'}"
  [[ -n "${_key}" ]] && export OPENAI_API_KEY="${_key}"
  unset _key
fi
# Fail fast instead of mid-rollout.
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY, or add it to ${REPO_ROOT}/.env}"

# Judge prompt/model. oumi's built-in rar_medicine_verl reward defaults to the
# repo-relative configs/examples/grpo_verl_medqa/judge.yaml, which only resolves
# when the cwd is the repo root. Point it at this folder's copy (identical
# content) by absolute path so the reward workers find it from anywhere.
export RAR_JUDGE_CONFIG="${RAR_JUDGE_CONFIG:-${RUN_DIR}/judge_config.yaml}"

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
echo "judge config: ${RAR_JUDGE_CONFIG}"
echo "logging to ${LOG}"
cd "${REPO_ROOT}"
exec oumi train -c "${RUN_DIR}/train_verl.yaml" "$@" >"${LOG}" 2>&1
