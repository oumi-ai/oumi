#!/usr/bin/env bash
# Launch RaR-Medicine GRPO training (Qwen3.5-4B, full FT) with the
# Variant B gpt-4.1-mini judge reward.
#
# After the 2026-09-10 merge with main, the dataset (anisha2102/RaR-Medicine),
# the reward (rar_medicine_verl) and the FSDP rank-buffer sync all ship
# inside oumi, so no extra-deps module or external_lib patch is needed.
# Qwen3.5 linear-attention layers use fast kernels when flash-linear-attention
# and causal-conv1d are installed; otherwise transformers falls back to a slow
# pure-torch path (works, ~2-3x slower actor/ref forwards).
#
# Pins the run to GPUs 0-3 (the yaml's trainer.n_gpus_per_node is 4 to match),
# takes the judge's OpenAI key from the repo-root .env, and writes the full
# driver + Ray worker output to logs/<run_name>.log at the repo root.
#
# Safe to re-run: verl auto-resumes from the newest checkpoint under the
# yaml's output_dir. The previous attempt's log is kept, not overwritten.
#
# Usage (from any directory, inside the `oumi` conda env):
#   bash experiments/rar_medicine/variant_b/qwen3.5-4B/run.sh
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash run.sh        # other GPUs
#   bash run.sh --training.max_steps 10             # extra oumi overrides
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUN_DIR}/../../../.." && pwd)"

# Environment sanity: oumi, verl and vllm must be importable.
if ! python -c "import oumi, verl, vllm" 2>/dev/null; then
  echo "oumi/verl/vllm not importable from $(command -v python); run: conda activate oumi" >&2
  exit 1
fi
# verl loads Qwen3.5's image processor (the checkpoint is vision-language) and
# imports qwen_vl_utils before every rollout, even for text-only prompts.
if ! python -c "import qwen_vl_utils" 2>/dev/null; then
  echo "qwen_vl_utils missing (verl needs it for Qwen3.5 rollouts); run: pip install qwen-vl-utils" >&2
  exit 1
fi
# Fast kernels for the Gated DeltaNet (linear attention) layers. Optional but
# strongly recommended: without them every actor/ref forward runs the pure-torch
# fallback. pip install flash-linear-attention causal-conv1d
if ! python -c "import fla, causal_conv1d" 2>/dev/null; then
  echo "WARNING: fla/causal_conv1d not installed; Qwen3.5 linear attention will use the slow torch fallback." >&2
elif ! python -c "import tilelang" 2>/dev/null; then
  # fla refuses its gated delta-rule backward on Hopper with Triton 3.4-3.7.0
  # (fla issue #640; torch 2.10 pins triton 3.6.0) unless tilelang provides it.
  echo "tilelang missing: fla's Gated DeltaNet backward will raise on H100 at the first update. run: pip install tilelang" >&2
  exit 1
fi
tf_ver="$(python -c 'import transformers; print(transformers.__version__)')"
if [[ "${tf_ver}" != "5.5.2" ]]; then
  echo "transformers==${tf_ver}; 5.5.2 is the version this recipe was set up with." >&2
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
RUN_NAME="${RUN_NAME:-medqa_qwen3.5-4b_fullft}"
LOG="${LOG_DIR}/${RUN_NAME}.log"
mkdir -p "${LOG_DIR}"
# Keep the previous attempt's log as <name>.<timestamp>.log.
[[ -s "${LOG}" ]] && mv "${LOG}" "${LOG%.log}.$(date +%Y%m%d-%H%M%S).log"

echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "judge config: ${RAR_JUDGE_CONFIG}"
echo "logging to ${LOG}"
cd "${REPO_ROOT}"
exec oumi train -c "${RUN_DIR}/train_verl.yaml" "$@" >"${LOG}" 2>&1
