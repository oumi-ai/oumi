#!/usr/bin/env bash
# Launch the isolated Gemma 4 LoRA/GRPO experiment. By default this first
# applies vllm_gemma4_lora_alias.patch to vLLM in the active Python environment.
set -euo pipefail

LORA_EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT_B_DIR="$(cd "${LORA_EXPERIMENT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${VARIANT_B_DIR}/../../.." && pwd)"
PATCH_FILE="${LORA_EXPERIMENT_DIR}/vllm_gemma4_lora_alias.patch"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Patch the package on disk so the fix is visible in every Ray/vLLM process,
# not only in the Oumi driver. The checks make this idempotent and fail closed
# if the installed vLLM source differs from the expected 0.19.1 implementation.
if [[ "${SKIP_VLLM_PATCH:-0}" != "1" ]]; then
  VLLM_SITE="$(${PYTHON_BIN} -c 'from pathlib import Path; import vllm; print(Path(vllm.__file__).resolve().parent.parent)')"
  if patch --dry-run --forward --batch -p1 -d "${VLLM_SITE}" <"${PATCH_FILE}" >/dev/null 2>&1; then
    patch --forward --batch -p1 -d "${VLLM_SITE}" <"${PATCH_FILE}"
    echo "applied Gemma 4 LoRA alias patch to ${VLLM_SITE}/vllm/lora/model_manager.py"
  elif patch --dry-run --reverse --batch -p1 -d "${VLLM_SITE}" <"${PATCH_FILE}" >/dev/null 2>&1; then
    echo "Gemma 4 LoRA alias patch is already applied"
  else
    echo "refusing to launch: vLLM model_manager.py matches neither side of ${PATCH_FILE}" >&2
    echo "inspect the installed vLLM version, or set SKIP_VLLM_PATCH=1 only if it already has an equivalent fix" >&2
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

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

if [[ -z "${OPENAI_API_KEY:-}" && -f "${REPO_ROOT}/.env" ]]; then
  _key="$(sed -n 's/^INTERNAL_OPENAI_API_KEY=//p' "${REPO_ROOT}/.env" | head -n1)"
  _key="${_key%\"}"; _key="${_key#\"}"; _key="${_key%\'}"; _key="${_key#\'}"
  [[ -n "${_key}" ]] && export OPENAI_API_KEY="${_key}"
  unset _key
fi
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY, or add INTERNAL_OPENAI_API_KEY to ${REPO_ROOT}/.env}"

export OUMI_EXTRA_DEPS_FILE="${VARIANT_B_DIR}/oumi_extra_deps.txt"
export PYTHONPATH="${VARIANT_B_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

LOG_DIR="${REPO_ROOT}/logs"
RUN_NAME="${RUN_NAME:-medqa_gemma4-e2b-it_lora_alias_patch}"
LOG="${LOG_DIR}/${RUN_NAME}.log"
mkdir -p "${LOG_DIR}"
[[ -s "${LOG}" ]] && mv "${LOG}" "${LOG%.log}.$(date +%Y%m%d-%H%M%S).log"

echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "logging to ${LOG}"
exec oumi train -c "${LORA_EXPERIMENT_DIR}/train_verl.yaml" "$@" >"${LOG}" 2>&1
