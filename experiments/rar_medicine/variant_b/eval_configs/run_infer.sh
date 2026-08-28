#!/usr/bin/env bash
# Run `oumi infer` for the untrained gemma-4-E2B-it and the Variant B GRPO policy on
# the RaR-Medicine 1000-sample test subset, one model per GPU, concurrently.
#
#   bash run_infer.sh                 # base on GPU 0, trained on GPU 1
#   BASE_GPU=4 TRAINED_GPU=5 bash run_infer.sh
#   ONLY=base bash run_infer.sh       # or ONLY=trained
#
# base    = infer_base.yaml    (vLLM, google/gemma-4-E2B-it), ~2 min
# trained = infer_trained.yaml (NATIVE transformers+PEFT, base + step-64 LoRA
#           adapter; vLLM's LoRA path is a no-op for gemma-4), ~15-25 min
#
# Builds the eval set if missing (prepare_eval_set.py). Logs go to
# logs/rar_medicine_infer_<base|trained>.log at the repo root; outputs land where
# the yamls' output_path points (output/rar_medicine_grpo_verl_variant_b/eval/outputs/).
# Re-running resumes: the engine skips conversations already in the output.
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/../../../.." && pwd)"
DATA_DIR="${REPO_ROOT}/output/rar_medicine_grpo_verl_variant_b/eval"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${DATA_DIR}/outputs"

BASE_GPU="${BASE_GPU:-0}"
TRAINED_GPU="${TRAINED_GPU:-1}"
ONLY="${ONLY:-both}"

# vLLM (base run): oumi initialises CUDA before vLLM forks its engine core; fork
# would die with "Cannot re-initialize CUDA in forked subprocess".
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [[ ! -s "${DATA_DIR}/test_1000.jsonl" ]]; then
  echo "building eval set ..."
  python "${EVAL_DIR}/prepare_eval_set.py"
fi

check_gpu_free() {
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' ')"
  if (( used > 1024 )); then
    echo "GPU $1 already has ${used} MiB in use; pick another (BASE_GPU/TRAINED_GPU) or FORCE=1" >&2
    [[ -n "${FORCE:-}" ]] || exit 1
  fi
}

run_one() {  # name gpu config
  local name="$1" gpu="$2" cfg="$3"
  local log="${LOG_DIR}/rar_medicine_infer_${name}.log"
  [[ -s "${log}" ]] && mv "${log}" "${log%.log}.$(date +%Y%m%d-%H%M%S).log"
  echo "[${name}] GPU ${gpu}, config ${cfg#${REPO_ROOT}/}, log ${log#${REPO_ROOT}/}"
  CUDA_VISIBLE_DEVICES="${gpu}" oumi infer -c "${cfg}" >"${log}" 2>&1
}

pids=()
if [[ "${ONLY}" == "both" || "${ONLY}" == "base" ]]; then
  check_gpu_free "${BASE_GPU}"
  run_one base "${BASE_GPU}" "${EVAL_DIR}/infer_base.yaml" & pids+=($!)
fi
if [[ "${ONLY}" == "both" || "${ONLY}" == "trained" ]]; then
  check_gpu_free "${TRAINED_GPU}"
  run_one trained "${TRAINED_GPU}" "${EVAL_DIR}/infer_trained.yaml" & pids+=($!)
fi

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || { echo "a run failed (pid ${pid}); see logs/" >&2; status=1; }
done

# Row counts, token stats, truncations, identical-to-base count. Also relabels
# metadata.finish_reason from the token count: the NATIVE engine marks every
# sequence in a batch `length` when any member hits the cap.
python "${EVAL_DIR}/summarize_outputs.py" --fix-finish-reason
exit "${status}"
