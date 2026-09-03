#!/usr/bin/env bash
# Scores both Variant B eval arms (base, trained) with the training judge via
# `oumi evaluate`, then runs the paired comparison. No GPU: the responses already
# exist in ../eval_outputs and the configs use the OPENAI engine as a stand-in.
#
# Usage (any cwd):
#   bash experiments/rar_medicine/variant_b/rar_eval/run_eval.sh [--num_samples N] [--skip-compare]
#
# --num_samples N   pilot on a seeded N-prompt subset (the same subset in both
#                   arms). Judgments are cached per conversation, so the full
#                   run afterwards only pays for the remaining prompts.
# --skip-compare    do not run compare_runs.py at the end.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
cd "${REPO_ROOT}"

NUM_SAMPLES=""
SKIP_COMPARE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
    --skip-compare) SKIP_COMPARE=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Judge key. The repo .env stores it as INTERNAL_OPENAI_API_KEY; judge_config.yaml
# reads OPENAI_API_KEY. Copy that one variable rather than sourcing the whole
# file (it also holds unrelated credentials). An existing OPENAI_API_KEY wins.
if [[ -z "${OPENAI_API_KEY:-}" && -f "${REPO_ROOT}/.env" ]]; then
  _key="$(sed -n 's/^INTERNAL_OPENAI_API_KEY=//p' "${REPO_ROOT}/.env" | head -n1)"
  _key="${_key%\"}"; _key="${_key#\"}"; _key="${_key%\'}"; _key="${_key#\'}"
  [[ -n "${_key}" ]] && export OPENAI_API_KEY="${_key}"
  unset _key
fi
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY, or add INTERNAL_OPENAI_API_KEY to ${REPO_ROOT}/.env}"

EXTRA_ARGS=()
if [[ -n "${NUM_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--tasks.0.num_samples "${NUM_SAMPLES}")
fi

for arm in base trained; do
  echo "=== rar_medicine: ${arm} ==="
  oumi evaluate -c "${HERE}/eval_${arm}.yaml" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
done

if [[ "${SKIP_COMPARE}" -eq 0 ]]; then
  python "${HERE}/compare_runs.py"
fi
