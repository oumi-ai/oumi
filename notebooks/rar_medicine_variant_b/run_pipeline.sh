#!/usr/bin/env bash
# End-to-end: rollouts for variant_b on the SAME 64 val prompts / seed as the
# gemma-4-E2B-it baseline in ../rar_medicine_judges/rollouts, judge them with
# the recommended judge (rubric_explicit + local gemma-4-E4B-it), then compare.
#
#   bash run_pipeline.sh              # GPU 0 for policy, GPU 1 for judge
#   POLICY_GPU=2 JUDGE_GPU=3 bash run_pipeline.sh
#   JUDGES="variant_b_training_reward:gpt-4.1-mini rubric_explicit:google/gemma-4-E4B-it" bash run_pipeline.sh
#
# JUDGES are VARIANT:MODEL pairs; the first is the training reward, the rest held-out.
# Local models run on JUDGE_GPU via vLLM; gpt-* models need OPENAI_API_KEY (repo .env).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JUDGES_DIR="$HERE/../rar_medicine_judges"
PY="${PY:-/root/miniconda3/envs/oumi/bin/python}"
POLICY_GPU="${POLICY_GPU:-0}"
JUDGE_GPU="${JUDGE_GPU:-1}"
MODEL_PATH="${MODEL_PATH:-$HERE/models/variant_b_merged}"
POLICY_NAME="${POLICY_NAME:-variant_b}"
BASELINE="${BASELINE:-gemma-4-E2B-it}"
JUDGES="${JUDGES:-variant_b_training_reward:gpt-4.1-mini rubric_explicit:google/gemma-4-E4B-it rubric_explicit:gpt-4.1}"

[ -e "$MODEL_PATH" ] || { echo "$MODEL_PATH missing: run download_model.py then merge_adapter.py"; exit 1; }

# 1. rollouts (same --seed/--num-prompts/--split as the baseline => identical prompt set)
if [ ! -f "$JUDGES_DIR/rollouts/$POLICY_NAME.jsonl" ]; then
  CUDA_VISIBLE_DEVICES=$POLICY_GPU $PY "$JUDGES_DIR/generate_rollouts.py" \
    --policy "$MODEL_PATH" --policy-name "$POLICY_NAME" \
    --split val --num-prompts 64 --num-generations 4 --seed 0 --max-new-tokens 512
fi

# 2. judge (cached per (policy, variant, judge))
for J in $JUDGES; do
  VARIANT="${J%%:*}"; MODEL="${J#*:}"
  case "$MODEL" in
    gpt-*) ENGINE=OPENAI; GPU_ENV="";;
    *)     ENGINE=VLLM;   GPU_ENV="CUDA_VISIBLE_DEVICES=$JUDGE_GPU";;
  esac
  for P in "$BASELINE" "$POLICY_NAME"; do
    env $GPU_ENV $PY "$JUDGES_DIR/run_judges.py" --policy "$P" --models "$MODEL" --engine $ENGINE --variants "$VARIANT"
  done
done

# 3. compare
$PY "$HERE/compare.py" --baseline "$BASELINE" --policy "$POLICY_NAME" --judges $JUDGES
