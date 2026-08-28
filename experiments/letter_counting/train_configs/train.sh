#!/usr/bin/env bash
# Launch the letter-counting verl GRPO run, logging to logs/.
#
# Safe to re-run: verl auto-resumes from the newest checkpoint under
# $OUT_DIR/verl_output, and the W&B run is resumed with it so the metrics stay on
# one set of charts. The previous attempt's log is kept, not overwritten.
set -u
cd "$(dirname "$0")/../../../.."

CONFIG=configs/examples/letter_counting/grpo/train_verl_v2_longer.yaml
OUT_DIR=output/letter_counting.grpo_verl_v2.bigger_dataset
LOG=logs/train_verl_v2_longer.log

mkdir -p logs "$OUT_DIR"

# Keep the previous attempt's log as <name>.<timestamp>.log.
[[ -s "$LOG" ]] && mv "$LOG" "${LOG%.log}.$(date +%Y%m%d-%H%M%S).log"

# Reuse the W&B run id only when verl has a checkpoint to resume from, so a
# from-scratch run never appends to an old run. Delete $OUT_DIR/wandb_run_id to
# force a brand-new W&B run.
TRACKER="$OUT_DIR/verl_output/latest_checkpointed_iteration.txt"
ID_FILE="$OUT_DIR/wandb_run_id"
if [[ -f "$TRACKER" && -s "$ID_FILE" ]]; then
  export WANDB_RUN_ID="$(<"$ID_FILE")" WANDB_RESUME=allow
  echo "resuming from global_step_$(<"$TRACKER"), continuing W&B run $WANDB_RUN_ID"
else
  export WANDB_RUN_ID="$(tr -dc a-z0-9 </dev/urandom | head -c 8)"
  echo "$WANDB_RUN_ID" >"$ID_FILE"
  echo "starting fresh, new W&B run $WANDB_RUN_ID"
  # Training resumes but W&B does not: the id of the run that produced these
  # checkpoints is gone. Put it back in $ID_FILE to keep one set of charts.
  [[ -f "$TRACKER" ]] && echo "WARNING: resuming training from global_step_$(<"$TRACKER") but $ID_FILE was missing, so W&B starts a NEW run"
fi

echo "logging to $LOG"
oumi train -c "$CONFIG" >"$LOG" 2>&1
