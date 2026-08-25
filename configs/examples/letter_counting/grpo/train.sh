#!/usr/bin/env bash
# Launch the letter-counting and countdown verl GRPO runs on separate GPU
# sets, streaming each run's output to its own log file under logs/.
#
# Relaunch-safe:
#   * An existing non-empty log is rotated to <name>.<timestamp>.log instead of
#     being overwritten.
#   * verl auto-resumes from the latest checkpoint under <output_dir>/verl_output
#     (trainer.resume_mode=auto is the default). When it does, the same W&B run
#     is resumed too (WANDB_RUN_ID + WANDB_RESUME=allow), so metrics land on the
#     existing charts instead of a second run. The run id is persisted in
#     <output_dir>/wandb_run_id; delete that file to force a new W&B run.
#     Note: W&B ignores logged steps <= the run's last logged step, so the steps
#     between the resumed checkpoint and the crash are dropped from W&B (they are
#     still in the log file); the charts stay continuous.
set -u
cd "$(dirname "$0")/../../../.."
mkdir -p logs

rotate_log() {
  local log="$1"
  if [[ -s "$log" ]]; then
    mv "$log" "${log%.log}.$(date +%Y%m%d-%H%M%S).log"
  fi
}

# Print training.output_dir from an oumi yaml (single-line value, optionally quoted).
yaml_output_dir() {
  sed -nE 's/^[[:space:]]*output_dir:[[:space:]]*"?([^"[:space:]]+)"?[[:space:]]*$/\1/p' "$1" | head -1
}

# wandb_env <name> <output_dir>: print W&B env assignments for the run (status to
# stderr). Reuses the persisted run id when verl has a checkpoint to resume from;
# otherwise mints a fresh id so a from-scratch run never attaches to an old W&B run.
wandb_env() {
  local name="$1" out_dir="$2"
  local id_file="$out_dir/wandb_run_id"
  local tracker="$out_dir/verl_output/latest_checkpointed_iteration.txt"
  mkdir -p "$out_dir"
  if [[ -f "$tracker" && -s "$id_file" ]]; then
    echo "$name: resuming from global_step_$(<"$tracker"), continuing W&B run $(<"$id_file")" >&2
    echo "WANDB_RUN_ID=$(<"$id_file") WANDB_RESUME=allow"
    return
  fi
  tr -dc a-z0-9 </dev/urandom | head -c 8 >"$id_file"
  echo "$name: fresh run, new W&B run $(<"$id_file")" >&2
  echo "WANDB_RUN_ID=$(<"$id_file")"
}

# launch <NAME> <yaml> <log>: start `oumi train` in the background, set <NAME>_PID.
# Prefix the call with VAR=VALUE (e.g. CUDA_VISIBLE_DEVICES=4,5,6,7) to scope env to it.
launch() {
  local name="$1" yaml="$2" log="$3"
  local out_dir wandb
  out_dir="$(yaml_output_dir "$yaml")"
  if [[ -z "$out_dir" ]]; then
    echo "$name: could not read training.output_dir from $yaml" >&2
    exit 1
  fi
  wandb="$(wandb_env "$name" "$out_dir")"
  rotate_log "$log"
  # shellcheck disable=SC2086  # $wandb holds VAR=VALUE words on purpose
  env $wandb oumi train -c "$yaml" >"$log" 2>&1 &
  printf -v "${name}_PID" '%s' "$!"
  echo "$name: PID $! -> $log"
}

launch LETTER configs/examples/letter_counting/grpo/train_verl_v2_longer.yaml logs/train_verl_v2_longer.log

# To run the countdown job alongside on GPUs 4-7, uncomment:
# CUDA_VISIBLE_DEVICES=4,5,6,7 launch COUNTDOWN configs/examples/grpo_verl_countdown/train.yaml logs/grpo_verl_countdown.log

wait "${LETTER_PID}"
LETTER_RC=$?
echo "LETTER exit code: ${LETTER_RC}"

COUNTDOWN_RC=0
if [[ -n "${COUNTDOWN_PID:-}" ]]; then
  wait "${COUNTDOWN_PID}"
  COUNTDOWN_RC=$?
  echo "COUNTDOWN exit code: ${COUNTDOWN_RC}"
fi

exit $(( LETTER_RC != 0 || COUNTDOWN_RC != 0 ? 1 : 0 ))
