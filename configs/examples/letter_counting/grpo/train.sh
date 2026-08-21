#!/usr/bin/env bash
# Launch the letter-counting and countdown verl GRPO runs on separate GPU
# sets, streaming each run's output to its own log file under logs/.
set -u
cd "$(dirname "$0")/../../../.."
mkdir -p logs

oumi train -c configs/examples/letter_counting/grpo/train_verl.yaml \
  > logs/letter_counting_grpo_verl.log 2>&1 &
LETTER_PID=$!

# CUDA_VISIBLE_DEVICES=4,5,6,7 oumi train -c configs/examples/grpo_verl_countdown/train.yaml \
#   > logs/grpo_verl_countdown.log 2>&1 &
# COUNTDOWN_PID=$!

echo "letter_counting: PID ${LETTER_PID} -> logs/letter_counting_grpo_verl.log"
# echo "countdown:       PID ${COUNTDOWN_PID} -> logs/grpo_verl_countdown.log"

wait "${LETTER_PID}"
LETTER_RC=$?
wait "${COUNTDOWN_PID}"
# COUNTDOWN_RC=$?
echo "letter_counting exit code: ${LETTER_RC}"
# echo "countdown exit code: ${COUNTDOWN_RC}"
# exit $(( LETTER_RC != 0 || COUNTDOWN_RC != 0 ? 1 : 0 ))
