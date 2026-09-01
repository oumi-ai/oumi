#!/usr/bin/env bash
# Launch the letter-counting verl GRPO run that uses the LLM judge
# (judge_count_letters_verl -> gpt-4o) as the reward.
set -u
cd "$(dirname "$0")/../../../.."
mkdir -p logs

# The judge reward calls the OpenAI API from inside the ray workers, so the
# key must be exported before ray starts. Sourcing .env with set -a exports
# every variable it defines.
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set (add it to .env or export it)." >&2
  exit 1
fi

# Make sure oumi is on PATH even when launched from a bare shell.
if ! command -v oumi >/dev/null 2>&1; then
  export PATH="/root/miniconda3/envs/oumi/bin:$PATH"
fi

# If a ray instance is still alive from an earlier run it was started WITHOUT
# the key in its environment and the judge calls will fail; stop it first.
ray stop --force >/dev/null 2>&1 || true

CUDA_VISIBLE_DEVICES=0,1,2,3 oumi train -c configs/examples/letter_counting/grpo/grpo_train_verl_gemma4_e2b_batch64_pergpu16_judgev2.yaml \
  > logs/letter_counting_grpo_verl_batch64_pergpu16_judgev2_smoke.log 2>&1 &
LETTER_PID=$!

echo "letter_counting (judge v2): PID ${LETTER_PID} -> logs/letter_counting_grpo_verl_batch64_pergpu16_judgev2.log"

wait "${LETTER_PID}"
LETTER_RC=$?
echo "letter_counting (judge v2) exit code: ${LETTER_RC}"
exit "${LETTER_RC}"
