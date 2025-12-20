#!/bin/bash
# Profile SmolLM training for 200 steps
# Usage: ./profile_train.sh

set -e

PROFILE_OUTPUT="/tmp/train_profile_200steps.stats"
LOG_OUTPUT="/tmp/train_profile_200steps.log"

echo "=========================================="
echo "Profiling SmolLM 135M training (200 steps)"
echo "=========================================="
echo "Profile output: $PROFILE_OUTPUT"
echo "Log output: $LOG_OUTPUT"
echo ""

# Run profiled training
python -m cProfile -o "$PROFILE_OUTPUT" -m oumi train \
    -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
    --training.max_steps 200 \
    --training.output_dir output/smollm135m.profile \
    --training.save_steps 1000 \
    --training.logging_steps 50 \
    2>&1 | tee "$LOG_OUTPUT"

echo ""
echo "=========================================="
echo "Training complete. Analyzing profile..."
echo "=========================================="

# Generate profile report
python -c "
import pstats
from pstats import SortKey

print('\n=== TOP 30 BY CUMULATIVE TIME ===\n')
p = pstats.Stats('$PROFILE_OUTPUT')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(30)

print('\n=== TOP 30 BY SELF TIME (tottime) ===\n')
p.strip_dirs().sort_stats(SortKey.TIME).print_stats(30)

print('\n=== DATALOADER SHUTDOWN ANALYSIS ===\n')
p.print_callers('_shutdown_workers')

print('\n=== NUM_TOKENS ANALYSIS ===\n')
p.print_callers('num_tokens')
"

echo ""
echo "Profile saved to: $PROFILE_OUTPUT"
echo "To explore interactively: python -m pstats $PROFILE_OUTPUT"
