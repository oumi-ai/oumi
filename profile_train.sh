#!/bin/bash
# Profile SmolLM training with cProfile and/or PyTorch profiler
#
# Usage:
#   ./profile_train.sh              # Run both cProfile and PyTorch profiler
#   ./profile_train.sh --cprofile   # Run only cProfile
#   ./profile_train.sh --torch      # Run only PyTorch profiler
#   ./profile_train.sh --steps 50   # Custom number of steps (default: 20)
#
# Output:
#   cProfile:  /tmp/train_profile.stats (use: python -m pstats /tmp/train_profile.stats)
#   PyTorch:   /tmp/oumi_torch_profile/ (view trace at https://ui.perfetto.dev/)

set -e

# Default settings
MAX_STEPS=20
RUN_CPROFILE=false
RUN_TORCH=false
CONFIG="configs/recipes/smollm/sft/135m/quickstart_train.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cprofile)
            RUN_CPROFILE=true
            shift
            ;;
        --torch)
            RUN_TORCH=true
            shift
            ;;
        --steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--cprofile] [--torch] [--steps N] [--config PATH]"
            echo ""
            echo "Options:"
            echo "  --cprofile    Run Python cProfile (function-level timing)"
            echo "  --torch       Run PyTorch profiler (operation-level timing)"
            echo "  --steps N     Number of training steps (default: 20)"
            echo "  --config PATH Training config file (default: quickstart_train.yaml)"
            echo ""
            echo "If neither --cprofile nor --torch is specified, both are run."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If neither specified, run both
if [[ "$RUN_CPROFILE" == "false" && "$RUN_TORCH" == "false" ]]; then
    RUN_CPROFILE=true
    RUN_TORCH=true
fi

echo "=========================================="
echo "Training Profiler"
echo "=========================================="
echo "Config: $CONFIG"
echo "Steps: $MAX_STEPS"
echo "cProfile: $RUN_CPROFILE"
echo "PyTorch Profiler: $RUN_TORCH"
echo ""

# =============================================================================
# cProfile - Python function-level profiling
# =============================================================================
if [[ "$RUN_CPROFILE" == "true" ]]; then
    CPROFILE_OUTPUT="/tmp/train_profile.stats"
    CPROFILE_LOG="/tmp/train_profile.log"

    echo "=========================================="
    echo "Running cProfile..."
    echo "=========================================="
    echo "Output: $CPROFILE_OUTPUT"
    echo ""

    python -m cProfile -o "$CPROFILE_OUTPUT" -m oumi train \
        -c "$CONFIG" \
        --training.max_steps "$MAX_STEPS" \
        --training.output_dir output/profile_cprofile \
        --training.save_steps 10000 \
        --training.logging_steps "$MAX_STEPS" \
        --training.include_performance_metrics false \
        2>&1 | tee "$CPROFILE_LOG"

    echo ""
    echo "=========================================="
    echo "cProfile Analysis"
    echo "=========================================="

    python -c "
import pstats
from pstats import SortKey

print('\n=== TOP 25 BY CUMULATIVE TIME ===\n')
p = pstats.Stats('$CPROFILE_OUTPUT')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(25)

print('\n=== TOP 25 BY SELF TIME ===\n')
p.strip_dirs().sort_stats(SortKey.TIME).print_stats(25)

print('\n=== DATALOADER SHUTDOWN ===\n')
p.print_callers('_shutdown_workers')

print('\n=== TENSOR.ITEM() CALLERS ===\n')
p.print_callers(\"'item' of 'torch\")
"

    echo ""
    echo "cProfile stats saved to: $CPROFILE_OUTPUT"
    echo "To explore interactively: python -m pstats $CPROFILE_OUTPUT"
    echo ""
fi

# =============================================================================
# PyTorch Profiler - Operation-level profiling
# =============================================================================
if [[ "$RUN_TORCH" == "true" ]]; then
    TORCH_PROFILE_DIR="/tmp/oumi_torch_profile"

    echo "=========================================="
    echo "Running PyTorch Profiler..."
    echo "=========================================="
    echo "Output: $TORCH_PROFILE_DIR"
    echo ""

    # Clean previous output
    rm -rf "$TORCH_PROFILE_DIR"

    oumi train \
        -c "$CONFIG" \
        --training.max_steps "$MAX_STEPS" \
        --training.output_dir output/profile_torch \
        --training.save_steps 10000 \
        --training.logging_steps "$MAX_STEPS" \
        --training.include_performance_metrics false \
        --training.dataloader_num_workers 0 \
        --training.profiler.save_dir "$TORCH_PROFILE_DIR" \
        --training.profiler.enable_cpu_profiling true \
        --training.profiler.record_shapes true \
        --training.profiler.profile_memory false \
        --training.profiler.with_stack true \
        --training.profiler.with_flops true \
        --training.profiler.with_modules true \
        --training.profiler.row_limit 30

    echo ""
    echo "=========================================="
    echo "PyTorch Profiler Output"
    echo "=========================================="
    ls -la "$TORCH_PROFILE_DIR"
    echo ""
    echo "View trace at: https://ui.perfetto.dev/"
    echo "  1. Open the URL above"
    echo "  2. Drag and drop: $TORCH_PROFILE_DIR/prof_*_pt_trace.json.gz"
    echo ""
fi

echo "=========================================="
echo "Profiling Complete"
echo "=========================================="
