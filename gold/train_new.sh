#!/bin/bash
# Ablation Experiment Training Script
# Tests different parameter combinations to identify the fix for model collapse
#
# Expected Results:
# - Exp 0 (baseline): Should collapse around step 300 (control)
# - Exp 1 (beta_only): Should be STABLE (validates beta=0.0 is main fix)
# - Exp 2 (weights_only): Should be UNSTABLE (beta=1.0 still problematic)
# - Exp 3 (full_fix): Should be MOST STABLE (HF demo match)
# - Exp 4 (no_hybrid): Should be STABLE (alternative approach)

set -e  # Exit on error

# Change to repo root
cd /data/shanghong/oumi

# Create log directory
mkdir -p gold/log

# Timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "GOLD Ablation Experiments"
echo "=========================================="
echo ""
echo "This will run 5 experiments (1 full epoch each, ~3-4 hours per experiment)"
echo "Testing different parameter combinations to fix model collapse issue"
echo "Logs will be saved to: gold/log/"
echo ""

# # Experiment 0: BASELINE (Control) - reproduce collapse
# echo "=========================================="
# echo "Experiment 0: BASELINE (Control)"
# echo "Config: beta=1.0, adaptive weights"
# echo "Expected: COLLAPSE around step 300"
# echo "=========================================="
# oumi train --config gold/configs/ablations/tatqa_ablation_0_baseline.yaml 2>&1 | tee gold/log/ablation_0_baseline_${TIMESTAMP}.log

# echo ""
# echo "✓ Experiment 0 complete (log: gold/log/ablation_0_baseline_${TIMESTAMP}.log)"
# echo ""

# Experiment 1: BETA ONLY - test if beta alone fixes it
echo "=========================================="
echo "Experiment 1: BETA ONLY"
echo "Config: beta=0.0, adaptive weights"
echo "Expected: STABLE (validates beta=0.0 is main fix)"
echo "=========================================="
oumi train --config gold/configs/ablations/tatqa_ablation_1_beta_only.yaml 2>&1 | tee gold/log/ablation_1_beta_only_${TIMESTAMP}.log

echo ""
echo "✓ Experiment 1 complete (log: gold/log/ablation_1_beta_only_${TIMESTAMP}.log)"
echo ""

# Experiment 2: WEIGHTS ONLY - test if weights alone help
echo "=========================================="
echo "Experiment 2: WEIGHTS ONLY"
echo "Config: beta=1.0, fixed weights (1.0, 1.0)"
echo "Expected: UNSTABLE (beta=1.0 still problematic)"
echo "=========================================="
oumi train --config gold/configs/ablations/tatqa_ablation_2_weights_only.yaml 2>&1 | tee gold/log/ablation_2_weights_only_${TIMESTAMP}.log

echo ""
echo "✓ Experiment 2 complete (log: gold/log/ablation_2_weights_only_${TIMESTAMP}.log)"
echo ""

# Experiment 3: FULL FIX - HF demo match
echo "=========================================="
echo "Experiment 3: FULL FIX (HF Demo Match)"
echo "Config: beta=0.0, fixed weights (1.0, 1.0)"
echo "Expected: MOST STABLE (HF-proven config)"
echo "=========================================="
oumi train --config gold/configs/ablations/tatqa_ablation_3_full_fix.yaml 2>&1 | tee gold/log/ablation_3_full_fix_${TIMESTAMP}.log

echo ""
echo "✓ Experiment 3 complete (log: gold/log/ablation_3_full_fix_${TIMESTAMP}.log)"
echo ""

# Experiment 4: NO HYBRID - alternative approach
echo "=========================================="
echo "Experiment 4: NO HYBRID ULD"
echo "Config: beta=1.0, no hybrid ULD"
echo "Expected: STABLE (avoids hybrid normalization bug)"
echo "=========================================="
oumi train --config gold/configs/ablations/tatqa_ablation_4_no_hybrid.yaml 2>&1 | tee gold/log/ablation_4_no_hybrid_${TIMESTAMP}.log

echo ""
echo "✓ Experiment 4 complete (log: gold/log/ablation_4_no_hybrid_${TIMESTAMP}.log)"
echo ""

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - output/ablation_0_baseline/"
echo "  - output/ablation_1_beta_only/"
echo "  - output/ablation_2_weights_only/"
echo "  - output/ablation_3_full_fix/"
echo "  - output/ablation_4_no_hybrid/"
echo ""
echo "Logs saved in:"
echo "  - gold/log/ablation_0_baseline_${TIMESTAMP}.log"
echo "  - gold/log/ablation_1_beta_only_${TIMESTAMP}.log"
echo "  - gold/log/ablation_2_weights_only_${TIMESTAMP}.log"
echo "  - gold/log/ablation_3_full_fix_${TIMESTAMP}.log"
echo "  - gold/log/ablation_4_no_hybrid_${TIMESTAMP}.log"
echo ""
echo "Check metrics_rank0000.jsonl in each directory for training metrics"
echo ""
echo "To analyze results, check for:"
echo "  ✓ matched_loss stays positive (> 0)"
echo "  ✓ matched_loss decreases smoothly"
echo "  ✓ unmatched_loss stays < 3.0"
echo "  ✓ grad_norm stays < 20 (no spikes > 25)"
echo ""
echo "Expected outcomes:"
echo "  Exp 0: ❌ Collapse (negative matched_loss)"
echo "  Exp 1: ✅ Stable (beta=0.0 fixes it)"
echo "  Exp 2: ⚠️  Unstable (weights alone insufficient)"
echo "  Exp 3: ✅ Most stable (full fix)"
echo "  Exp 4: ✅ Stable (alternative fix)"
echo ""
