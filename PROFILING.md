# Training Performance Profiling Guide

This guide explains how to profile oumi training to identify performance bottlenecks.

## Quick Start

```bash
# Run both cProfile and PyTorch profiler
./profile_train.sh

# Run only cProfile (function-level)
./profile_train.sh --cprofile

# Run only PyTorch profiler (operation-level)
./profile_train.sh --torch

# Custom steps and config
./profile_train.sh --steps 50 --config configs/recipes/smollm/sft/135m/quickstart_train.yaml
```

## Profiling Methods

### 1. cProfile (Python Function-Level)

**Best for:** Identifying which Python functions consume the most time, finding overhead in framework code (DataLoader, callbacks, metrics).

**How it works:** cProfile is Python's built-in profiler that measures time spent in each function.

#### Manual Usage

```bash
# Profile training
python -m cProfile -o /tmp/profile.stats -m oumi train \
    -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
    --training.max_steps 20

# Analyze results
python -m pstats /tmp/profile.stats
```

#### Interactive Analysis

```python
import pstats
from pstats import SortKey

p = pstats.Stats('/tmp/profile.stats')

# Top functions by cumulative time
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(30)

# Top functions by self time (excluding children)
p.strip_dirs().sort_stats(SortKey.TIME).print_stats(30)

# Who calls a specific function?
p.print_callers('_shutdown_workers')

# What does a function call?
p.print_callees('num_tokens')
```

#### Key Metrics

- **tottime**: Time spent in the function itself (excluding subfunctions)
- **cumtime**: Total time including all subfunctions
- **ncalls**: Number of times the function was called
- **percall**: Average time per call

### 2. PyTorch Profiler (Operation-Level)

**Best for:** Analyzing tensor operations, GPU kernels, memory usage, and visualizing execution timeline.

**How it works:** PyTorch's profiler traces low-level operations (aten ops) and can export Chrome traces for visualization.

#### Using oumi's Built-in Profiler

```bash
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
    --training.max_steps 20 \
    --training.profiler.save_dir /tmp/torch_profile \
    --training.profiler.enable_cpu_profiling true \
    --training.profiler.record_shapes true \
    --training.profiler.with_stack true \
    --training.profiler.with_flops true \
    --training.profiler.with_modules true
```

#### Profiler Options

| Option | Description |
|--------|-------------|
| `enable_cpu_profiling` | Profile CPU operations |
| `enable_cuda_profiling` | Profile CUDA kernels (requires GPU) |
| `record_shapes` | Record tensor shapes for each operation |
| `profile_memory` | Track memory allocations |
| `with_stack` | Include Python call stacks |
| `with_flops` | Calculate FLOP counts |
| `with_modules` | Include PyTorch module names |
| `row_limit` | Number of rows in summary tables |

#### Scheduled Profiling (for longer runs)

Profile only specific steps to reduce overhead:

```bash
oumi train -c your_config.yaml \
    --training.profiler.save_dir /tmp/profile \
    --training.profiler.enable_cpu_profiling true \
    --training.profiler.schedule.enable_schedule true \
    --training.profiler.schedule.skip_first 5 \
    --training.profiler.schedule.wait 2 \
    --training.profiler.schedule.warmup 1 \
    --training.profiler.schedule.active 3 \
    --training.profiler.schedule.repeat 2
```

This profiles steps: 8-10, 15-17 (skip 5, then [wait 2, warmup 1, active 3] × 2)

#### Viewing Chrome Traces

The profiler exports traces that can be visualized:

1. **Perfetto UI** (recommended): https://ui.perfetto.dev/
   - Drag and drop the `.json.gz` file
   - Supports zooming, filtering, and flame graphs

2. **Chrome Tracing**: Open `chrome://tracing` in Chrome
   - Load the trace file
   - Click and drag to zoom

### Output Files

| File | Description |
|------|-------------|
| `prof_*_cpu_time_total.txt` | Operations sorted by total CPU time |
| `prof_*_self_cpu_time_total.txt` | Operations sorted by self CPU time |
| `prof_*_pt_trace.json.gz` | Chrome trace for visualization |

## Common Bottlenecks

### 1. DataLoader Worker Shutdown (10-20s)

**Symptom:** Large time in `poll.poll()` or `_shutdown_workers`

**Cause:** PyTorch DataLoader waits 5s per worker during shutdown

**Fix:**

```yaml
training:
  dataloader_persistent_workers: true  # Keep workers alive between epochs
```

### 2. Token Counting Overhead (10-15s)

**Symptom:** Time in `num_tokens` function

**Cause:** HuggingFace Trainer iterates DataLoader to count tokens

**Fix:**

```yaml
training:
  include_performance_metrics: false
```

### 3. Synchronous `.item()` Calls (10-40s)

**Symptom:** Large time in `aten::_local_scalar_dense` or `tensor.item()`

**Cause:** TRL SFT Trainer computes metrics synchronously every step

**Fix:** Use OUMI trainer or install liger-kernel:

```yaml
training:
  trainer_type: OUMI
  # OR
  trainer_kwargs:
    use_liger_kernel: true
```

### 4. Memory Copies (variable)

**Symptom:** Large time in `aten::copy_` or `aten::to`

**Cause:** Tensors being moved between devices or dtypes

**Fix:** Ensure model and data use consistent dtypes:

```yaml
model:
  torch_dtype_str: bfloat16
```

## Example Analysis Session

```bash
# 1. Run profiling
./profile_train.sh --steps 50

# 2. Check cProfile summary in terminal output

# 3. Open PyTorch trace for visual analysis
#    Go to https://ui.perfetto.dev/
#    Drag /tmp/oumi_torch_profile/prof_*_pt_trace.json.gz

# 4. Interactive cProfile exploration
python -m pstats /tmp/train_profile.stats
# At the prompt:
#   sort cumtime
#   stats 30
#   callers _shutdown_workers
```

## Comparing Configurations

To compare performance between configurations:

```bash
# Baseline
./profile_train.sh --steps 100 --config config_baseline.yaml
mv /tmp/train_profile.stats /tmp/baseline.stats

# Optimized
./profile_train.sh --steps 100 --config config_optimized.yaml
mv /tmp/train_profile.stats /tmp/optimized.stats

# Compare
python -c "
import pstats
print('=== BASELINE ===')
pstats.Stats('/tmp/baseline.stats').strip_dirs().sort_stats('cumtime').print_stats(10)
print('=== OPTIMIZED ===')
pstats.Stats('/tmp/optimized.stats').strip_dirs().sort_stats('cumtime').print_stats(10)
"
```

## GPU Profiling

For GPU training, enable CUDA profiling:

```bash
oumi train -c your_config.yaml \
    --training.profiler.save_dir /tmp/gpu_profile \
    --training.profiler.enable_cpu_profiling true \
    --training.profiler.enable_cuda_profiling true \
    --training.profiler.profile_memory true
```

The trace will show:

- CUDA kernel execution times
- GPU memory allocations
- CPU-GPU synchronization points
- Memory transfer overhead

## References

- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
- [Python cProfile Documentation](https://docs.python.org/3/library/profile.html)
- [Perfetto Trace Viewer](https://ui.perfetto.dev/)
