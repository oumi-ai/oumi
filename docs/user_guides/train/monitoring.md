# Monitoring & Debugging

## Introduction

Effective monitoring and debugging tools are crucial for successful model training. In this guide, we'll cover key signals to monitor during training, how to interpret them, and how to address common issues.

In particular, we'll cover:
- Tracking training metrics and progress in real-time
- Monitoring system resource utilization (GPU, Memory, CPU) and training efficiency (MFU, Throughput)
- Debugging training issues and performance bottlenecks

## Training Monitoring

### Training Metrics

During training, Oumi provides detailed progress information in the terminal:

```text
Epoch 1/3: 100%|██████████| 1000/1000 [00:45<00:00, 22.22it/s]
{
    'loss': 2.1234,
    'learning_rate': 5e-5,
    'epoch': 1.0,
    'tokens_per_second': 156.7,  # Tokens/second
}
```

If an `output_dir` is specified, the trainer will also save logs and artifacts to the specified directory.

This includes training metrics, the full configuration, environment and package versions, to allow for reproducibility and debugging.

Let's see how to configure logging in more detail:

```yaml
training:
  # Logging frequency
  logging_steps: 50
  logging_strategy: "steps"  # "steps", "epoch", or "no"
  logging_first_step: false  # the first step is not logged by default as it's often an initialization step

  # Log levels
  log_level: "info"
  dep_log_level: "warning"  # the default log level for dependencies (e.g. transformers, tokenizers, etc.)

  # Performance metrics
  include_performance_metrics: true  # enable performance metrics (throughput, mfu, etc.)
  log_model_summary: true  # log model architecture and parameters summary

  # Logging output
  output_dir: "output/my_run"  # the directory to save logs and artifacts
  run_name: "my_run"  # the name of the run

  # Third-party integrations
  enable_wandb: true  # enable Weights & Biases integration
  enable_tensorboard: true  # enable TensorBoard integration
```

**Note:**
- If `training.enable_wandb` is set to `true`, make sure you have a W&B account and API key configured.
- If `training.enable_tensorboard` is set to `true`, you can launch TensorBoard locally by running `tensorboard --logdir <output_dir> --port 6006`, or use the `vscode` tensorboard extension.

### Remote Job Logs

When running locally, accessing all this information is straightforward, as everything is saved to the specified `output_dir`.

However, when deploying training jobs on remote clusters or cloud instances, this can be a bit more challenging.

Here are some ways to access logs and artifacts from remote jobs from your local machine:

**View Live Logs**

To view live logs of your remote jobs:

```bash
# For SkyPilot managed clusters
sky logs oumi-cluster

# For specific job ID
sky logs oumi-cluster --job-ID JOB_ID
```

**Download Logs**

To download logs and artifacts from remote jobs:

```bash
# Download all logs from a cluster
sky logs --sync-down oumi-cluster

# Download specific files
rsync -Pavz oumi-cluster:/path/to/file .
```

**Log Locations**

Common log locations:
- Training logs: `output/<run_name>/logs/`
- TensorBoard logs: `output/<run_name>/tensorboard/`
- Profiler traces: `output/<run_name>/profiler/`
- Telemetry data: `output/<run_name>/telemetry/`


### Torch Profiler Integration

Oumi integrates PyTorch's profiler for detailed performance analysis.

The [PyTorch profiler](https://pytorch.org/docs/stable/profiler.html) provides comprehensive performance insights including CPU and CUDA operation timings, memory usage analysis, tensor shapes and sizes tracking, FLOPS calculations, module-level profiling, and Chrome trace export capabilities for visualization.

See the {py:class}`~oumi.core.configs.params.profiler_params.ProfilerParams` class for more details.

Here's an example configuration:

```yaml
training:
  profiler:
    # Enable profiling
    enable_cpu_profiling: true
    enable_cuda_profiling: true
    profile_memory: false

    # Configure profiling schedule
    schedule:
      enable_schedule: true
      skip_first: 5
      wait: 1
      warmup: 1
      active: 3
      repeat: 2

    # Output configuration, enable as needed
    save_dir: "profiler_output"
    record_shapes: false
    with_stack: false
    with_flops: false
    with_modules: false
```

### Distributed Training

The telemetry logger provides comprehensive system monitoring by tracking CPU timing metrics, CUDA operation timing, GPU memory usage, and GPU temperature.

It also collects cross-rank statistics for distributed training scenarios and monitors overall system resource utilization to give a complete picture of training performance.

In distributed training, each rank (GPU) collects its own telemetry data. The telemetry tracker provides methods to:
1. Collect telemetry from all ranks using `get_summaries_from_all_ranks()`
2. Compute cross-rank statistics using `compute_cross_rank_summaries()`
3. Analyze performance variations across GPUs

Here's an example configuration:

```yaml
training:
  telemetry:
    # Collection settings
    collect_telemetry_for_all_ranks: true  # Collect from all ranks
    track_gpu_temperature: true  # Monitor GPU temperatures

  # Enable performance metrics for telemetry
  include_performance_metrics: true
```

The cross-rank statistics provide valuable insights into distributed training performance by helping identify several key metrics across GPUs.

These include load imbalances between different GPUs, any stragglers that may be slowing down distributed execution, variations in memory usage patterns, temperature differences between GPU devices, and timing variations in critical operations.

For example, in a 4-GPU training setup, you might see output like:

```text
Cross-Rank Statistics:
GPU Temperature:
    max:
        mean: 75.2C   # Average of max temperatures across ranks
        std_dev: 2.1C # Variation in max temperatures
        min: 72.0C    # Lowest max temperature among ranks
        max: 78.0C    # Highest max temperature among ranks
    mean:
        mean: 71.5C   # Average of mean temperatures
        std_dev: 1.8C # Variation in mean temperatures
        min: 69.0C    # Lowest mean temperature
        max: 74.0C    # Highest mean temperature

Timers:
    model_forward:
        mean:
            mean: 0.096s    # Average forward pass time across ranks
            std_dev: 0.008s # Variation in forward pass times
            min: 0.085s     # Fastest rank
            max: 0.105s     # Slowest rank
        max:
            mean: 0.125s    # Average of max forward times
            std_dev: 0.010s # Variation in max times
            min: 0.115s     # Lowest max time
            max: 0.142s     # Highest max time
    preprocessing:
        mean:
            mean: 0.016s    # Average preprocessing time
            std_dev: 0.003s # Variation in preprocessing
            min: 0.012s     # Fastest rank
            max: 0.020s     # Slowest rank
```


### Adding Telemetry to Your Code

When writing your own training code or pre-processing pipelines, you can add telemetry by using the {py:class}`~oumi.performance.telemetry.TelemetryTracker` class.

Here's an example:

```python
from oumi.performance.telemetry import TelemetryTracker

# Initialize telemetry
telemetry = TelemetryTracker()

for step, batch in enumerate(dataloader):
    # Track CUDA operations (e.g. forward pass)
    with telemetry.cuda_timer("forward_pass"):
        outputs = model(batch)

    # Track CPU operations (e.g. loss computation)
    with telemetry.timer("backward_pass"):
        loss = criterion(outputs, batch["labels"])
        loss.backward()
        optimizer.step()

    # Monitor GPU resources periodically
    if step % 50 == 0:
        telemetry.log_gpu_memory()
        telemetry.record_gpu_temperature()

    if step % 100 == 0:
        # Print summary at step
        telemetry.print_summary()
```

Example summary output:

```text
Telemetry Summary (my-machine):
Total time: 3600.0 seconds

CPU Timers:
    total_training:
        Total: 3600.0s Mean: 3600.0s
        Count: 1 Percentage of total time: 100.00%
    epoch:
        Total: 3590.0s Mean: 897.5s
        Count: 4 Percentage of total time: 99.72%
    backward_pass:
        Total: 720.0s Mean: 0.18s
        Count: 4000 Percentage of total time: 20.00%

CUDA Timers:
    forward_pass:
        Total: 2160.0s Mean: 0.54s
        Count: 4000 Percentage of total time: 60.00%

Peak GPU memory usage: 14.5 GB
GPU temperature: Mean: 72°C Max: 76°C
```

The telemetry tracker provides:
- CPU timers via `timer()` context manager
- CUDA timers via `cuda_timer()` context manager
- GPU memory tracking via `log_gpu_memory()`
- GPU temperature monitoring via `record_gpu_temperature()`
- Automatic statistics calculation and reporting with `print_summary()`

For distributed training scenarios, see the [Distributed Training Telemetry](#distributed-training) section above.

### Callbacks

Oumi provides several callbacks for monitoring training progress and performance:

```yaml
training:
  # Enable Performance Metrics, MFU monitoring, NaN/Inf detection, and telemetry callbacks
  include_performance_metrics: true
```

#### Model Flops Utilization (MFU)

MFU measures how efficiently your model utilizes the available compute resources:

```yaml
training:
  # Enable MFU monitoring
  include_performance_metrics: true

  # For accurate MFU calculation
  model:
    model_max_length: 2048  # Must be set for MFU calculation
```

MFU monitoring provides:
- Step-level MFU: Efficiency during actual training steps
- Overall training MFU: Efficiency including all overhead


#### Telemetry Callback

While the telemetry tracker can be used directly in your code, Oumi also provides a {py:class}`~oumi.core.callbacks.telemetry_callback.TelemetryCallback` that integrates with the training loop to automatically collect timing metrics, GPU stats, and other telemetry data.

The callback will automatically:
1. Time each phase of training (microsteps, steps, epochs)
2. Log GPU temperature if enabled
3. Save detailed telemetry data to files:
   - `telemetry_callback_metrics_rankXXXX.json`: Training metrics
   - `telemetry_callback_rankXXXX.json`: Timing statistics
   - `telemetry_callback_all_ranks.json`: Cross-rank data (distributed training)
   - `telemetry_callback_gpu_temperature_summary.json`: GPU temperature stats

For distributed training, the callback can collect data from all ranks and compute cross-rank statistics. This is controlled by the `world_process_zero_only` parameter:
- If `True`: Only collect data on the main process (rank 0)
- If `False`: Collect data from all ranks and compute cross-rank statistics

Example output files in distributed training:
```text
output/my_run/
├── telemetry_callback_metrics_rank0000.json  # Training metrics from rank 0
├── telemetry_callback_rank0000.json         # Timing stats from rank 0
├── telemetry_callback_all_ranks.json        # Data from all ranks
└── telemetry_callback_gpu_temperature_summary.json  # Cross-rank GPU stats
```

The callback integrates with both W&B and TensorBoard logging, adding telemetry metrics to your training logs when `include_timer_metrics=True`:
```text
rank000_microsteps_mean: 0.118s
rank000_steps_mean: 0.453s
rank000_gpu_temperature_max: 75.2C
...
```

## Next Steps

- Explore {doc}`training methods <training_methods>` for specific tasks
- Check out {doc}`configuration options </user_guides/train/configuration>` for detailed settings
