# Performance Optimization

Oumi provides various performance optimization techniques to help you train large foundation models efficiently.

This guide covers profiling, memory optimization, and other techniques to improve training speed and reduce resource usage.

## Profiling

Before we can optimize our training pipeline, we need to understand its performance.

### Enabling Performance Monitoring

You can enable or disable performance monitoring using the {py:obj}`oumi.core.configs.TrainingParams.include_performance_metrics` parameter:

```python
config = TrainingConfig(
    training=TrainingParams(
        include_performance_metrics=True,
    ),
)
```

### PyTorch Profiling

You can enable profiling using the {py:obj}`oumi.core.configs.TrainingParams.profiler` parameter.

```{note}
Refer to the [PyTorch Profiler documentation](https://pytorch.org/docs/stable/profiler.html) for more details on the available parameters.
```

```python
from oumi.core.configs import TrainingConfig, TrainingParams, ProfilerParams

config = TrainingConfig(
    training=TrainingParams(
        profiler=ProfilerParams(
            enable_cpu_profiling=True,
            enable_cuda_profiling=True,
            schedule=ProfilerScheduleParams(
                wait=1,
                warmup=1,
                active=3,
                repeat=2,
            ),
        ),
    ),
)
```

### Training Telemetry

Oumi provides telemetry results through the {py:obj}`oumi.performance.telemetry.TelemetryTracker` class.

Telemetry results are saved to the `telemetry` directory in the output directory.

You can analyze these results to identify performance bottlenecks and optimize your training pipeline.

## Faster Training

### Model Configuration

1. Use flash attention:

   ```python
   config = TrainingConfig(
       model=ModelParams(
           attn_implementation="sdpa",  # or "flash_attention2"
       ),
   )
   ```

2. Enable model compilation:

   ```python
   config = TrainingConfig(
       training=TrainingParams(
           compile=True,
       ),
   )
   ```

3. Enable Liger Kernels:

    ```python
    from oumi.core.configs import ModelParams

    config = TrainingConfig(
        model=ModelParams(
            enable_liger_kernel=True,
        ),
    )
    ```

### Data Loading

1. Increase Number of Worker Processes

    Optimize data loading by configuring the number of worker processes:

    ```python
    config = TrainingConfig(
        training=TrainingParams(
                dataloader_num_workers="auto",  # or specify a number
                dataloader_prefetch_factor=2,
            ),
        )
    ```

2. Use data streaming:

    ```python
    from oumi.core.configs import DataParams

    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                stream=True,
                pack=True,
            ),
        ),
    )
    ```

### Training Optimizations

1. Maximize batch size:

   ```python
   from oumi.core.configs import TrainingConfig, TrainingParams

   config = TrainingConfig(
       training=TrainingParams(
           per_device_train_batch_size=8,  # Increase this value
           gradient_accumulation_steps=4,  # Increase this value
       ),
   )
   ```

2. Use fused optimizers:

Prefer using fused kernels when available to improve performance.

```python
config = TrainingConfig(
    training=TrainingParams(
        optimizer="adamw_torch_fused",
    ),
)
```

### FSDP Optimizations

1. Use Hybrid Shard

For multi-node training, you can use hybrid sharding to optimize communication, assuming your model and training state fits under a single nodes's memory:

```python
from oumi.core.configs import FSDPParams
from oumi.core.configs.params.fsdp_params import ShardingStrategy

config = TrainingConfig(
    fsdp=FSDPParams(
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    ),
    # ... other config options ...
)
```

```{seealso}
For more information on PyTorch performance optimization, refer to the [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html).
```
