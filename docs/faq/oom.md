# Out of Memory (OOM)

## Introduction

Out of Memory (OOM) errors are a common challenge when working with large language models and datasets.

In this guide, we will discuss a few strategies to reduce GPU memory requirements.

```{admonition} Best Practices
:class: tip

- Always monitor memory usage and performance metrics when applying these optimizations, using `nvidia-smi` and Oumi's telemetry output.
- Combine multiple techniques for best results, but introduce changes gradually to isolate their effects.
- Some techniques may trade off speed and model accuracy for memory efficiency. Choose the right balance for your specific use case.
```

## Training Optimizations

1. Reduce batch size:

   ```python
   from oumi.core.configs import TrainingConfig, TrainingParams

   config = TrainingConfig(
       training=TrainingParams(
           per_device_train_batch_size=8,  # Decrease this value
           gradient_accumulation_steps=4,  # Increase this value
       ),
   )
   ```

2. Enable gradient checkpointing:

   ```python
   config = TrainingConfig(
       training=TrainingParams(
           enable_gradient_checkpointing=True,
           gradient_checkpointing_kwargs={"use_reentrant": False},
       ),
   )
   ```

3. Use fused optimizers:

   ```python
   config = TrainingConfig(
       training=TrainingParams(
           optimizer="adamw_torch_fused",
       ),
   )
   ```

4. Use mixed precision training:

   ```python
   config = TrainingConfig(
       training=TrainingParams(
           mixed_precision_dtype="bf16",  # or "fp16"
       ),
   )
   ```

5. Train in half-precision:

   ```python
   config = TrainingConfig(
       model=ModelParams(
           torch_dtype_str="bfloat16",  # or "float16"
       ),
   )
   ```

6. Empty GPU cache more frequently:

   ```python
   config = TrainingConfig(
       training=TrainingParams(
           empty_device_cache_steps=50,  # Clear GPU cache every 50 steps
       ),
   )
   ```

7. Use Paged Adam:

```python
config = TrainingConfig(
    training=TrainingParams(
        optimizer="paged_adamw_32bit",
    ),
)
```

```{note}
Paged Adam requires `bitsandbytes` to be installed.
```

## Model Configuration

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

4. Reduce training sequence length:

   ```python
   config = TrainingConfig(
       model=ModelParams(
           model_max_length=2048,  # Reduce sequence length
       ),
   )
   ```

5. Selectively freeze layers:

   ```python
   config = TrainingConfig(
       model=ModelParams(
           freeze_layers=["layer.0", "layer.1", "layer.2"],
       ),
   )
   ```

6. Enable ring attention:

````{versionadded} 0.2.0 (Coming soon)

```python
config = TrainingConfig(
    model=ModelParams(
        attn_implementation="ring_attention",
    ),
)
```
````

## Parameter-Efficient Fine-Tuning (PEFT)

1. Enable LoRA:

   ```python
   from oumi.core.configs import PeftParams

   config = TrainingConfig(
       training=TrainingParams(use_peft=True),
       peft=PeftParams(
           lora_r=16,
           lora_alpha=32,
           lora_dropout=0.05,
       ),
   )
   ```

## Distributed Training with FSDP

If you have access to multiple GPUs, you can leverage FSDP to distribute the training process across multiple GPUs.

1. Enable distributed training:

   ```python
   from oumi.core.configs import FSDPParams
   from oumi.core.configs.params.fsdp_params import ShardingStrategy

   config = TrainingConfig(
       fsdp=FSDPParams(
           enable_fsdp=True,
           sharding_strategy=ShardingStrategy.FULL_SHARD,
       ),
   )
   ```

2. Enable CPU offloading:

   ```python
   config = TrainingConfig(
       fsdp=FSDPParams(
           enable_fsdp=True,
           cpu_offload=True,
       ),
   )
   ```

3. Disable Forward Prefetch:

   ```python
   config = TrainingConfig(
       fsdp=FSDPParams(
           enable_fsdp=True,
           forward_prefetch=False,
       ),
   )
   ```

4. Disable Backward Prefetch:

   ```python
   config = TrainingConfig(
       fsdp=FSDPParams(
           enable_fsdp=True,
           backward_prefetch=BackwardPrefetch.NONE,
       ),
   )
   ```

```{attention}
Disabling FSDP's forward and backward prefetch can lead to significant slower training times, use with caution.
```
