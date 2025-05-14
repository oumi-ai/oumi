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

    ::::{tab-set-code}
    :::{code-block} python
    from oumi.core.configs import TrainingConfig, TrainingParams

    config = TrainingConfig(
        training=TrainingParams(
            per_device_train_batch_size=8,  # Decrease this value
            gradient_accumulation_steps=4,  # Increase this value
        ),
    )
    :::

    :::{code-block} yaml
    training:
        per_device_train_batch_size: 8  # Decrease this value
        gradient_accumulation_steps: 4  # Increase this value
    :::
    ::::

2. Enable gradient checkpointing:

    ::::{tab-set-code}
    :::{code-block} python
    config = TrainingConfig(
        training=TrainingParams(
            enable_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
    )
    :::

    :::{code-block} yaml
    training:
        enable_gradient_checkpointing: true
        gradient_checkpointing_kwargs:
            use_reentrant: false
    :::
    ::::

3. Use fused optimizers:

    ::::{tab-set-code}
    :::{code-block} python
    config = TrainingConfig(
        training=TrainingParams(
            optimizer="adamw_torch_fused",
        ),
    )
    :::

    :::{code-block} yaml
    training:
        optimizer: adamw_torch_fused
    :::
    ::::

4. Use mixed precision training:

    ::::{tab-set-code}
    :::{code-block} python
    config = TrainingConfig(
        training=TrainingParams(
            mixed_precision_dtype="bf16",  # or "fp16"
        ),
    )
    :::

    :::{code-block} yaml
    training:
        mixed_precision_dtype: bf16  # or fp16
    :::
    ::::

5. Train in half-precision:

    ::::{tab-set-code}
    :::{code-block} python
    config = TrainingConfig(
        model=ModelParams(
            torch_dtype_str="bfloat16",  # or "float16"
        ),
    )
    :::

    :::{code-block} yaml
    model:
        torch_dtype_str: bfloat16  # or float16
    :::
    ::::

6. Empty GPU cache more frequently:

    ::::{tab-set-code}
    :::{code-block} python
    config = TrainingConfig(
        training=TrainingParams(
            empty_device_cache_steps=50,  # Clear GPU cache every 50 steps
        ),
    )
    :::

    :::{code-block} yaml
    training:
        empty_device_cache_steps: 50  # Clear GPU cache every 50 steps
    :::
    ::::

7. Tune CUDA Allocator Settings

    It's sometimes possible to eliminate OOM errors (e.g., OOM-s caused by GPU VRAM fragmentation) by tuning CUDA allocator configuration as described in [PyTorch Optimizing Memory Usage](https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf) e.g., by switching to a different allocator, tuning garbage collection settings. Example:

    ::::{tab-set-code}
    :::{code-block} yaml
    envs:
        PYTORCH_CUDA_ALLOC_CONF: "garbage_collection_threshold:0.8,max_split_size_mb:128"
    :::

    :::{code-block} shell
    export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:128"
    :::
    ::::

8. Use Paged Adam:

    ::::{tab-set-code}
    :::{code-block} python
    config = TrainingConfig(
        training=TrainingParams(
            optimizer="paged_adamw_32bit",
        ),
    )
    :::

    :::{code-block} yaml
    training:
        optimizer: paged_adamw_32bit
    :::
    ::::

    ```{note}
    Paged Adam requires `bitsandbytes` to be installed.
    ```

## Model Configuration

1. Use flash attention:

    ::::{tab-set-code}
    :::{code-block} python
    config = TrainingConfig(
        model=ModelParams(
            attn_implementation="sdpa",  # or "flash_attention2"
        ),
    )
    :::

    :::{code-block} yaml
    model:
        attn_implementation: sdpa  # or flash_attention2
    :::
    ::::

2. Enable model compilation:

    ::::{tab-set-code}
    :::{code-block} python
    config = TrainingConfig(
        training=TrainingParams(
            compile=True,
        ),
    )
    :::

    :::{code-block} yaml
    training:
        compile: true
    :::
    ::::

3. Enable Liger Kernels:

    ::::{tab-set-code}
    :::{code-block} python
    from oumi.core.configs import ModelParams

    config = TrainingConfig(
        model=ModelParams(
            enable_liger_kernel=True,
        ),
    )
    :::

    :::{code-block} yaml
    model:
        enable_liger_kernel: true
    :::
    ::::

4. Reduce training sequence length:

    ::::{tab-set-code}
    :::{code-block} python
    config = TrainingConfig(
        model=ModelParams(
            model_max_length=2048,  # Reduce sequence length
        ),
    )
    :::

    :::{code-block} yaml
    model:
        model_max_length: 2048  # Reduce sequence length
    :::
    ::::

5. Selectively freeze layers:

    ::::{tab-set-code}
    :::{code-block} python
    config = TrainingConfig(
        model=ModelParams(
            freeze_layers=["layer.0", "layer.1", "layer.2"],
        ),
    )
    :::

    :::{code-block} yaml
    model:
        freeze_layers:
            - layer.0
            - layer.1
            - layer.2
    :::
    ::::

6. Enable ring attention:

````{versionadded} 0.2.0 (Coming soon)

::::{tab-set-code}
:::{code-block} python
config = TrainingConfig(
    model=ModelParams(
        attn_implementation="ring_attention",
    ),
)
:::

:::{code-block} yaml
model:
  attn_implementation: ring_attention
:::
::::
````

## Parameter-Efficient Fine-Tuning (PEFT)

1. Enable LoRA:

    ::::{tab-set-code}
    :::{code-block} python
    from oumi.core.configs import PeftParams

    config = TrainingConfig(
        training=TrainingParams(use_peft=True),
        peft=PeftParams(
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
        ),
    )
    :::

    :::{code-block} yaml
    training:
        use_peft: true

    peft:
        lora_r: 16
        lora_alpha: 32
        lora_dropout: 0.05
    :::
    ::::

## Distributed Training with FSDP

If you have access to multiple GPUs, you can leverage FSDP to distribute the training process across multiple GPUs. To run FSDP jobs, make sure to invoke your training job with [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) to run on multiple GPUs/nodes. We also provide the `oumi distributed` wrapper to automatically try to set the flags needed for `torchrun`. For example, you can simply run `oumi distributed torchrun -m oumi train -c path/to/train.yaml`.

1. Enable distributed training:

    ::::{tab-set-code}
    :::{code-block} python
    from oumi.core.configs import FSDPParams
    from oumi.core.configs.params.fsdp_params import ShardingStrategy

    config = TrainingConfig(
        fsdp=FSDPParams(
            enable_fsdp=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        ),
    )
    :::

    :::{code-block} yaml
    fsdp:
        enable_fsdp: true
        sharding_strategy: FULL_SHARD
    :::
    ::::

2. Enable CPU offloading:

    ::::{tab-set-code}
    :::{code-block} python
    config = TrainingConfig(
        fsdp=FSDPParams(
            enable_fsdp=True,
            cpu_offload=True,
        ),
    )
    :::

    :::{code-block} yaml
    fsdp:
        enable_fsdp: true
        cpu_offload: true
    :::
    ::::

3. Disable Forward Prefetch:

    ::::{tab-set-code}
    :::{code-block} python
    config = TrainingConfig(
        fsdp=FSDPParams(
            enable_fsdp=True,
            forward_prefetch=False,
        ),
    )
    :::

    :::{code-block} yaml
    fsdp:
        enable_fsdp: true
        forward_prefetch: false
    :::
    ::::

4. Disable Backward Prefetch:

    ::::{tab-set-code}
    :::{code-block} python
    config = TrainingConfig(
        fsdp=FSDPParams(
            enable_fsdp=True,
            backward_prefetch=BackwardPrefetch.NO_PREFETCH,
        ),
    )
    :::

    :::{code-block} yaml
    fsdp:
        enable_fsdp: true
        backward_prefetch: NO_PREFETCH
    :::
    ::::

    ```{attention}
    Disabling FSDP's forward and backward prefetch can lead to significant slower training times, use with caution.
    ```
