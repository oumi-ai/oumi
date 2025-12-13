# Training FAQ

Common questions and solutions for training models with Oumi.

## Memory Issues

### I'm getting Out of Memory (OOM) errors

See the {doc}`oom` guide for comprehensive strategies. Quick fixes:

1. **Reduce batch size**:

    ```yaml
    training:
        per_device_train_batch_size: 4  # Try halving this
        gradient_accumulation_steps: 8  # Increase to compensate
    ```

2. **Enable gradient checkpointing**:

    ```yaml
    training:
        enable_gradient_checkpointing: true
    ```

3. **Use mixed precision**:

    ```yaml
    training:
        mixed_precision_dtype: bf16
    ```

### How much GPU memory do I need?

Rough estimates for full fine-tuning in BF16:

| Model Size | GPU VRAM |
|------------|----------|
| 1B         | ~4 GB    |
| 3B         | ~12 GB   |
| 7B         | ~28 GB   |
| 13B        | ~52 GB   |
| 70B        | ~280 GB  |

Using LoRA/QLoRA reduces memory requirements by 50-80%.

## Training Stability

### My loss is NaN

This usually indicates numerical instability. Try:

```yaml
training:
    learning_rate: 1e-5          # Lower learning rate
    max_grad_norm: 0.5           # Enable gradient clipping
    warmup_ratio: 0.03           # Add warmup
    optimizer: adamw_torch_fused
```

### My loss spikes or doesn't decrease

Common causes and fixes:

1. **Learning rate too high**: Start with `1e-5` and adjust
2. **No warmup**: Add `warmup_ratio: 0.03`
3. **Bad data**: Check for corrupt samples in your dataset
4. **Wrong tokenizer**: Ensure tokenizer matches the model

### Training is very slow

1. **Enable fused optimizer**:

    ```yaml
    training:
        optimizer: adamw_torch_fused
    ```

2. **Use bf16 precision**:

    ```yaml
    training:
        mixed_precision_dtype: bf16
    ```

3. **Increase batch size** (if memory allows):

    ```yaml
    training:
        per_device_train_batch_size: 8
    ```

## LoRA/QLoRA Training

### When should I use LoRA vs full fine-tuning?

| Scenario | Recommendation |
|----------|----------------|
| Limited GPU memory | LoRA/QLoRA |
| Large model (>7B) | LoRA/QLoRA |
| Small dataset | LoRA |
| Unlimited compute | Full fine-tuning |
| Maximum quality | Full fine-tuning |

### What LoRA rank should I use?

Start with `lora_r: 16` and `lora_alpha: 32`. For more complex tasks, try `lora_r: 64`.

```yaml
peft:
    peft_type: LORA
    lora_r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    lora_target_modules:
        - q_proj
        - k_proj
        - v_proj
        - o_proj
```

### How do I merge LoRA weights back into the base model?

Use the Oumi CLI:

```bash
oumi merge-lora \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --adapter-path ./lora-checkpoint \
    --output-path ./merged-model
```

## Dataset Issues

### How do I format my dataset for training?

Oumi expects conversations in the standard format:

```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"}
    ]
}
```

### My dataset isn't loading

Check:

1. **File format**: Ensure it's JSON Lines (`.jsonl`) or supported format
2. **Path**: Use absolute paths in configs
3. **Schema**: Verify your data matches the expected format

```bash
# Validate your JSONL file
head -1 your_data.jsonl | python -m json.tool
```

### How do I use a HuggingFace dataset?

Specify the dataset in your config:

```yaml
data:
    dataset:
        dataset_name: "allenai/tulu-3-sft-mixture"
        split: "train"
```

## Checkpointing and Resumption

### How do I save checkpoints during training?

Configure checkpoint settings:

```yaml
training:
    save_strategy: steps
    save_steps: 500
    save_total_limit: 3        # Keep only last 3 checkpoints
    output_dir: ./checkpoints
```

### How do I resume from a checkpoint?

Set `resume_from_checkpoint`:

```yaml
training:
    resume_from_checkpoint: ./checkpoints/checkpoint-1000
```

Or use the latest checkpoint:

```yaml
training:
    resume_from_checkpoint: true  # Resumes from latest in output_dir
```

### How do I load a pre-trained model?

Specify the model in your config:

```yaml
model:
    model_name: meta-llama/Llama-3.1-8B-Instruct
    # Or for a local checkpoint:
    model_name: ./my-local-model
```

## Multi-GPU Training

### How do I train on multiple GPUs?

Use torchrun or the Oumi launcher:

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 -m oumi train -c your_config.yaml

# Or with accelerate
accelerate launch --num_processes=4 oumi train -c your_config.yaml
```

### How do I enable DeepSpeed?

Add DeepSpeed configuration:

```yaml
training:
    enable_gradient_checkpointing: true
    deepspeed: configs/deepspeed/zero3.json
```

### What's the difference between DDP and FSDP?

| Feature | DDP | FSDP |
|---------|-----|------|
| Memory | Higher per GPU | Lower per GPU |
| Speed | Faster for small models | Better for large models |
| Use case | Models fit in 1 GPU | Models don't fit in 1 GPU |

## Logging and Monitoring

### How do I enable Weights & Biases logging?

```yaml
training:
    enable_wandb: true
    run_name: my-training-run
```

Make sure you have your WandB credentials configured:

```bash
wandb login
```

### Where can I find training logs?

Logs are saved to `output_dir/logs/`. You can also view them in:

- Terminal output
- TensorBoard: `tensorboard --logdir output_dir`
- Weights & Biases (if enabled)

## See Also

- {doc}`oom` - Out of memory solutions
- {doc}`troubleshooting` - General troubleshooting
- {doc}`/user_guides/train/configuration` - Training configuration guide
- {doc}`/user_guides/train/training_methods` - Training methods overview
