# Training Configuration

## Introduction

This guide covers the configuration options available for training in Oumi. The configuration system is designed to be:

- **Modular**: Each aspect of training (model, data, optimization, etc.) is configured separately
- **Type-safe**: All configuration options are validated at runtime
- **Flexible**: Supports various training scenarios from single-GPU to distributed training
- **Extensible**: Easy to add new configuration options and validate them

The configuration system is built on the {py:obj}`~oumi.core.configs.training_config.TrainingConfig` class, which contains all training settings. This class is composed of several parameter classes:

- [Model Configuration](#model-configuration): Model architecture and loading settings
- [Data Configuration](#data-configuration): Dataset and data loading configuration
- [Training Configuration](#training-configuration): Core training parameters
- [PEFT Configuration](#peft-configuration): Parameter-efficient fine-tuning options
- [FSDP Configuration](#fsdp-configuration): Distributed training settings

All configuration files in Oumi are YAML files, which provide a human-readable format for specifying training settings. The configuration system automatically validates these files and converts them to the appropriate Python objects.

## Basic Structure

A typical configuration file has this structure:

```yaml
model:  # Model settings
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
  trust_remote_code: true

data:   # Dataset settings
  train:
    datasets:
      - dataset_name: "your_dataset"
        split: "train"

training:  # Training parameters
  output_dir: "output/my_run"
  num_train_epochs: 3
  learning_rate: 5e-5
  grpo:   # Optional GRPO settings
    num_generations: 2

peft:  # Optional PEFT settings
  peft_method: "lora"
  lora_r: 8

fsdp:  # Optional FSDP settings
  enable_fsdp: false
```

Each section in the configuration file maps to a specific parameter class and contains settings relevant to that aspect of training. The following sections detail each configuration component.

## Configuration Components

### Model Configuration

Configure the model architecture and loading using the {py:obj}`~oumi.core.configs.params.model_params.ModelParams` class:

```yaml
model:
  # Required
  model_name: "meta-llama/Llama-3.1-8B-Instruct"    # Model ID or path (REQUIRED)

  # Model loading
  adapter_model: null                                # Path to adapter model (auto-detected if model_name is adapter)
  tokenizer_name: null                               # Custom tokenizer name/path (defaults to model_name)
  tokenizer_pad_token: null                          # Override pad token
  tokenizer_kwargs: {}                               # Additional tokenizer args
  model_max_length: null                             # Max sequence length (positive int or null)
  load_pretrained_weights: true                      # Load pretrained weights
  trust_remote_code: false                           # Allow remote code execution (use with trusted models only)
  model_revision: null                               # Model revision to use (e.g., "prequantized")

  # Model precision and hardware
  torch_dtype_str: "float32"                         # Model precision (float32/float16/bfloat16/float64)
  device_map: "auto"                                 # Device placement strategy (auto/null)
  compile: false                                     # JIT compile model (use TrainingParams.compile for training)

  # Attention and optimization
  attn_implementation: null                          # Attention impl (null/sdpa/flash_attention_2/eager)
  enable_liger_kernel: false                         # Enable Liger CUDA kernel for potential speedup

  # Model behavior
  chat_template: null                                # Chat formatting template
  freeze_layers: []                                  # Layer names to freeze during training

  # Additional settings
  model_kwargs: {}                                   # Additional model constructor args
```

### Data Configuration

Configure datasets and data loading using the {py:obj}`~oumi.core.configs.params.data_params.DataParams` class. Each split (`train`/`validation`/`test`) is configured using {py:obj}`~oumi.core.configs.params.data_params.DatasetSplitParams`, and individual datasets are configured using {py:obj}`~oumi.core.configs.params.data_params.DatasetParams`:

```yaml
data:
  train:  # Training dataset configuration
    datasets:  # List of datasets for this split
      - dataset_name: "text_sft"            # Required: Dataset format/type
        dataset_path: "/path/to/data"       # Optional: Path for local datasets
        subset: null                        # Optional: Dataset subset name
        split: "train"                      # Dataset split (default: "train")
        sample_count: null                  # Optional: Number of examples to sample
        mixture_proportion: null            # Optional: Proportion in mixture (0-1)
        shuffle: false                      # Whether to shuffle before sampling
        seed: null                          # Random seed for shuffling
        shuffle_buffer_size: 1000           # Size of shuffle buffer
        trust_remote_code: false            # Trust remote code when loading
        transform_num_workers: null         # Workers for dataset processing
        dataset_kwargs: {}                  # Additional dataset constructor args

    # Split-level settings
    collator_name: "text_with_padding"      # Data collator type
    collator_kwargs: {}                     # Additional collator constructor args
    pack: false                             # Pack text into constant-length chunks
    stream: false                           # Enable dataset streaming
    mixture_strategy: "first_exhausted"     # Strategy for mixing datasets
    seed: null                              # Random seed for mixing
    use_torchdata: false                    # Use `torchdata` (experimental)

  validation:  # Optional validation dataset config
    datasets:
      - dataset_name: "text_sft"
        dataset_path: "/path/to/val"
        split: "validation"
```

Notes:

- When using multiple datasets in a split with `mixture_proportion`:
  - All datasets must specify a `mixture_proportion`
  - The sum of all proportions must equal 1.0
  - The `mixture_strategy` determines how datasets are combined:
    - `first_exhausted`: Stops when any dataset is exhausted
    - `all_exhausted`: Continues until all datasets are exhausted (may oversample)
- When `pack` is enabled:
  - `stream` must also be enabled
  - `target_col` must be specified
- All splits must use the same collator type if specified
- If a collator is specified for validation/test, it must also be specified for train
- `collator_kwargs` allows customizing collator behavior with additional parameters:
  - For `text_with_padding`: Can set `max_variable_sized_dims` to control padding dimensions
  - For `vision_language_with_padding`: Can override `allow_multi_image_inputs` or `main_image_feature`
  - For `vision_language_sft`: Can override `allow_multi_image_inputs`, `truncation_side`, etc.
  - Config-provided kwargs take precedence over automatically determined values


### Training Configuration

Configure the training process using the {py:obj}`~oumi.core.configs.params.training_params.TrainingParams` class:

```yaml
training:
  # Basic settings
  output_dir: "output"                    # Directory for saving outputs
  run_name: null                          # Unique identifier for the run
  seed: 42                                # Random seed for reproducibility
  use_deterministic: false                # Use deterministic CuDNN algorithms

  # Training duration
  num_train_epochs: 3                     # Number of training epochs
  max_steps: -1                           # Max training steps (-1 to use epochs)

  # Batch size settings
  per_device_train_batch_size: 8          # Training batch size per device
  per_device_eval_batch_size: 8           # Evaluation batch size per device
  gradient_accumulation_steps: 1          # Steps before weight update

  # Optimization
  learning_rate: 5e-5                     # Initial learning rate
  optimizer: "adamw_torch"                # Optimizer type ("adam", "adamw", "adamw_torch", "adamw_torch_fused", "sgd", "adafactor")
                                          # "adamw_8bit", "paged_adamw_8bit", "paged_adamw", "paged_adamw_32bit" (requires bitsandbytes)
  weight_decay: 0.0                       # Weight decay for regularization
  max_grad_norm: 1.0                      # Max gradient norm for clipping

  # Optimizer specific settings
  adam_beta1: 0.9                         # Adam beta1 parameter
  adam_beta2: 0.999                       # Adam beta2 parameter
  adam_epsilon: 1e-8                      # Adam epsilon parameter
  sgd_momentum: 0.0                       # SGD momentum (if using SGD)

  # Learning rate schedule
  lr_scheduler_type: "linear"             # LR scheduler type
  warmup_ratio: null                      # Warmup ratio of total steps
  warmup_steps: null                      # Number of warmup steps

  # Mixed precision and performance
  mixed_precision_dtype: "none"           # Mixed precision type ("none", "fp16", "bf16")
  compile: false                          # Whether to JIT compile model
  enable_gradient_checkpointing: false    # Trade compute for memory

  # Checkpointing
  save_steps: 500                         # Save every N steps
  save_epoch: false                       # Save at end of each epoch
  save_final_model: true                  # Save model at end of training
  resume_from_checkpoint: null            # Path to resume from
  try_resume_from_last_checkpoint: false  # Try auto-resume from last checkpoint

  # Evaluation
  eval_strategy: "steps"                  # When to evaluate ("no", "steps", "epoch")
  eval_steps: 500                         # Evaluate every N steps
  metrics_function: null                  # Name of metrics function to use

  # Logging
  log_level: "info"                       # Main logger level
  dep_log_level: "warning"                # Dependencies logger level
  enable_wandb: false                     # Enable Weights & Biases logging
  enable_tensorboard: true                # Enable TensorBoard logging
  logging_strategy: "steps"               # When to log ("steps", "epoch", "no")
  logging_steps: 50                       # Log every N steps
  logging_first_step: false               # Log first step metrics

  # DataLoader settings
  dataloader_num_workers: 0               # Number of dataloader workers (int or "auto")
  dataloader_prefetch_factor: null        # Batches to prefetch per worker (requires workers > 0)
  dataloader_main_process_only: null      # Iterate dataloader on main process only (auto if null)

  # Distributed training
  ddp_find_unused_parameters: false       # Find unused parameters in DDP
  nccl_default_timeout_minutes: null      # NCCL timeout in minutes

  # Performance monitoring
  include_performance_metrics: false      # Include token statistics
  include_alternative_mfu_metrics: false  # Include alternative MFU metrics
  log_model_summary: false                # Print model layer summary
  empty_device_cache_steps: null          # Steps between cache clearing

  # Settings if using GRPO. See below for more details.
  grpo:
    num_generations: null
```

### GRPO Configuration

Configure group relative policy optimization using the {py:obj}`~oumi.core.configs.params.grpo_params.GrpoParams` class:

```yaml
training:
  grpo:
    model_init_kwargs: {}                     # Keyword args for AutoModelForCausalLM.from_pretrained
    max_prompt_length: null                   # Max prompt length in input
    max_completion_length: null               # Max completion length during generation
    num_generations: null                     # Generations per prompt
    temperature: 0.9                          # Sampling temperature (higher = more random)
    remove_unused_columns: false              # If true, only keep the "prompt" column
    repetition_penalty: 1.0                   # Penalty for token repetition (>1 discourages repetition)

    # vLLM settings for generation
    use_vllm: false                           # Use vLLM for generation
    vllm_device: null                         # Device for vLLM (e.g., "cuda:1")
    vllm_gpu_memory_utilization: 0.9          # VRAM fraction for vLLM (0-1)
    vllm_dtype: null                          # Data type for vLLM
    vllm_max_model_len: null                  # Maximum model length for vLLM
```

### PEFT Configuration

Configure parameter-efficient fine-tuning using the {py:obj}`~oumi.core.configs.params.peft_params.PeftParams` class:

```yaml
peft:
  # LoRA settings
  lora_r: 8                          # Rank of update matrices
  lora_alpha: 8                      # Scaling factor
  lora_dropout: 0.0                  # Dropout probability
  lora_target_modules: null          # Modules to apply LoRA to
  lora_modules_to_save: null         # Modules to unfreeze and train
  lora_bias: "none"                  # Bias training type
  lora_task_type: "CAUSAL_LM"        # Task type for adaptation
  lora_init_weights: "DEFAULT"       # Initialization of LoRA weights

  # Q-LoRA settings
  q_lora: false                      # Enable quantization
  q_lora_bits: 4                     # Quantization bits
  bnb_4bit_quant_type: "fp4"         # 4-bit quantization type
  use_bnb_nested_quant: false        # Use nested quantization
  bnb_4bit_quant_storage: "uint8"    # Storage type for params
  bnb_4bit_compute_dtype: "float32"  # Compute type for params
  llm_int8_skip_modules: "none"      # A list of modules that we do not want to convert in 8-bit.
```

### FSDP Configuration

Configure fully sharded data parallel training using the {py:obj}`~oumi.core.configs.params.fsdp_params.FSDPParams` class:

```yaml
fsdp:
  enable_fsdp: false                        # Enable FSDP training
  sharding_strategy: "FULL_SHARD"           # How to shard model
  cpu_offload: false                        # Offload to CPU
  mixed_precision: null                     # Mixed precision type
  backward_prefetch: "BACKWARD_PRE"         # When to prefetch params
  forward_prefetch: false                   # Prefetch forward results
  use_orig_params: null                     # Use original module params
  state_dict_type: "FULL_STATE_DICT"        # Checkpoint format

  # Auto wrapping settings
  auto_wrap_policy: "NO_WRAP"               # How to wrap layers
  min_num_params: 100000                    # Min params for wrapping
  transformer_layer_cls: null               # Transformer layer class

  # Other settings
  sync_module_states: true                  # Sync states across processes
```

Notes on FSDP sharding strategies:

- `FULL_SHARD`: Shards model parameters, gradients, and optimizer states. Most memory efficient but may impact performance.
- `SHARD_GRAD_OP`: Shards gradients and optimizer states only. Balances memory and performance.
- `HYBRID_SHARD`: Shards parameters within a node, replicates across nodes.
- `NO_SHARD`: No sharding (use DDP instead).
- `HYBRID_SHARD_ZERO2`: Uses SHARD_GRAD_OP within node, replicates across nodes.

## Example Configurations

You can find these examples and many more in the {doc}`/resources/recipes` section.

We aim to provide a comprehensive (and growing) set of recipes for all the common training scenarios:

### Full Fine-tuning (SFT)

This example shows how to fine-tune a small model ('SmolLM2-135M') without any parameter-efficient methods:

````{dropdown} configs/recipes/smollm/sft/135m/quickstart_train.yaml
```{literalinclude} ../../../configs/recipes/smollm/sft/135m/quickstart_train.yaml
:language: yaml
```
````

### Parameter-Efficient Fine-tuning (LoRA)

This example shows how to fine-tune a large model ('Llama-3.1-70b') using LoRA:

````{dropdown} configs/recipes/llama3_1/sft/70b_lora/train.yaml
```{literalinclude} ../../../configs/recipes/llama3_1/sft/70b_lora/train.yaml
:language: yaml
```
````

### Distributed Training (FSDP)

This example shows how to fine-tune a medium-sized model ('Llama-3.1-8b') using FSDP for distributed training:

````{dropdown} configs/recipes/llama3_1/sft/8b_full/train.yaml
```{literalinclude} ../../../configs/recipes/llama3_1/sft/8b_full/train.yaml
:language: yaml
```
````

### Group Relative Policy Optimization (GRPO)

This example shows how to train a model using the GRPO reinforcement learning algorithm:

````{dropdown} configs/examples/grpo_tldr/train.yaml
```{literalinclude} ../../../configs/examples/grpo_tldr/train.yaml
:language: yaml
```
````

### Vision-Language Fine-tuning

This example shows how to fine-tune a vision-language model ('LLaVA-7B'):

````{dropdown} configs/recipes/vision/llava_7b/sft/train.yaml
```{literalinclude} ../../../configs/recipes/vision/llava_7b/sft/train.yaml
:language: yaml
```
````
