# VERL PPO Integration for Oumi

This directory contains configuration examples for using the VERL (Volcano Engine Reinforcement Learning) integration with Oumi. VERL is a flexible, efficient, and production-ready Reinforcement Learning framework for Large Language Models.

## Available Configurations

- `train.yaml`: Standard PPO configuration using a small language model
- `train_grpo.yaml`: GRPO (Group Relative Policy Optimization) configuration 

## Key Features

VERL provides:

- **Distributed Training**: Efficient model parallelism and data parallelism with FSDP or Megatron-LM
- **Flexible Architecture**: Easily switch between various RL algorithms (PPO, GRPO, ReMax, etc.)
- **Fast Generation**: Integrated with vLLM for maximum inference throughput
- **Resource Optimization**: Flexible device mapping for efficient GPU utilization

## Usage

To train using one of these configurations:

```bash
# For standard PPO training
oumi train -c configs/examples/verl_ppo/train.yaml

# For GRPO training
oumi train -c configs/examples/verl_ppo/train_grpo.yaml
```

## Configuration Parameters

The VERL PPO integration exposes the following key parameters:

- `adv_estimator`: Algorithm for advantage estimation ('GAE', 'GRPO', etc.)
- `training_strategy`: 'FSDP' for PyTorch FSDP or 'MEGATRON' for Megatron-LM
- `rollout_engine`: Engine for generation ('VLLM', 'SGLANG', 'TRANSFORMERS')
- `n_gpus_per_node` and `nnodes`: GPU resource allocation
- `use_reward_model`: Whether to use a model-based reward or function-based reward

The trainer automatically configures additional required parameters based on your settings:

- When using `adv_estimator: "GRPO"`, it sets `rollout.n: 4` (for multiple completions per prompt)
- When using `adv_estimator: "GAE"`, it sets `rollout.n: 1` (standard PPO)

Batch sizes are calculated automatically based on the number of GPUs:
- `ppo_micro_batch_size_per_gpu`: Default is 4 samples per GPU
- `ppo_micro_batch_size`: Calculated as `per_gpu_batch_size * n_gpus_per_node * nnodes`
- `ppo_mini_batch_size`: Calculated to be at least twice the total micro batch size

All other VERL-specific parameters are handled automatically by the VerlPpoTrainer class to ensure compatibility between Oumi and VERL.

## Troubleshooting

If you encounter errors during initialization:

1. **Dataset errors including "No objects to concatenate" or "Parquet magic bytes not found"**: 
   - VERL requires datasets in **Parquet format** specifically
   - The current implementation creates temporary Parquet files with dummy data to satisfy this requirement
   - If you still see errors, your dataset might be misconfigured or the dummy dataset approach is incompatible with your VERL version
   - Solutions to try:
     - Ensure pandas and pyarrow are installed: `pip install pandas pyarrow`
     - Try providing a valid Parquet dataset of your own
     - Increase the dummy dataset size in `_create_dummy_dataset_files` to include more examples:
       ```python
       # In verl_ppo_trainer.py
       dummy_train_file, dummy_val_file = self._create_dummy_dataset_files(num_examples=100)  # Try more examples
       ```
     - If you need to convert your own dataset to Parquet format, use pandas:
       ```python
       import pandas as pd
       df = pd.DataFrame({"prompt": prompts, "completion": completions})
       df.to_parquet("dataset.parquet", index=False)
       ```

2. **Missing parameter errors** (like `Missing key ppo_micro_batch_size`): The trainer is designed to automatically add required parameters based on the VERL configuration format. VERL requires `ppo_micro_batch_size` to be defined at multiple levels in the configuration hierarchy:
   - Top level
   - `actor_rollout_ref` level
   - `actor` level 
   - `rollout` level
   - `ref` level
   - `critic` level
   - `algorithm` level
   - `reward_model` level
   - `trainer` level
   
   If you encounter "Missing key" errors, please report them so we can add them to the default configuration.

3. **GPU resource errors**: Check Ray initialization and GPU availability using `nvidia-smi`.

4. **VERL validation errors**: These are handled automatically by the trainer, but if you encounter persistent validation errors, you might need to check:
   - Your VERL version compatibility
   - Dataset format requirements
   - Batch size relationships (micro batch sizes must be consistent and mini batch size must be larger than micro batch size)
   - Memory requirements for your model size and batch sizes

## Development Status

This integration is currently experimental. We're working on:

1. Supporting proper dataset conversion for actual training
2. Adding more configuration options for advanced use cases
3. Implementing full integration with Oumi's monitoring and logging systems

For more information and examples, see the VERL documentation: [HybridFlow/VERL documentation](https://verl.readthedocs.io/)