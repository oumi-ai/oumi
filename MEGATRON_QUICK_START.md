# Megatron Integration Quick Start Guide

## 🚀 Quick Start

### Basic Training with Evaluation

```yaml
# config.yaml
model:
  model_name: "meta-llama/Llama-3.1-70B-Instruct"
  torch_dtype_str: "bfloat16"

training:
  trainer_type: "MEGATRON_GRPO"
  max_steps: 1000
  eval_steps: 100        # NEW: Evaluate every 100 steps
  save_steps: 200
  learning_rate: 5.0e-7

megatron:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 2
  micro_batch_size: 1
  inference_backend: "vllm"

  # NEW: Enable sequence packing for 30-50% speedup
  enable_sequence_packing: True

  grpo_config:
    max_prompt_length: 512
    max_completion_length: 512
```

### Run Training

```python
from oumi.train import train

# Will automatically use evaluation and sequence packing
train("config.yaml")
```

## 📦 Sequence Packing

### When to Use
- ✅ Variable-length sequences (RL workloads, instruction tuning)
- ✅ Large models where GPU utilization matters
- ✅ Datasets with high length variance

### When NOT to Use
- ❌ Fixed-length sequences (already efficient)
- ❌ Using vLLM inference (not yet supported)
- ❌ Debugging (adds complexity)

### Configuration

```yaml
megatron:
  enable_sequence_packing: True  # Enable packing
  grpo_config:
    max_prompt_length: 512       # Affects bin size
    max_completion_length: 512   # Bin = prompt + completion
```

### Monitoring

Packing efficiency is logged every 100 steps:

```
INFO: Sequence packing efficiency at step 100:
      87.3% utilization, 3.2 sequences/bin, 8 bins
```

## 📊 Evaluation

### Configuration

```yaml
training:
  eval_steps: 100  # Evaluate every N steps (default: None)

# Provide eval_dataset
trainer = OumiMegatronGrpoTrainer(
    ...,
    eval_dataset=eval_dataset,  # Required
)
```

### Metrics

Evaluation returns these metrics:

```python
{
    "eval/reward_mean": 2.34,
    "eval/reward_std": 0.45,
    "eval/reward_min": 1.20,
    "eval/reward_max": 3.50,
    "eval/reward_median": 2.30,
    "eval/completion_length_mean": 128.5,
    "eval/total_samples": 100,
}
```

### Disable Evaluation

```yaml
training:
  eval_steps: null  # Disable evaluation
```

## 🧪 Testing

### Run All Tests

```bash
# Install test dependencies
pip install pytest

# Run all Megatron tests
pytest tests/ -k megatron -v
```

### Run Specific Test Suites

```bash
# Integration tests only
pytest tests/integration/train/test_megatron_grpo_integration.py -v

# vLLM tests (requires vLLM + CUDA)
pytest tests/integration/train/test_megatron_vllm_integration.py -v

# Sequence packing unit tests
pytest tests/unit/core/trainers/test_sequence_packing.py -v
```

### Skip GPU Tests

```bash
# Skip tests that require CUDA
pytest tests/ -k megatron -m "not cuda"
```

## 🐛 Troubleshooting

### Sequence Packing Issues

**Problem**: Low packing efficiency (<50%)

```
Solution: Sequences may be too uniform in length. Check length distribution:

import matplotlib.pyplot as plt
lengths = [len(s) for s in dataset['input_ids']]
plt.hist(lengths, bins=50)
plt.show()
```

**Problem**: vLLM warning when packing enabled

```
WARNING: vLLM backend doesn't support packed sequences efficiently.
         Using Megatron generation for this bin.
```

```
Solution: This is expected. Packed sequences fall back to Megatron
          generation. To use vLLM, disable packing:

megatron:
  enable_sequence_packing: False
```

### Evaluation Issues

**Problem**: Evaluation skipped

```
WARNING: No evaluation dataset provided. Skipping evaluation.
```

```
Solution: Provide eval_dataset to trainer:

trainer = OumiMegatronGrpoTrainer(
    ...,
    eval_dataset=eval_dataset,  # Add this
)
```

**Problem**: Evaluation too slow

```
Solution: Reduce eval_steps to run evaluation less frequently:

training:
  eval_steps: 500  # Instead of 100
```

### Test Failures

**Problem**: `ImportError: No module named 'megatron'`

```
Solution: Install Megatron-Bridge:

pip install megatron-bridge
# or from source:
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge && pip install -e .
```

**Problem**: Multi-GPU tests skipped

```
tests/integration/train/test_megatron_grpo_integration.py::TestMegatronGRPODistributed::test_tensor_parallelism_init SKIPPED (Requires 2+ GPUs)
```

```
Solution: This is expected on single-GPU systems. Tests are automatically
          skipped when hardware is unavailable.
```

## 📈 Performance Tips

### Maximize Packing Efficiency

1. **Sort by length**: Pre-sort dataset by sequence length
2. **Batch size**: Use larger batches to give packing more sequences
3. **Max length**: Set `max_prompt_length + max_completion_length` to actual max, not arbitrary large value

### Optimize Evaluation

1. **Limit eval steps**: Don't evaluate entire dataset if unnecessary
2. **Reduce frequency**: Evaluate every 200-500 steps instead of every 100
3. **Use smaller eval set**: Sample subset of eval data

### Memory Optimization

```yaml
megatron:
  # Reduce memory usage
  optimizer_config:
    optimizer_cpu_offload: True          # Offload to CPU
    optimizer_offload_fraction: 0.5      # Offload 50%

  transformer_config:
    recompute_granularity: "selective"   # Activation checkpointing
    recompute_modules: ["core_attn"]     # Recompute attention only
```

## 🔗 Related Documentation

- Full review: See Megatron integration critical review document
- Implementation details: `MEGATRON_IMPROVEMENTS_SUMMARY.md`
- Original docs: `docs/MEGATRON_INTEGRATION.md`

## 📞 Support

For issues or questions:
1. Check existing tests for usage examples
2. Review `MEGATRON_IMPROVEMENTS_SUMMARY.md` for detailed explanations
3. File issue at https://github.com/oumi-ai/oumi/issues
