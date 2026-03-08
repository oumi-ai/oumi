# JAX Inference Examples

Example configurations for JAX-based high-performance inference.

**⚠️ Experimental**: This is an experimental feature under active development. The JAX models are sourced from [jax-llm-examples](https://github.com/jax-ml/jax-llm-examples) and are being integrated into Oumi's inference pipeline.

## Requirements

```bash
# Install JAX dependencies
pip install oumi[jax]

# For GPU support (recommended for large models)
pip install oumi[jax,gpu]

# For TPU support (includes TPU utilities)
pip install oumi[jax]
# Then install JAX TPU backend:
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Quick Start

### Basic Inference
```bash
# Llama3 with JAX backend
oumi infer -i -c configs/examples/jax_inference/llama3_basic.yaml

# DeepSeek R1 with MLA attention
oumi infer -i -c configs/examples/jax_inference/deepseek_r1_basic.yaml
```

### Quantized Inference
```bash
# INT8 quantization for memory efficiency
oumi infer -i -c configs/examples/jax_inference/llama3_int8.yaml
```

### Multi-GPU Deployment
```bash
# Tensor parallelism across GPUs
oumi infer -i -c configs/examples/jax_inference/llama3_multigpu.yaml
```

## Examples Overview

| Model | Config | Description | Hardware |
|-------|--------|-------------|----------|
| Llama3 8B | `llama3_basic.yaml` | Basic JAX inference | 1x GPU |
| Llama3 8B | `llama3_int8.yaml` | INT8 quantization | 1x GPU |
| Llama3 8B | `llama3_multigpu.yaml` | Multi-GPU parallel | 2-4x GPU |
| DeepSeek R1 | `deepseek_r1_basic.yaml` | MLA attention demo | 1x GPU |
| GPT OSS 120B | `gpt_oss_distributed.yaml` | Large model inference | 8x GPU |

## Performance Benefits

JAX provides significant performance improvements:

- **XLA Compilation**: Optimized compute kernels
- **Memory Efficiency**: Advanced memory management
- **Parallelism**: Tensor and pipeline parallelism
- **Hardware Agnostic**: CPU, GPU, and TPU support

## Learn More

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX LLM Examples](https://github.com/jax-ml/jax-llm-examples)
- [JAX Scaling Guide](https://jax-ml.github.io/scaling-book/)
- [Oumi Inference Docs](https://oumi.ai/docs/en/latest/user_guides/infer/infer.html)
