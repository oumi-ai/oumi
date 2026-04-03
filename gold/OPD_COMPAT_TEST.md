# OPD Compatibility Test Summary

**Date:** 2026-04-01
**Branch:** oumi main (0.6.1.dev189)
**Hardware:** 8× H100 80GB
**Config:** `gold/configs/tatqa_llama_lambda0.5_1epoch_lora_bf16.yaml`

## Objective

Verify that OPD (TRL_GOLD on-policy distillation) works with the latest open-source versions of
vllm, trl, and transformers. Config: Llama-3.1-8B student + Llama-3.3-70B teacher, TatQA dataset,
LoRA, colocate vLLM mode.

---

## Environment Setup Issues

### scipy build failures during `pip install oumi[gpu]`
- **Cause:** `mlflow → skops` pulled scipy 1.9.1, which has no Python 3.11 binary wheel and requires
  a C compiler not present in a fresh conda env.
- **Fix:** Pre-install scipy via conda-forge; pin `scipy==1.17.1` as a pip constraint.

### `CXXABI_1.3.15` not found
- **Cause:** Conda-installed packages use a newer C++ ABI than the system `libstdc++.so.6`.
- **Fix:** `conda install -c conda-forge libstdcxx-ng` and set `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`
  at launch.

### vllm installed to `~/.local` instead of conda env
- **Cause:** Earlier pip invocations ran without `--no-user`, polluting user site-packages with
  incomplete vllm dependencies (missing `zmq`, `msgspec`, etc.).
- **Fix:** Reinstall vllm explicitly into the conda env.

---

## Dependency Conflict: transformers v5 + vllm 0.18.1

**Attempted:** transformers 5.4.0, vllm 0.18.1, trl 0.29.1, verl 0.7.1

**Finding:** vllm 0.18.1 hard-requires `transformers<5,>=4.56.0`. Every vllm version in oumi's
supported range (0.10–0.18) carries this same constraint. vllm 0.19 does not exist on PyPI yet.
**transformers 5.x and vllm 0.18.1 cannot coexist in the same environment.**

**Resolution:** Proceeded with **transformers 4.57.6** (latest 4.x compatible with vllm 0.18.1).

---

## Final Test Environment

| Package      | Before                  | After                            |
|--------------|-------------------------|----------------------------------|
| oumi         | 0.6.0 (site-packages)   | **0.6.1.dev189** (main, editable)|
| vllm         | 0.10.2                  | **0.18.1**                       |
| trl          | 0.26.2                  | **0.29.1**                       |
| transformers | 4.57.3                  | **4.57.6**                       |
| torch        | 2.8.0                   | **2.10.0**                       |
| verl         | 0.5.x                   | **0.7.1**                        |

---

## Training Run Issues

### Issue 1 — `torch_dtype_str: bfloat16` + mixed precision validation

- **Error:** `ValueError: Model must be loaded in fp32 to enable mixed precision training`
- **Cause:** oumi validates that `torch_dtype_str` must be `float32` when `mixed_precision_dtype`
  is `BF16` or `FP16` (`src/oumi/core/configs/training_config.py:138`).
- **Fix:** Set `mixed_precision_dtype: "NONE"` — loads the model directly in bfloat16 with no AMP
  casting.

### Issue 2 — wandb not configured

- **Error:** `wandb.errors.UsageError: No API key configured`
- **Cause:** `WANDB_API_KEY` not present in `gold/.env`; new conda env had no wandb credentials.
- **Fix:** Pass `WANDB_API_KEY` via environment variable at launch.

### Issue 3 — vllm OOM with `vllm_gpu_memory_utilization: 0.25`

- **Error:** `ValueError: Free memory on cuda:0 (12.48/79.19 GiB) is less than desired GPU memory
  utilization (19.8 GiB)`
- **Cause:** After the 70B teacher loads across 8 GPUs (~17.5 GB/GPU via `device_map: auto`), GPU 0
  has only 12.48 GB free. vllm 0.18.1's startup check requires 25% of 80 GB = 19.8 GB free on the
  target GPU before allocating KV cache.

### Issue 4 — vllm OOM with `vllm_gpu_memory_utilization: 0.10`

- **Error:** `ValueError: No available memory for the cache blocks`
- **Cause:** With `vllm_tensor_parallel_size: 1`, vllm coordinates inference on GPU 0 only. But
  the 70B teacher's weights are split across all 8 GPUs — GPU 0 holds only 1/8th of the model,
  which is insufficient for single-GPU inference. vllm's profiler finds zero viable cache blocks.

---

## Root Cause of Training Failure

**vllm 0.18.1 broke colocate mode for multi-GPU setups.**

The previous successful run used vllm 0.10.2 with `vllm_gpu_memory_utilization: 0.4` on the same
hardware and completed successfully. In vllm 0.18.1, the memory profiling startup check is stricter
and does not correctly handle the case where the teacher model is already distributed across GPUs
via HuggingFace's `device_map: auto` while vllm operates with `tensor_parallel_size: 1`.

### Previous successful run (for reference)
| Parameter | Value |
|-----------|-------|
| vllm | 0.10.2 |
| vllm_gpu_memory_utilization | 0.4 |
| vllm_tensor_parallel_size | 1 |
| Result | ✅ Completed full epoch |

---

## Recommended Fix

Set `vllm_tensor_parallel_size: 8` in the training config to match the actual GPU distribution of
the teacher model. This tells vllm to coordinate inference across all 8 GPUs, consistent with how
`device_map: auto` distributes the 70B model's weights.

```yaml
gold:
  vllm_tensor_parallel_size: 8  # match device_map: auto distribution across 8 GPUs
  vllm_gpu_memory_utilization: 0.25
```

---

## Next Steps

- [ ] Test with `vllm_tensor_parallel_size: 8` to verify the fix
- [ ] Open issue / track compatibility between vllm 0.18.1 colocate mode and HF `device_map: auto`
- [ ] Revisit transformers 5.x once a vllm version supporting it is released (expected 0.19+)
