# Quantize runtime test — run report

Branch: `oelachqar/quantize-refactor-per-backend-files` @ `c05d524c`
Date: 2026-05-06
Hardware: NVIDIA H100 80GB HBM3 (SM 9.0)
Stack: torch 2.10.0+cu128, transformers 4.57.6, llmcompressor 0.10.0.2, bitsandbytes 0.49.2, compressed-tensors 0.14.0.1
Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## Result

`python scripts/quantize/test_quantize_runtime.py --cleanup` — **exit 0, all green**.

| # | Case | Time | Output size |
|---|---|---|---|
| 1 | fp8_dynamic:auto | 10.5s | 1.2 GB |
| 2 | fp8_block:auto | 8.7s | 1.2 GB |
| 3 | w4a16:auto | 141.7s | 730.1 MB |
| 4 | w4a16_asym:auto | 27.3s | 733.8 MB |
| 5 | w8a16:auto | 96.2s | 1.2 GB |
| 6 | bnb_nf4:auto | 2.6s | 730.6 MB |
| 7 | bnb_fp4:auto | 2.5s | 730.6 MB |
| 8 | bnb_int8:auto | 3.5s | 1.2 GB |
| 9 | fp8_dynamic:gptq | 90.1s | 1.2 GB |
| 10 | fp8_dynamic:awq | 21.2s | 1.2 GB |
| 11 | w4a16:rtn | 11.2s | 730.1 MB |

Negative tests: 3/3 passed (BnB+gptq, LLMC+bnb, bad output_format — each raised `OumiConfigError` with the expected substring).

Per-case validation passed: `QuantizationResult` returned, `config.json` + ≥1 `.safetensors` + tokenizer artifact present, dir size > 1 MB, reported size matched dir size, `AutoConfig.from_pretrained(output_dir)` succeeded.

HF auth was not required — `HuggingFaceH4/ultrachat_200k` (used by the calibration cases w4a16 / w4a16_asym / w8a16 / fp8_dynamic:gptq / fp8_dynamic:awq / w4a16:rtn) downloaded anonymously.

## Issues hit while bringing up the run

### 1. Test-script import shadowed by `oumi.quantize` submodule (fixed in `c05d524c`)

`scripts/quantize/test_quantize_runtime.py:_run_case` originally did
`from oumi import quantize as oumi_quantize`. Running the script triggered
`TypeError: 'module' object is not callable` on the first quantize call, after
`_print_env()` had already done `from oumi.quantize import _SCHEME_TO_BACKEND`.

Root cause is in `src/oumi/__init__.py:278` — there is a `def quantize(config)`
wrapper at the top level *and* a sibling `oumi.quantize` package. After
`oumi/__init__.py` finishes, `oumi.quantize` resolves to the function. As soon
as anything imports a name from the submodule, Python rebinds the
`oumi.quantize` attribute on the `oumi` package to the module (standard
submodule-import behaviour), and the function reference is lost. From that point
on, `from oumi import quantize` returns the module.

The script fix sidesteps this by importing from the submodule directly:

```python
from oumi.quantize import quantize as oumi_quantize
```

The underlying issue in `src/oumi/__init__.py` is **pre-existing on this branch**
(not introduced by the per-backend refactor) and breaks the documented public
API `from oumi import quantize; quantize(config)` whenever any other code path
has touched the submodule first. Worth a separate fix — either drop the wrapper
in favour of `oumi.quantize.quantize`, or rename one of the two collidng names.

### 2. Stale `torch._inductor.kernel.unpack_mixed_mm.py` from a prior install

`pip install --force-reinstall torch==2.10.0` did not remove
`unpack_mixed_mm.py`, which is no longer in torch 2.10's RECORD. Its
`from .mm_common import mm_args, mm_configs, ...` raised
`ImportError: cannot import name 'mm_configs'` at import time, killing
`import llmcompressor`. Deleting the orphan file (and its `.pyc`) was enough to
restore imports — torch 2.10's own RECORD does not list it.

This is environment-only and unrelated to the test or this branch.

### 3. Disk pressure on `/`

A failed `pip install --force-reinstall` filled `/` (20 GB rootfs, 78% used
before the run). `pip cache purge` (~7 GB) was needed before reinstall could
complete. Worth keeping in mind if the harness is wired into CI on a constrained
runner.
