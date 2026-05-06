# Config Health Fix Log

Changes made in response to the config-health report (`./report/`).

---

## Real Config Bugs Fixed

### 1. `configs/examples/misc/tulu3_sft_mini.yaml` — `model_max_length` too large
- **Issue**: `model_max_length: 4096` for `EleutherAI/pythia-14m`, which only supports 2048 tokens (`max_position_embeddings = 2048`). Would produce garbage output or RoPE extrapolation errors.
- **Fix**: Changed `model_max_length: 4096` → `model_max_length: 2048`.

---

### 2. `configs/recipes/falcon_h1/dpo/falcon_h1_0_5b/qlora_dpo.yaml` — `model_max_length` off-by-one
- **Issue**: `model_max_length: 16385` for `tiiuae/Falcon-H1-0.5B-Instruct`, which has `max_position_embeddings = 16384`. Off by one.
- **Fix**: Changed `model_max_length: 16385` → `model_max_length: 16384`.

---

### 3–6. Phi-3-mini LoRA target modules wrong in 4 configs
- **Issue**: Phi-3-mini uses **fused** projections (`qkv_proj`, `gate_up_proj`) not separate ones (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`). The SFT/DPO/KTO configs had wrong LoRA targets that would match nothing in the model, effectively training no attention/MLP layers.
- **Verified via**: `inspect.getsource(modeling_phi3)` shows `self.qkv_proj` and `self.gate_up_proj`.
- **Fix**: Replaced `q_proj, k_proj, v_proj, gate_proj, up_proj` → `qkv_proj, gate_up_proj` in:
  - `configs/recipes/phi3/dpo/train.yaml`
  - `configs/recipes/phi3/kto/train.yaml`
  - `configs/recipes/phi3/sft/lora_train.yaml`
  - `configs/recipes/phi3/sft/lora_macos_train.yaml`

---

### 7. `configs/recipes/phi4/sft/reasoning_plus/qlora_train.yaml` — phi4 MLP modules wrong
- **Issue**: Phi-4-reasoning-plus (also `model_type: phi3`) uses `gate_up_proj` (fused gate+up), not separate `gate_proj` and `up_proj`. The config had `gate_proj, up_proj` which don't exist in the model.
- **Verified via**: meta-device model load confirmed `gate_up_proj` present, `gate_proj`/`up_proj` absent.
- **Fix**: Replaced `gate_proj, up_proj` → `gate_up_proj`. Kept `qkv_proj, o_proj, down_proj` which were already correct.

---

### 8–9. Phi-3-vision LoRA target modules wrong in 2 configs
- **Issue**: Same as above for the text decoder in Phi-3-vision-128k-instruct. The vision model's CLIP encoder does have separate `q_proj, k_proj, v_proj` (so those are fine), but the text decoder's MLP uses `gate_up_proj` (fused), not `gate_proj, up_proj`.
- **Verified via**: meta-device load of `microsoft/Phi-3-vision-128k-instruct` confirmed `gate_up_proj` present, `gate_proj`/`up_proj` absent.
- **Fix**: Replaced `gate_proj, up_proj` → `gate_up_proj` in:
  - `configs/recipes/vision/phi3/dpo/train.yaml`
  - `configs/recipes/vision/phi3/sft/lora/train.yaml`

---

## Config-Health False Positives Fixed

### 10. Hub checker: oumi dataset class names flagged as missing HF Hub datasets
- **Issue**: `hub_checker.py` tried to look up dataset names like `text_sft_jsonl`, `HuggingFaceDataset`, `hf_vision`, `PromptResponseDataset`, `vision_dpo_jsonl` on HuggingFace Hub. These are **oumi registered dataset class names**, not HF Hub IDs. HF Hub IDs always contain a `/` (e.g. `allenai/tulu-3-sft-mixture`).
- **Affected configs**: coalm, halloumi, limo, dcvlr, vision/phi3/dpo, vision/phi3/sft configs.
- **Fix**: In `hub_checker.py::_check_dataset`, skip dataset names that don't contain `/` with status `SKIP` and message "Oumi registered dataset (not an HF Hub ID)".

---

### 11. Hub checker: `output/` model paths flagged as missing HF Hub models
- **Issue**: Falcon-E evaluation configs reference locally-built model checkpoints (e.g. `output/falcon_e_1b.fft/checkpoint-800-quantized`). The hub checker only skipped paths starting with `./`, `/`, `~`, but not `output/`.
- **Fix**: Added `"output/"` to the local-path prefix list in `hub_checker.py::_check_model`.

---

### 12. `max_length_context` check: Molmo configs with explicit `max_position_embeddings` override
- **Issue**: The Molmo dcvlr configs set `model_max_length: 8192` alongside `model_kwargs.max_position_embeddings: 8192`, explicitly overriding the model's native context limit. This is intentional. The check only handled `rope_scaling` as a context extension signal.
- **Affected configs**: `configs/projects/dcvlr/starter_kit/molmo-d-train-openr1.yaml`, `molmo-o-train-openr1.yaml`.
- **Fix**: In `tier0_checks.py::_check_max_length_vs_context`, also skip the check if `model_kwargs.max_position_embeddings` is set.

---

### 13. Non-oumi YAML files reported as "Could not parse" errors
- **Issue**: Two YAML files in `configs/` are not oumi configs but were scanned and flagged as FAIL:
  - `configs/examples/deploy/fireworks_deploy.yaml` — a Fireworks provider deployment config
  - `configs/recipes/llama3_1/sft/8b_full/accelerate.yaml` — an HF Accelerate launcher config
- **Fix**: In `classifier.py::_validate_with_oumi`, after all oumi class parsings fail, check if the YAML has **any** oumi-recognized top-level keys (`model`, `data`, `training`, `tasks`, etc.). If it has none, return `"Not an oumi config file"` instead of `"Could not parse..."`. In `static_checks.py::_check_parse`, errors starting with `"Not an oumi config file"` are reported as `SKIP` (INFO) instead of `FAIL` (ERROR).

---

## Not Fixed (Expected / Environment Limitations)

The following failures in the report are **not bugs** and were left as-is:

- **Dry-run failures: insufficient GPU/RAM** — Most configs require more VRAM/RAM than the CI machine has (e.g. 44 GB GPU with 8B+ models needing full fine-tuning). These are expected — the configs are meant to run on larger hardware.
- **`causal_conv1d_cuda` not available** — Falcon-H1 models require `causal-conv1d` which is not installed. This is a dependency issue, not a config bug.
- **Model not found: unreleased models** — `allenai/Olmo-3-32B-Instruct`, `Qwen/Qwen3-30B-A3B-Instruct`, `Qwen/Qwen3-Next-80B-A3B`, `openai/gpt-oss-*` — configs for upcoming model releases that don't exist on HF Hub yet.
- **`grpo_verl_geometry3k/train.yaml` VLM error** — Uses `Qwen2.5-VL` with `AutoModelForCausalLM` which doesn't support VLMs; this is a known limitation of that config's trainer setup.
- **Coverage gaps** — Missing inference/eval configs for some model families are informational warnings, not errors.
