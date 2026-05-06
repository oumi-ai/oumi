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

---

### 14. `configs/recipes/vision/llava_7b/dpo/train.yaml` — FSDP without `transformer_layer_cls`
- **Issue**: FSDP configured with `auto_wrap_policy: TRANSFORMER_BASED_WRAP` but no `transformer_layer_cls`. Without specifying which layers to wrap, FSDP cannot properly shard the model.
- **Fix**: Added `transformer_layer_cls: "LlamaDecoderLayer,CLIPEncoderLayer"` — LLaVA 1.5 has two transformer components: a Llama-2 language model and a CLIP vision encoder.

---

### 15. `configs/projects/coalm/8b_train.yaml` — PEFT enabled with no `lora_target_modules`
- **Issue**: `training.use_peft: True` but no `peft:` section in the config. Without specifying LoRA targets, PEFT would fall back to framework defaults (typically all linear layers), which is likely not the intent.
- **Fix**: Added a `peft:` section with standard Llama-3.1 LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (r=16, alpha=32).

---

### 16–18. Three inference configs missing `generation.max_new_tokens`
- **Issue**: Remote API inference configs with no `max_new_tokens` set, which can cause runaway generation.
- **Fix**: Added `generation.max_new_tokens: 2048` to:
  - `configs/recipes/gpt_oss/inference/120b_together_infer.yaml`
  - `configs/recipes/qwen3/inference/235b_a22b_fireworks_infer.yaml`
  - `configs/recipes/qwen3/inference/235b_a22b_together_infer.yaml`

---

## Config-Health False Positives Fixed (continued)

### 19. VLM configs incorrectly reported as dry-run failures
- **Issue**: The dry-run check used `AutoModelForCausalLM` for all training configs. VLM models (Qwen2VL, Qwen2.5VL, Qwen3VL, etc.) are not in the CausalLM model mapping, causing `ValueError: Unrecognized configuration class`. This affected configs for Qwen2-VL, Qwen2.5-VL, Qwen3-VL, and the grpo_verl_geometry3k config.
- **Fix**: In `dry_run.py::_execute_dry_run`, added VLM detection before the CausalLM lookup. If the model maps to `AutoModelForImageTextToText` or `AutoModelForVision2Seq`, the dry-run is skipped with a clear message ("VLM model: dry-run uses text-only dummy data").

---

### 20. Batch removal of unnecessary `trust_remote_code: True` from model configs (~263 configs)
- **Issue**: The vast majority of oumi training/inference/eval configs set `model.trust_remote_code: True`, including models like Llama, Qwen, Gemma, GPT2, Falcon-E, Falcon-H1, Phi, DeepSeek, SmolLM, etc. that are natively supported by transformers and do not require remote code execution. This is a security risk (allows arbitrary Python from HuggingFace Hub to run) with no functional benefit for these models.
- **Exception**: 5 `oumi-ai/Molmo-*` configs were kept, but this also inadvertently removed `trust_remote_code` from models that genuinely need it — fixed in change #26.
- **Fix**: Removed `trust_remote_code: True` from the `model:` section of 263 YAML configs across `configs/`.

---

### 21. Missing `tokenizer_pad_token` in 23 Llama-3.x training configs
- **Issue**: 23 training configs using Llama-3.1 or Llama-3.2 models did not set `tokenizer_pad_token`. The Llama-3.x tokenizer's EOS token would be reused as pad, which can confuse the model during training (padding positions look like sequence terminators).
- **Affected**: llama3_1 SFT (8b_full, 8b_full/longctx, 8b_lora, 8b_lora/fsdp, 8b_qlora, 70b_full, 70b_lora, 70b_qlora), llama3_1/pretraining/8b, llama3_2 SFT (1b_full, 3b_full, 3b_full/fsdp, 3b_lora, 3b_lora/fsdp, 3b_qlora, 3b_qlora/fsdp, 1b_qlora_dpo), deepspeed examples (z2/z3/z3_offload), letter_counting/grpo, aya/sft, wc50m/base_ultrachat.
- **Fix**: Added `tokenizer_pad_token: "<|finetune_right_pad_id|>"` to all 23 configs. This matches the dedicated pad token already used in the 405b variants.

---

### 22. `configs/recipes/vision/phi4/sft/lora/train.yaml` — phi4-multimodal-instruct LoRA targets wrong
- **Issue**: Phi-4-multimodal-instruct uses the same `Phi3`-family text decoder as Phi-3/Phi-4-reasoning-plus, which has fused MLP projections (`gate_up_proj`). The config had separate `gate_proj, up_proj` which don't exist in the model.
- **Fix**: Replaced `gate_proj, up_proj` → `gate_up_proj`. Kept `q_proj, k_proj, v_proj, o_proj, down_proj`.

---

---

### 23. Missing `torch_dtype_str` in 3 GRPO/VERL training configs
- **Issue**: `grpo_verl_countdown/train.yaml`, `grpo_verl_gsm8k/train.yaml`, and `grpo_verl_geometry3k/train.yaml` had no `torch_dtype_str` set. Without it, models load in float32 (default), consuming 2× more GPU memory and running slower. All modern GPUs support bfloat16.
- **Fix**: Added `torch_dtype_str: "bfloat16"` to the `model:` section of all three configs.

---

### 24. Missing `logging_steps` in phi3 training configs and other minimal configs
- **Issue**: 10 training configs had no `logging_steps` set, meaning HF Trainer defaults to logging every 500 steps — too infrequent for short training runs and makes debugging difficult.
- **Affected**: `phi3/sft/lora_train.yaml`, `phi3/sft/lora_macos_train.yaml`, `phi3/dpo/train.yaml`, `phi3/dpo/macos_train.yaml`, `phi3/dpo/nvidia_24g_train.yaml`, `phi3/dpo/nvidia_80g_train.yaml`, `phi3/kto/train.yaml`, `falcon_e/dpo/train.yaml`, `gpt2/pretraining/train.yaml`, `gpt2/pretraining/macos_train.yaml`, `llama3_2/dpo/1b_qlora_dpo.yaml`, `grpo_verl_countdown/train.yaml`, `grpo_verl_geometry3k/train.yaml`, `vision/molmo/grpo/train.yaml`.
- **Fix**: Added `logging_steps: 10` to all affected configs.

---

---

### 25. Missing `torch_dtype_str` in phi3, phi3-vision, InternVL3, Qwen2-VL, and other training configs
- **Issue**: Several training and inference configs had no `torch_dtype_str` set. Without it, models load in float32 by default (2× memory, slower training/inference). These configs were inconsistent with their sibling configs.
- **Affected**:
  - `phi3/dpo/train.yaml`, `phi3/kto/train.yaml` → `bfloat16`
  - `phi3/sft/lora_macos_train.yaml` → `float16` (macOS, consistent with `phi3/dpo/macos_train.yaml`)
  - `vision/internvl3/sft/full/train.yaml` → `bfloat16`
  - `vision/phi3/dpo/train.yaml` → `bfloat16`
  - `vision/qwen2_5_vl_3b/dpo/train.yaml`, `vision/qwen2_vl_2b/dpo/train.yaml` → `bfloat16`
  - `gpt2/inference/infer.yaml` → `bfloat16`
  - `grpo_verl_countdown`, `grpo_verl_gsm8k`, `grpo_verl_geometry3k` → `bfloat16` (see change #23)
- **Skipped**: `vision/molmo/*` configs — Molmo uses a custom image processor with float32-sensitive operations; `dcvlr/molmo-*` configs already explicitly set `float32`; `gpt_oss` inference configs — reference unreleased models.

---

## Not Fixed (Expected / Environment Limitations)

The following failures in the report are **not bugs** and were left as-is:

- **Dry-run failures: insufficient GPU/RAM** — Most configs require more VRAM/RAM than the CI machine has (e.g. 44 GB GPU with 8B+ models needing full fine-tuning). These are expected — the configs are meant to run on larger hardware.
- **`causal_conv1d_cuda` not available** — Falcon-H1 models require `causal-conv1d` which is not installed. This is a dependency issue, not a config bug.
- **Model not found: unreleased models** — `allenai/Olmo-3-32B-Instruct`, `Qwen/Qwen3-30B-A3B-Instruct`, `Qwen/Qwen3-Next-80B-A3B`, `openai/gpt-oss-*` — configs for upcoming model releases that don't exist on HF Hub yet.
- **`grpo_verl_geometry3k/train.yaml` VLM error** — Uses `Qwen2.5-VL` with `AutoModelForCausalLM` which doesn't support VLMs; this is a known limitation of that config's trainer setup.
- **Coverage gaps** — Missing inference/eval configs for some model families are informational warnings, not errors.
- **`pad_token == eos_token` for Phi-3/Phi-4/DeepSeek-R1/SmolLM/Molmo configs** — These model families don't have a standard dedicated pad token in their tokenizers (their EOS token doubles as pad). Community convention for all these is to leave it as-is. Llama-3.x configs were fixed in changes #21 (they have `<|finetune_right_pad_id|>`).
- **`gradient_accumulation + FSDP + PEFT LoRA + bf16` dtype warnings** — Several configs (gemma3 lora, gpt_oss lora, llama3.1 lora+FSDP) have this combination. The warning is valid but fixing it requires either gradient_accumulation_steps=1 or switching peft to fp32, which changes training behavior.
- **`DynamicCache.from_legacy_cache` / `SlidingWindowCache` import errors** — These are transformers version compatibility issues for phi3-vision and phi4-vision. Not config bugs.
- **Unreleased/gated models** — `allenai/Olmo-3-32B-Instruct`, `Qwen/Qwen3-30B-A3B-Instruct`, `Qwen/Qwen3-Next-80B-A3B` don't exist on HF Hub yet; their configs are forward-looking placeholders.

---

## Fixes from Second Report (report-2)

### 26. `trust_remote_code` incorrectly removed from 11 configs that require it
- **Issue**: The batch removal in change #20 was too broad — it removed `trust_remote_code: True` from `microsoft/Phi-3-vision-128k-instruct`, `microsoft/Phi-4-multimodal-instruct`, `THUDM/glm-4-9b-chat`, and `nvidia/Llama-3.3-Nemotron-Super-49B-v1`, all of which require `trust_remote_code` to load their custom code from the Hub.
- **Affected**: `vision/phi3` (5 configs: dpo, vllm_infer, sft/full, sft/full/completions_only, sft/lora), `vision/phi4` (4 configs: infer, vllm_infer, sft/full, sft/lora), `glm4/inference/air_vllm_infer.yaml`, `llama3_3/inference/nemotron_super_49b_vllm_infer.yaml`.
- **Fix**: Restored `trust_remote_code: True` to the `model:` section of all 11 configs.

---

### 27. Config-health dry_run: `_LazyAutoMapping.get()` missing default argument
- **Issue**: The VLM detection code added in fix #19 called `_vlm_auto._model_mapping.get(type(hf_config))` without a default argument. In newer versions of transformers, `_LazyAutoMapping.get()` requires a default — causing `TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'` for every config that passed through the VLM detection path. This caused 149 of 149 non-RAM dry-run failures.
- **Fix**: Changed to `_vlm_auto._model_mapping.get(type(hf_config), None)` in `dry_run.py`.

---

## Fixes from Third Report (report-3)

### 28. `vision/phi3/sft/full/train.yaml` and `completions_only_train.yaml` — `freeze_layers` structure broken
- **Issue**: The `trust_remote_code` restoration script (fix #26) inserted the key between `freeze_layers:` (which has no inline value, only a comment) and its list item, breaking the YAML structure. The list item `- "model.vision_embed_tokens"` ended up at sibling level of `model:` rather than as a child of `freeze_layers:`.
- **Fix**: Reordered the keys so `freeze_layers:` is immediately followed by its list item `- "model.vision_embed_tokens"`, then `trust_remote_code: True` comes after.

---

### 29. Config-health: dry_run "Skipped: model needs X GB" showed as FAIL
- **Issue**: `dry_run_to_check_results` used exact string matching against `_SKIP_ERRORS` to decide SKIP vs FAIL. "Skipped: model needs ~83.9 GB GPU but only 44.0 GB free" wasn't in `_SKIP_ERRORS`, so it landed in the FAIL branch, producing confusing "Dry-run failed: Skipped: ..." messages.
- **Fix**: Added `or dr.error.startswith("Skipped:")` to the condition, routing all "Skipped:..." messages to SKIP status.

---

### 30. Config-health: environment-specific package errors showed as FAIL
- **Issue**: Missing packages (`causal_conv1d_cuda`, `is_flash_attn_greater_or_equal_2_10`, `SlidingWindowCache`) caused dry-run FAIL for Falcon-H1, Phi-3-vision, and Phi-4-vision configs. These are environment/dependency issues, not config bugs.
- **Fix**: Added `_ENV_SKIP_SUBSTRINGS` and `_is_env_skip()` in `dry_run_to_check_results`. Errors matching these patterns are classified as SKIP with a clear "missing environment package" message.

---

### 31. Config-health: VLM dry-runs were failing with image token mismatch
- **Root cause**: The old fallback `processor(images=img, text="dummy")` produced `pixel_values` (image patches) but `input_ids` with 0 image tokens, because `"dummy"` has no image placeholder. The model's forward pass then raised "Image features and image tokens do not match, tokens: 0, features: N".
- **Fix**: Replaced the broken single-attempt + bad-fallback with a 3-attempt processor call in `dry_run.py`:
  1. `apply_chat_template` with `[{"type":"image"}, {"type":"text","text":"dummy"}]` → `processor(images=[img], text=[prompt])` — correct image tokens in `input_ids` (verified for SmolVLM, InternVL3, Gemma3, Qwen2-VL, Qwen2.5-VL, Qwen3-VL)
  2. `processor.image_token` manual insertion → `processor(images=[img], text=["<tok> dummy"])` — fallback for processors where attempt 1 fails
  3. Text-only `processor(text=["dummy"])` — last resort, notes that vision was not exercised
- **Verified**: End-to-end dry-run now PASSES for SmolVLM-Instruct (21.5 GB), InternVL3-1B (7.6 GB), Qwen2.5-VL-3B (35.6 GB).
- **No SKIP masking**: VLM image token errors are NOT classified as SKIP — if the 3-attempt approach fails, it's a real error worth investigating.

---

### 32. Config-health: model-not-found OSError showed as FAIL in dry_run
- **Issue**: For configs referencing unreleased/renamed models (`allenai/Olmo-3-32B-Instruct`, `Qwen/Qwen3-Next-80B-A3B`), the dry-run raised `OSError: not a valid model identifier`. This was already caught by the `model_exists` check (FAIL) and `model_config_load` (WARN), making the dry_run FAIL redundant and noisy.
- **Fix**: Added "is not a local folder and is not a valid model identifier" to `_ENV_SKIP_SUBSTRINGS`. The dry_run now reports SKIP for these with a clear message.
