# Offline Tests Progress

## Goal
Make all tests in the pretests and GPU tests workflows work without HuggingFace Hub access by pre-downloading all required resources.

## Approach
1. Run tests with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1` and a fresh `HF_HOME`
2. Identify failures caused by missing HF resources
3. Add required downloads to the download scripts
4. Re-run until all offline-related failures are resolved

---

## CPU Unit Tests (pretests workflow)

**Script**: `scripts/download_pretest_resources.sh`
**Test command**: `pytest tests/unit/ -m "not e2e and not e2e_eternal and not single_gpu and not multi_gpu"`

### Progress

#### Run 1 - Initial baseline (no downloads)
- **Result**: 1 collection error, could not proceed
- **Issue**: `test_sglang_inference_engine.py` calls `SGLangInferenceEngine()` at module level (via `@pytest.mark.parametrize` + `_generate_all_engines()`), which calls `build_tokenizer` for `openai-community/gpt2` and `llava-hf/llava-1.5-7b-hf`
- **Fix**: Download gpt2 and llava-1.5-7b-hf tokenizer + config files

#### Run 2 - After gpt2 + llava download
- **Result**: 40 failed, 4211 passed, 45 skipped
- **Failing categories**:
  - `test_data_mixtures`: needs `tasksource/mmlu` dataset (abstract_algebra subset)
  - `test_models`, `test_supported_models`: needs configs for SmolLM2-135M-Instruct, SmolVLM-Instruct, blip2-opt-2.7b, Phi-3-vision, Qwen2-VL-2B-Instruct
  - `test_chat_templates`: needs tokenizers for Phi-3-vision, Phi-3-mini, Qwen2-VL, Qwen2.5-VL
  - `test_huggingface_vision_dataset`, `test_vision_language_jsonlines_dataset`: needs Salesforce/blip2-opt-2.7b config (via `find_internal_model_config_using_model_name`)
  - `test_analysis_utils`: needs SmolVLM-256M-Instruct config
  - `test_hf_utils`: needs SmolLM2-135M-Instruct, SmolVLM-256M-Instruct tokenizer_config
- **Fix**: Download all 10 model configs/tokenizers

#### Run 3 - After all model downloads
- **Result**: 5 failed, 4246 passed
- **Issue**: Phi-3 models use `trust_remote_code=True`, requiring Python module files (not just JSON)
- **Fix**: Download `*.py` files for `microsoft/Phi-3-vision-128k-instruct` and `microsoft/Phi-3-mini-4k-instruct`

#### Run 4 - After Phi-3 .py files
- **Result**: 2 failed, 4249 passed
- **Issue**: `tasksource/mmlu` dataset was downloaded via `snapshot_download` (parquet files only), but `datasets.load_dataset` needs the arrow cache built
- **Fix**: Use `datasets.load_dataset()` in the download script to build the full builder cache

#### Run 5 - Final (clean cache, full script)
- **Result**: 4251 passed, 45 skipped, 0 failed
- All tests pass in fully offline mode

### Models (config + tokenizer files only, no weights)
| Model | Why |
|-------|-----|
| `openai-community/gpt2` | sglang engine parametrize, chat templates, hf_utils |
| `llava-hf/llava-1.5-7b-hf` | sglang engine parametrize, chat templates |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | test_models, test_hf_utils |
| `HuggingFaceTB/SmolVLM-256M-Instruct` | test_hf_utils, test_analysis_utils |
| `HuggingFaceTB/SmolVLM-Instruct` | test_models (is_image_text_llm) |
| `Salesforce/blip2-opt-2.7b` | vision dataset tests, test_supported_models |
| `microsoft/Phi-3-vision-128k-instruct` | chat templates, test_models (needs *.py for trust_remote_code) |
| `microsoft/Phi-3-mini-4k-instruct` | chat templates (needs *.py for trust_remote_code) |
| `Qwen/Qwen2-VL-2B-Instruct` | chat templates, test_models, test_supported_models |
| `Qwen/Qwen2.5-VL-3B-Instruct` | chat templates |

### Datasets
| Dataset | Why |
|---------|-----|
| `tasksource/mmlu` (abstract_algebra, test split) | test_data_mixtures |

---

## GPU Integration Tests (gpu_tests workflow)

**Script**: `tests/scripts/predownload_for_github_gpu_tests.sh`
**Test command**: `pytest tests/integration/ -m "not e2e and not e2e_eternal and not multi_gpu"`

### Progress

#### Run 1 - With existing script resources only
- **Result**: 16 failed, 4 errors, 39 passed, 54 skipped
- **Offline failures**:
  - `yahma/alpaca-cleaned` dataset: `snapshot_download` doesn't build datasets arrow cache (test_train_basic, test_train_unregistered_metrics_function)
  - `cais/mmlu` dataset: lm_harness uses `cais/mmlu` not `tasksource/mmlu` (test_evaluate_lm_harness, test_get_task_dict)
  - `HuggingFaceTB/SmolLM2-360M-Instruct`: teacher model for GKD not downloaded (test_train_gkd)
  - `microsoft/Phi-3-vision-128k-instruct`: needs *.py files for trust_remote_code (test_vision_language_completions_only - 4 errors)
  - `Qwen/Qwen2.5-0.5B`: model for verl tests not downloaded (test_verl_train - 4 failures)
  - `d1shs0ap/countdown`: dataset for verl tests not downloaded (test_verl_grpo_train_1_step)
- **Non-offline failures** (environment issues):
  - 4x cuDNN not initialized (test_infer_*_with_images) - needs proper GPU
  - 1x missing `vllm_ascend` (test_train_gold) - package issue
  - 1x missing `torchdata` (test_integration_cnn_classifier) - package issue
  - 1x Ray timeout (test_verl_grpo_train_1_step) - needs GPU for training
- **Fix**: Added all missing resources to download script

#### Run 2 - After adding missing resources
- **Result**: 7 failed, 52 passed, 54 skipped - 0 offline failures
- All 7 remaining failures are environment issues (cuDNN/GPU, missing packages), not offline

### Models (config + tokenizer + weights)
| Model | Why |
|-------|-----|
| `openai-community/gpt2` | test_train (basic, dpo, kto, etc.), test_evaluate_lm_harness |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | test_train (gkd student), test_infer |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | test_train (gkd teacher) |
| `HuggingFaceTB/SmolVLM-256M-Instruct` | test_infer (with images) |
| `Qwen/Qwen3-0.6B` | test_train_gold |
| `Qwen/Qwen2.5-0.5B` | test_verl_train |
| `microsoft/Phi-3-vision-128k-instruct` | test_vision_language_completions_only (needs *.py) |

### Datasets
| Dataset | Why |
|---------|-----|
| `cais/mmlu` (abstract_algebra + college_computer_science) | test_evaluate_lm_harness |
| `yahma/alpaca-cleaned` | test_train (basic, unregistered_metrics) |
| `d1shs0ap/countdown` | test_verl_train (verl grpo) |

---

## Key Findings

1. **Collection-time downloads**: `test_sglang_inference_engine.py` triggers tokenizer downloads during pytest collection because `_generate_all_engines()` is called inside `@pytest.mark.parametrize`. This blocks ALL test collection.

2. **trust_remote_code models**: Phi-3 models need Python files (`.py`) in addition to JSON configs because they use custom model classes via `get_class_from_dynamic_module`.

3. **datasets vs snapshot_download**: For HF datasets, `snapshot_download` downloads parquet files but doesn't build the arrow cache. The `datasets` library needs `load_dataset()` to be run once online to create the builder cache for offline use. The old script using `hf download --repo-type dataset` is insufficient.

4. **Different MMLU sources**: Unit tests use `tasksource/mmlu`, but lm_harness evaluation uses `cais/mmlu`. Both need to be downloaded.

5. **Environment variables needed**: Three env vars are required for full offline mode:
   - `HF_HUB_OFFLINE=1` - huggingface_hub library
   - `TRANSFORMERS_OFFLINE=1` - transformers library
   - `HF_DATASETS_OFFLINE=1` - datasets library
