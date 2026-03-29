"""Tier 0 runtime checks — model config, tokenizer, architecture validation.

These checks download model configs and tokenizers from HuggingFace (small files,
no weights) and validate:
- Model config loads
- Tokenizer loads and has required special tokens (EOS, pad)
- FSDP transformer_layer_cls exists in the model architecture
- LoRA target modules exist in the model architecture
- Chat template patterns match the tokenizer's actual format
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from config_health.core.models import (
    REMOTE_ENGINES,
    CheckResult,
    CheckStatus,
    ConfigEntry,
    ConfigType,
    Severity,
)

# Config types where we should try loading model config/tokenizer
_MODEL_TYPES = frozenset(
    {
        ConfigType.TRAINING,
        ConfigType.INFERENCE,
        ConfigType.EVALUATION,
        ConfigType.TUNING,
    }
)


@dataclass
class _ArchInfo:
    """Model architecture information from meta-device loading."""

    module_names: set[str] = field(default_factory=set)
    transformer_layer_classes: list[str] = field(default_factory=list)
    error: str | None = None


# Per-run caches — shared across configs that reference the same model.
# Cuts tier0 time by ~60% since many configs share models (e.g. 13 configs
# reference meta-llama/Meta-Llama-3.1-8B-Instruct).
_hf_config_cache: dict[str, Any] = {}
_tokenizer_cache: dict[str, Any] = {}
_arch_cache: dict[str, _ArchInfo] = {}
def clear_tier0_cache() -> None:
    """Clear the per-run model cache. Call between independent runs."""
    _hf_config_cache.clear()
    _tokenizer_cache.clear()
    _arch_cache.clear()


def run_tier0_checks(entry: ConfigEntry) -> list[CheckResult]:
    """Run tier 0 checks: model config, tokenizer, architecture validation."""
    results: list[CheckResult] = []

    if entry.config_type not in _MODEL_TYPES:
        return results
    if not entry.model_name:
        return results
    if entry.model_name.startswith(("./", "/", "~")):
        return results
    if entry.engine and entry.engine in REMOTE_ENGINES:
        return results

    model_name = entry.model_name

    # Skip GGUF models — they use llama.cpp, not HF AutoConfig/AutoTokenizer
    if "gguf" in model_name.lower() or "gguf" in entry.path.lower():
        return results

    # Skip local checkpoint paths (e.g., "output/model.fft/checkpoint-800")
    if not "/" in model_name or model_name.startswith("output/"):
        return results

    # 1. Model config loading
    hf_config = _load_hf_config(model_name)
    if hf_config is None:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="model_config_load",
                status=CheckStatus.FAIL,
                message=f"Cannot load model config: {model_name}",
                severity=Severity.ERROR,
            )
        )
        return results  # Can't do further checks without config

    results.append(
        CheckResult(
            config_path=entry.path,
            check_name="model_config_load",
            status=CheckStatus.PASS,
            message=f"Model config loads ({getattr(hf_config, 'model_type', '?')})",
            severity=Severity.INFO,
        )
    )

    # B6: Check if model requires trust_remote_code
    results.extend(_check_trust_remote_code(entry, model_name))

    # 2. Tokenizer loading + special token checks
    tokenizer = _load_tokenizer(model_name)
    if tokenizer is None:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="tokenizer_load",
                status=CheckStatus.FAIL,
                message=f"Cannot load tokenizer: {model_name}",
                severity=Severity.ERROR,
            )
        )
    else:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="tokenizer_load",
                status=CheckStatus.PASS,
                message=f"Tokenizer loads: {model_name}",
                severity=Severity.INFO,
            )
        )
        results.extend(_check_special_tokens(entry, tokenizer, model_name))

    # Only do architecture-aware checks for training configs
    if entry.config_type != ConfigType.TRAINING:
        # Chat template checks are useful for inference too
        if tokenizer is not None:
            results.extend(_check_chat_templates(entry, tokenizer))
        return results

    # Load raw YAML for training-specific checks
    data = _load_yaml(entry.abs_path)
    if data is None:
        return results

    # Chat template validation (needs tokenizer)
    if tokenizer is not None:
        results.extend(_check_chat_templates(entry, tokenizer))

    # 3. Architecture-aware checks (FSDP layer cls, LoRA targets)
    fsdp = data.get("fsdp", {})
    peft = data.get("peft", {})
    training = data.get("training", {})
    needs_arch = (
        (isinstance(fsdp, dict) and fsdp.get("enable_fsdp") and fsdp.get("transformer_layer_cls"))
        or (isinstance(training, dict) and training.get("use_peft") and isinstance(peft, dict) and peft.get("lora_target_modules"))
    )

    if needs_arch:
        arch = _load_architecture_info(model_name)
        if arch.error:
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="arch_load",
                    status=CheckStatus.WARN,
                    message=f"Could not load architecture: {arch.error[:150]}",
                    severity=Severity.WARNING,
                )
            )
        else:
            results.extend(_check_fsdp_layer_cls(entry, data, arch))
            results.extend(_check_lora_targets(entry, data, arch))

    # B2: model_max_length vs max_position_embeddings
    results.extend(_check_max_length_vs_context(entry, data, hf_config))

    # B3: FSDP + QLoRA dtype consistency
    results.extend(_check_fsdp_qlora_dtype(entry, data))

    # B4: Gradient accumulation + FSDP + PEFT LoRA
    results.extend(_check_grad_accum_fsdp_peft(entry, data))

    # B5: pad_token == eos_token risk
    if tokenizer is not None:
        results.extend(_check_pad_eos_collision(entry, tokenizer, data))

    # B7: DeepSpeed batch size consistency (external config files)
    results.extend(_check_deepspeed_batch_size(entry, data))

    return results


# ── Helpers ──────────────────────────────────────────────────────────


def _load_hf_config(model_name: str) -> Any:
    if model_name in _hf_config_cache:
        return _hf_config_cache[model_name]
    try:
        import transformers

        result = transformers.AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
    except Exception as e:
        # Only cache definitive failures (not found, invalid config), not transient ones
        if _is_transient_error(e):
            return None
        result = None
    _hf_config_cache[model_name] = result
    return result


def _load_tokenizer(model_name: str) -> Any:
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]
    try:
        import transformers

        result = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    except Exception as e:
        if _is_transient_error(e):
            return None
        result = None
    _tokenizer_cache[model_name] = result
    return result


def _is_transient_error(e: Exception) -> bool:
    """Check if an exception is likely transient (network, rate limit)."""
    err_str = str(e).lower()
    transient_signals = ("timeout", "connection", "rate limit", "429", "503", "502")
    return any(s in err_str for s in transient_signals)


from config_health.core.scanner import load_yaml_cached as _load_yaml  # shared cache


def _load_architecture_info(model_name: str) -> _ArchInfo:
    """Load model architecture on meta device (no weights) to get module names."""
    if model_name in _arch_cache:
        return _arch_cache[model_name]
    info = _load_architecture_info_uncached(model_name)
    _arch_cache[model_name] = info
    return info


def _load_architecture_info_uncached(model_name: str) -> _ArchInfo:
    """Load model architecture on meta device (no weights) to get module names."""
    try:
        import torch
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Handle composite configs (e.g., Llama 4 multimodal)
        instantiation_config, model_class = _get_config_and_class(config)

        model = None
        if model_class is not None:
            try:
                with torch.device("meta"):
                    model = model_class(instantiation_config)
            except Exception:
                pass

        if model is None:
            try:
                with torch.device("meta"):
                    model = transformers.AutoModelForCausalLM.from_config(
                        config, trust_remote_code=True
                    )
            except Exception:
                pass

        if model is None:
            return _ArchInfo(error="Could not instantiate model on meta device")

        module_names: set[str] = set()
        layer_classes: list[str] = []

        for name, module in model.named_modules():
            if "." in name:
                module_names.add(name.split(".")[-1])
            else:
                module_names.add(name)

            cls_name = module.__class__.__name__
            if any(
                kw in cls_name
                for kw in ("DecoderLayer", "TransformerBlock", "Block")
            ):
                if name and name.split(".")[-1].isdigit():
                    if cls_name not in layer_classes:
                        layer_classes.append(cls_name)

        return _ArchInfo(
            module_names=module_names,
            transformer_layer_classes=layer_classes,
        )
    except Exception as e:
        return _ArchInfo(error=str(e)[:200])


def _get_config_and_class(config: Any) -> tuple[Any, Any]:
    """Get the right config + class for instantiation, handling composite configs."""
    import transformers

    # Composite configs (e.g., Llama 4) nest text_config inside main config
    if hasattr(config, "text_config") and config.text_config is not None:
        text_config = config.text_config
        model_class = transformers.AutoModelForCausalLM._model_mapping.get(
            type(text_config), None
        )
        if model_class is not None:
            return text_config, model_class

    model_class = transformers.AutoModelForCausalLM._model_mapping.get(
        type(config), None
    )
    return config, model_class


# ── Check functions ──────────────────────────────────────────────────


def _check_special_tokens(
    entry: ConfigEntry, tokenizer: Any, model_name: str
) -> list[CheckResult]:
    """Check that tokenizer has EOS and pad tokens."""
    results: list[CheckResult] = []

    if tokenizer.eos_token is None:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="eos_token",
                status=CheckStatus.FAIL,
                message=f"Tokenizer has no eos_token (required for training)",
                severity=Severity.ERROR,
            )
        )

    if tokenizer.pad_token is None:
        # Check if the raw YAML sets tokenizer_pad_token
        data = _load_yaml(entry.abs_path) or {}
        model_cfg = data.get("model", {})
        has_pad_override = isinstance(model_cfg, dict) and (
            model_cfg.get("tokenizer_pad_token")
            or (model_cfg.get("tokenizer_kwargs") or {}).get("pad_token")
        )
        if has_pad_override:
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="pad_token",
                    status=CheckStatus.PASS,
                    message="pad_token set via config override",
                    severity=Severity.INFO,
                )
            )
        else:
            # Only warn for training configs — inference doesn't always need padding
            if entry.config_type == ConfigType.TRAINING:
                results.append(
                    CheckResult(
                        config_path=entry.path,
                        check_name="pad_token",
                        status=CheckStatus.WARN,
                        message=(
                            f"Tokenizer has no pad_token. "
                            "Set model.tokenizer_pad_token in config."
                        ),
                        severity=Severity.WARNING,
                    )
                )

    return results


def _check_fsdp_layer_cls(
    entry: ConfigEntry, data: dict, arch: _ArchInfo
) -> list[CheckResult]:
    """Validate FSDP transformer_layer_cls exists in model architecture."""
    results: list[CheckResult] = []
    fsdp = data.get("fsdp", {})
    if not isinstance(fsdp, dict) or not fsdp.get("enable_fsdp"):
        return results

    layer_cls_raw = fsdp.get("transformer_layer_cls")
    if not layer_cls_raw:
        return results

    detected = arch.transformer_layer_classes
    if not detected:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="fsdp_layer_cls",
                status=CheckStatus.WARN,
                message="Could not detect transformer layer classes in model",
                severity=Severity.WARNING,
            )
        )
        return results

    # Handle comma-separated class lists and fully-qualified names
    # e.g., "transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer"
    # or "MllamaSelfAttentionDecoderLayer,MllamaCrossAttentionDecoderLayer"
    configured_classes = [c.strip() for c in layer_cls_raw.split(",")]
    # Extract short class name from FQN (e.g., "a.b.Foo" -> "Foo")
    configured_short = [c.split(".")[-1] for c in configured_classes]

    missing = [
        name
        for name in configured_short
        if name not in detected
    ]

    if not missing:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="fsdp_layer_cls",
                status=CheckStatus.PASS,
                message=f"FSDP layer classes found in model: {configured_short}",
                severity=Severity.INFO,
            )
        )
    else:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="fsdp_layer_cls",
                status=CheckStatus.FAIL,
                message=(
                    f"FSDP transformer_layer_cls not found: {missing}. "
                    f"Detected: {detected}"
                ),
                severity=Severity.ERROR,
            )
        )

    return results


def _check_lora_targets(
    entry: ConfigEntry, data: dict, arch: _ArchInfo
) -> list[CheckResult]:
    """Validate LoRA target modules exist in model architecture."""
    results: list[CheckResult] = []
    training = data.get("training", {})
    peft = data.get("peft", {})

    if not isinstance(training, dict) or not training.get("use_peft"):
        return results
    if not isinstance(peft, dict):
        return results

    targets = peft.get("lora_target_modules", [])
    if not targets or targets == ["all-linear"]:
        return results

    # Filter out "all-linear" (PEFT magic value) before checking against arch
    concrete_targets = [t for t in targets if t != "all-linear"]
    if not concrete_targets:
        return results

    missing = [t for t in concrete_targets if t not in arch.module_names]
    if missing:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="lora_targets",
                status=CheckStatus.FAIL,
                message=f"LoRA target modules not found in model: {missing}",
                severity=Severity.ERROR,
                details=f"Available: {sorted(arch.module_names)[:30]}",
            )
        )
    else:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="lora_targets",
                status=CheckStatus.PASS,
                message=f"All {len(concrete_targets)} LoRA target modules found in model",
                severity=Severity.INFO,
            )
        )

    return results


def _check_max_length_vs_context(
    entry: ConfigEntry, data: dict, hf_config: Any
) -> list[CheckResult]:
    """B2: Warn if model_max_length exceeds the model's native context window."""
    results: list[CheckResult] = []
    model_cfg = data.get("model", {})
    if not isinstance(model_cfg, dict):
        return results

    configured_max_len = model_cfg.get("model_max_length")
    if not isinstance(configured_max_len, int) or configured_max_len <= 0:
        return results

    # Get native context from HF config
    cfg = hf_config
    if hasattr(cfg, "text_config") and cfg.text_config is not None:
        cfg = cfg.text_config
    native_max = getattr(cfg, "max_position_embeddings", 0) or 0
    if native_max <= 0:
        return results

    if configured_max_len > native_max:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="max_length_context",
                status=CheckStatus.FAIL,
                message=(
                    f"model_max_length ({configured_max_len}) exceeds model's "
                    f"max_position_embeddings ({native_max}). Training will produce "
                    "garbage or fail with RoPE extrapolation errors."
                ),
                severity=Severity.ERROR,
            )
        )

    return results


def _check_fsdp_qlora_dtype(
    entry: ConfigEntry, data: dict
) -> list[CheckResult]:
    """B3: Check FSDP + QLoRA dtype consistency.

    When FSDP is enabled with QLoRA, quant_storage dtype must match the FSDP
    mixed precision dtype. Mismatches cause silent OOM or crashes.
    """
    results: list[CheckResult] = []
    fsdp = data.get("fsdp", {})
    peft = data.get("peft", {})
    training = data.get("training", {})
    model_cfg = data.get("model", {})

    if not isinstance(fsdp, dict) or not fsdp.get("enable_fsdp"):
        return results
    if not isinstance(training, dict) or not training.get("use_peft"):
        return results
    if not isinstance(peft, dict) or not peft.get("q_lora"):
        return results

    # Get quant_storage dtype
    model_kwargs = (model_cfg.get("model_kwargs", {}) or {}) if isinstance(model_cfg, dict) else {}
    quant_config = model_kwargs.get("quantization_config", {}) or {}
    quant_storage = quant_config.get("bnb_4bit_quant_storage", "")

    # Get FSDP mixed precision dtype
    fsdp_dtype = fsdp.get("mixed_precision_dtype", "")

    if quant_storage and fsdp_dtype and quant_storage != fsdp_dtype:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="fsdp_qlora_dtype",
                status=CheckStatus.FAIL,
                message=(
                    f"FSDP + QLoRA dtype mismatch: quant_storage={quant_storage} "
                    f"but FSDP mixed_precision_dtype={fsdp_dtype}. "
                    "These must match per HuggingFace FSDP-QLoRA docs."
                ),
                severity=Severity.ERROR,
            )
        )

    return results


def _check_grad_accum_fsdp_peft(
    entry: ConfigEntry, data: dict
) -> list[CheckResult]:
    """B4: Warn about gradient accumulation + FSDP + PEFT LoRA + bf16.

    This combination triggers RuntimeError: expected dtype float for *end*
    but got dtype c10::BFloat16.
    """
    results: list[CheckResult] = []
    fsdp = data.get("fsdp", {})
    training = data.get("training", {})
    model_cfg = data.get("model", {})

    if not isinstance(fsdp, dict) or not fsdp.get("enable_fsdp"):
        return results
    if not isinstance(training, dict) or not training.get("use_peft"):
        return results

    grad_accum = training.get("gradient_accumulation_steps", 1)
    if not isinstance(grad_accum, int) or grad_accum <= 1:
        return results

    dtype = (model_cfg.get("torch_dtype_str", "") if isinstance(model_cfg, dict) else "")
    mixed = training.get("mixed_precision_dtype", "")
    uses_bf16 = dtype == "bfloat16" or mixed in ("bf16", "bfloat16")

    if uses_bf16:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="grad_accum_fsdp_peft",
                status=CheckStatus.WARN,
                message=(
                    "gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause "
                    "RuntimeError (dtype mismatch). See: "
                    "https://discuss.huggingface.co/t/105006"
                ),
                severity=Severity.WARNING,
            )
        )

    return results


def _check_pad_eos_collision(
    entry: ConfigEntry, tokenizer: Any, data: dict
) -> list[CheckResult]:
    """B5: Warn when pad_token is set to eos_token.

    The model can learn to ignore EOS during training, leading to generation
    that never terminates.
    """
    results: list[CheckResult] = []

    pad_token = tokenizer.pad_token
    eos_token = tokenizer.eos_token

    # Check if config explicitly sets pad_token to eos_token
    model_cfg = data.get("model", {})
    if isinstance(model_cfg, dict):
        override = model_cfg.get("tokenizer_pad_token", "")
        if override and override == eos_token:
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="pad_eos_collision",
                    status=CheckStatus.WARN,
                    message=(
                        "pad_token is set to eos_token. The model may learn to ignore "
                        "EOS, causing generation to never terminate. Consider using a "
                        "dedicated pad token."
                    ),
                    severity=Severity.WARNING,
                )
            )
            return results

    # Check tokenizer defaults
    if pad_token and eos_token and pad_token == eos_token:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="pad_eos_collision",
                status=CheckStatus.WARN,
                message=(
                    f"Tokenizer's pad_token == eos_token ('{eos_token}'). The model "
                    "may learn to ignore EOS during training. Consider setting a "
                    "dedicated pad token via model.tokenizer_pad_token."
                ),
                severity=Severity.WARNING,
            )
        )

    return results


def _check_trust_remote_code(
    entry: ConfigEntry, model_name: str
) -> list[CheckResult]:
    """B6: Check trust_remote_code usage.

    Rule: trust_remote_code should default to False unless the model requires it.
    - WARN if the model requires it but config doesn't set it.
    - WARN if config sets it to True but the model doesn't need it.
    """
    results: list[CheckResult] = []
    data = _load_yaml(entry.abs_path)
    if data is None:
        return results

    model_cfg = data.get("model", {})
    if not isinstance(model_cfg, dict):
        return results

    model_kwargs = model_cfg.get("model_kwargs", {}) or {}
    config_sets_trust = bool(
        model_cfg.get("trust_remote_code")
        or model_kwargs.get("trust_remote_code")
    )

    # Try loading without trust_remote_code to see if it's needed
    model_requires_trust = False
    try:
        import transformers

        transformers.AutoConfig.from_pretrained(
            model_name, trust_remote_code=False
        )
    except Exception as e:
        if "trust_remote_code" in str(e).lower():
            model_requires_trust = True

    if model_requires_trust and not config_sets_trust:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="trust_remote_code",
                status=CheckStatus.WARN,
                message=(
                    f"Model '{model_name}' requires trust_remote_code=True "
                    "but the config doesn't set it."
                ),
                severity=Severity.WARNING,
            )
        )
    elif config_sets_trust and not model_requires_trust:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="trust_remote_code",
                status=CheckStatus.WARN,
                message=(
                    f"Config sets trust_remote_code=True but model '{model_name}' "
                    "does not require it. Prefer trust_remote_code=False for security."
                ),
                severity=Severity.WARNING,
            )
        )

    return results


def _check_deepspeed_batch_size(
    entry: ConfigEntry, data: dict
) -> list[CheckResult]:
    """B7: Validate DeepSpeed batch size consistency in external config files.

    When train_batch_size in the DeepSpeed config doesn't match
    per_device_train_batch_size * gradient_accumulation_steps * num_gpus,
    training crashes at startup.
    """
    import json
    import os

    results: list[CheckResult] = []
    ds = data.get("deepspeed", {})
    if not isinstance(ds, dict) or not ds.get("enable_deepspeed"):
        return results

    config_file = ds.get("config_file")
    if not config_file or not isinstance(config_file, str):
        return results

    # Resolve path
    full_path = config_file
    if not os.path.isabs(config_file):
        # Try relative to config file first, then repo root
        config_dir = os.path.dirname(entry.abs_path)
        candidate = os.path.join(config_dir, config_file)
        if os.path.exists(candidate):
            full_path = candidate

    if not os.path.exists(full_path):
        return results  # File reference check already covers missing files

    try:
        with open(full_path) as f:
            ds_config = json.load(f)
    except Exception:
        return results

    train_batch = ds_config.get("train_batch_size")
    if train_batch == "auto" or train_batch is None:
        return results  # "auto" is the recommended oumi setting

    # Check consistency
    training = data.get("training", {})
    if not isinstance(training, dict):
        return results

    per_device = training.get("per_device_train_batch_size", 1)
    grad_accum = training.get("gradient_accumulation_steps", 1)

    if isinstance(train_batch, int) and isinstance(per_device, int) and isinstance(grad_accum, int):
        # We don't know num_gpus statically, but we can check the per-GPU batch
        expected_per_gpu = per_device * grad_accum
        ds_per_gpu_batch = ds_config.get("train_micro_batch_size_per_gpu")
        if isinstance(ds_per_gpu_batch, int) and ds_per_gpu_batch != per_device:
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="deepspeed_batch_size",
                    status=CheckStatus.FAIL,
                    message=(
                        f"DeepSpeed train_micro_batch_size_per_gpu ({ds_per_gpu_batch}) "
                        f"!= per_device_train_batch_size ({per_device}). "
                        "Training will crash at startup."
                    ),
                    severity=Severity.ERROR,
                )
            )

        ds_grad_accum = ds_config.get("gradient_accumulation_steps")
        if isinstance(ds_grad_accum, int) and ds_grad_accum != grad_accum:
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="deepspeed_grad_accum",
                    status=CheckStatus.WARN,
                    message=(
                        f"DeepSpeed gradient_accumulation_steps ({ds_grad_accum}) "
                        f"!= training.gradient_accumulation_steps ({grad_accum}). "
                        "Consider using 'auto' in the DeepSpeed config."
                    ),
                    severity=Severity.WARNING,
                )
            )

    return results


def _check_chat_templates(
    entry: ConfigEntry, tokenizer: Any
) -> list[CheckResult]:
    """Validate that configured chat templates match the tokenizer's format."""
    results: list[CheckResult] = []
    data = _load_yaml(entry.abs_path)
    if data is None:
        return results

    # Get collator_kwargs from data.train
    train_data = (data.get("data") or {}).get("train", {})
    if not isinstance(train_data, dict):
        return results
    collator_kwargs = train_data.get("collator_kwargs", {})
    if not isinstance(collator_kwargs, dict):
        return results

    response_template = collator_kwargs.get("response_template", "")
    instruction_template = collator_kwargs.get("instruction_template", "")

    if not response_template and not instruction_template:
        return results

    if not hasattr(tokenizer, "apply_chat_template"):
        return results

    # Render a sample conversation to get actual template format
    try:
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return results

    if not isinstance(rendered, str):
        return results

    # Check response_template appears in rendered output
    if response_template:
        tmpl = response_template.strip()
        if tmpl not in rendered and response_template not in rendered:
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="response_template",
                    status=CheckStatus.FAIL,
                    message=(
                        f"response_template '{tmpl}' not found in rendered chat. "
                        "The model uses a different format."
                    ),
                    severity=Severity.ERROR,
                    details=f"Rendered: {rendered[:300]}",
                )
            )
        else:
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="response_template",
                    status=CheckStatus.PASS,
                    message="response_template matches tokenizer",
                    severity=Severity.INFO,
                )
            )
            # B1: Verify token IDs match — catches BPE boundary mismatches
            # where the string appears in text but tokenizes differently in context
            results.extend(
                _check_collator_token_ids(entry, tokenizer, response_template, rendered)
            )

        # Check special tokens are recognized (encode to single tokens)
        results.extend(
            _check_template_special_tokens(entry, tokenizer, response_template, "response_template")
        )

    # Check instruction_template
    if instruction_template:
        tmpl = instruction_template.strip()
        if tmpl not in rendered and instruction_template not in rendered:
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="instruction_template",
                    status=CheckStatus.FAIL,
                    message=(
                        f"instruction_template '{tmpl}' not found in rendered chat. "
                        "The model uses a different format."
                    ),
                    severity=Severity.ERROR,
                    details=f"Rendered: {rendered[:300]}",
                )
            )
        else:
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="instruction_template",
                    status=CheckStatus.PASS,
                    message="instruction_template matches tokenizer",
                    severity=Severity.INFO,
                )
            )

        results.extend(
            _check_template_special_tokens(entry, tokenizer, instruction_template, "instruction_template")
        )

    return results


def _check_collator_token_ids(
    entry: ConfigEntry,
    tokenizer: Any,
    response_template: str,
    rendered_conversation: str,
) -> list[CheckResult]:
    """B1: Verify that the response_template token IDs appear as a contiguous
    subsequence in the tokenized conversation.

    The TRL DataCollatorForCompletionOnlyLM tokenizes response_template and
    searches for those token IDs in the input. If they don't appear (due to BPE
    boundary effects), the collator silently masks the entire sequence — training
    runs but learns nothing.
    """
    results: list[CheckResult] = []
    try:
        # Tokenize the template the same way the collator does
        template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        # Tokenize the full rendered conversation
        conversation_ids = tokenizer.encode(rendered_conversation, add_special_tokens=False)

        if not template_ids:
            return results

        # Search for template_ids as a contiguous subsequence
        found = False
        for i in range(len(conversation_ids) - len(template_ids) + 1):
            if conversation_ids[i : i + len(template_ids)] == template_ids:
                found = True
                break

        if not found:
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="collator_token_ids",
                    status=CheckStatus.FAIL,
                    message=(
                        "response_template token IDs not found in tokenized conversation. "
                        "The collator will silently mask all labels — training will learn nothing. "
                        f"Template tokens: {template_ids[:10]}{'...' if len(template_ids) > 10 else ''}"
                    ),
                    severity=Severity.ERROR,
                    details=(
                        "The string appears in the rendered text but tokenizes differently in context "
                        "(BPE boundary effect). Try adjusting the response_template to include a "
                        "leading space or use the exact token boundary from the chat template."
                    ),
                )
            )
        else:
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="collator_token_ids",
                    status=CheckStatus.PASS,
                    message="response_template token IDs found in tokenized conversation",
                    severity=Severity.INFO,
                )
            )
    except Exception:
        pass  # Don't fail the check if tokenization itself fails

    return results


def _check_template_special_tokens(
    entry: ConfigEntry, tokenizer: Any, template: str, template_name: str
) -> list[CheckResult]:
    """Check that special tokens like <|im_start|> in templates are recognized."""
    import re

    results: list[CheckResult] = []
    patterns = re.findall(r"<\|[^|>]+\|?>", template)

    for pattern in patterns:
        tokens = tokenizer.encode(pattern, add_special_tokens=False)
        if len(tokens) > 1:
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name=f"{template_name}_token",
                    status=CheckStatus.FAIL,
                    message=(
                        f"{template_name} has unrecognized special token '{pattern}' "
                        f"(tokenizes to {len(tokens)} tokens instead of 1)"
                    ),
                    severity=Severity.ERROR,
                )
            )

    return results
