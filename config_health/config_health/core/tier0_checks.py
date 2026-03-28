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

# Remote inference engines — model_name is a provider ID, not an HF repo
_REMOTE_ENGINES = frozenset(
    {
        "ANTHROPIC",
        "OPENAI",
        "GOOGLE",
        "GOOGLE_GEMINI",
        "GOOGLE_VERTEX",
        "OPENROUTER",
        "TOGETHER",
        "FIREWORKS",
        "PARASAIL",
        "LAMBDA",
        "REMOTE",
        "REMOTE_VLLM",
    }
)


@dataclass
class _ArchInfo:
    """Model architecture information from meta-device loading."""

    module_names: set[str] = field(default_factory=set)
    transformer_layer_classes: list[str] = field(default_factory=list)
    error: str | None = None


def run_tier0_checks(entry: ConfigEntry) -> list[CheckResult]:
    """Run tier 0 checks: model config, tokenizer, architecture validation."""
    results: list[CheckResult] = []

    if entry.config_type not in _MODEL_TYPES:
        return results
    if not entry.model_name:
        return results
    if entry.model_name.startswith(("./", "/", "~")):
        return results
    if entry.engine and entry.engine in _REMOTE_ENGINES:
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

    return results


# ── Helpers ──────────────────────────────────────────────────────────


def _load_hf_config(model_name: str) -> Any:
    try:
        import transformers

        return transformers.AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
    except Exception:
        return None


def _load_tokenizer(model_name: str) -> Any:
    try:
        import transformers

        return transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    except Exception:
        return None


def _load_yaml(path: str) -> dict | None:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _load_architecture_info(model_name: str) -> _ArchInfo:
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

    missing = [t for t in targets if t not in arch.module_names]
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
                message=f"All {len(targets)} LoRA target modules found in model",
                severity=Severity.INFO,
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
