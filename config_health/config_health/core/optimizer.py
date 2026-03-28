"""Config optimization suggestions."""

from __future__ import annotations

import yaml

from config_health.core.models import (
    ConfigEntry,
    ConfigType,
    OptimizationSuggestion,
)


def suggest_optimizations(entry: ConfigEntry) -> list[OptimizationSuggestion]:
    """Generate optimization suggestions for a config."""
    data = _load_yaml(entry.abs_path)
    if data is None:
        return []

    suggestions: list[OptimizationSuggestion] = []

    if entry.config_type == ConfigType.TRAINING:
        suggestions.extend(_training_suggestions(entry, data))
    elif entry.config_type == ConfigType.INFERENCE:
        suggestions.extend(_inference_suggestions(entry, data))
    elif entry.config_type == ConfigType.EVALUATION:
        suggestions.extend(_evaluation_suggestions(entry, data))

    return suggestions


def _training_suggestions(
    entry: ConfigEntry, data: dict
) -> list[OptimizationSuggestion]:
    suggestions: list[OptimizationSuggestion] = []
    training = data.get("training", {})
    model = data.get("model", {})
    fsdp = data.get("fsdp", {})

    if not isinstance(training, dict):
        return suggestions
    if not isinstance(model, dict):
        model = {}
    if not isinstance(fsdp, dict):
        fsdp = {}

    # Missing gradient checkpointing
    if not training.get("enable_gradient_checkpointing"):
        suggestions.append(
            OptimizationSuggestion(
                config_path=entry.path,
                category="performance",
                title="Enable gradient checkpointing",
                suggestion=(
                    "Add `enable_gradient_checkpointing: true` to reduce memory "
                    "usage at a small speed cost."
                ),
                priority="medium",
            )
        )

    # Missing bf16
    dtype = model.get("torch_dtype_str", "")
    mixed = training.get("mixed_precision_dtype", "")
    if dtype not in ("bfloat16", "float16") and mixed not in ("bf16", "fp16"):
        suggestions.append(
            OptimizationSuggestion(
                config_path=entry.path,
                category="performance",
                title="Use mixed precision",
                suggestion=(
                    "Set `model.torch_dtype_str: bfloat16` for faster training "
                    "and lower memory usage."
                ),
                priority="medium",
            )
        )

    # Large batch without gradient accumulation
    batch_size = training.get("per_device_train_batch_size", 1)
    grad_accum = training.get("gradient_accumulation_steps", 1)
    if isinstance(batch_size, int) and batch_size > 4 and grad_accum == 1:
        suggestions.append(
            OptimizationSuggestion(
                config_path=entry.path,
                category="performance",
                title="Consider gradient accumulation",
                suggestion=(
                    f"Batch size is {batch_size} with no gradient accumulation. "
                    "Consider reducing batch size and using gradient_accumulation_steps "
                    "for better memory efficiency."
                ),
                priority="low",
            )
        )

    # Full finetune without LoRA on potentially large model
    use_peft = training.get("use_peft", False)
    if not use_peft and fsdp.get("enable_fsdp"):
        suggestions.append(
            OptimizationSuggestion(
                config_path=entry.path,
                category="efficiency",
                title="Consider LoRA",
                suggestion=(
                    "Full finetune with FSDP — consider LoRA (use_peft: true) "
                    "for faster training with fewer resources."
                ),
                priority="low",
            )
        )

    # Missing logging_steps
    if not training.get("logging_steps"):
        suggestions.append(
            OptimizationSuggestion(
                config_path=entry.path,
                category="best_practice",
                title="Set logging_steps",
                suggestion="Add `logging_steps: 10` to monitor training progress.",
                priority="low",
            )
        )

    return suggestions


def _inference_suggestions(
    entry: ConfigEntry, data: dict
) -> list[OptimizationSuggestion]:
    suggestions: list[OptimizationSuggestion] = []
    gen = data.get("generation", {})
    model = data.get("model", {})

    if isinstance(gen, dict):
        batch = gen.get("batch_size", 1)
        if isinstance(batch, int) and batch == 1:
            suggestions.append(
                OptimizationSuggestion(
                    config_path=entry.path,
                    category="performance",
                    title="Increase inference batch size",
                    suggestion=(
                        "Batch size is 1 — increase for better throughput "
                        "if memory allows."
                    ),
                    priority="low",
                )
            )

    if isinstance(model, dict):
        dtype = model.get("torch_dtype_str", "")
        if dtype not in ("bfloat16", "float16"):
            engine = data.get("engine", data.get("inference_engine", ""))
            if isinstance(engine, str) and engine.upper() == "NATIVE":
                suggestions.append(
                    OptimizationSuggestion(
                        config_path=entry.path,
                        category="performance",
                        title="Use half precision for inference",
                        suggestion="Set `model.torch_dtype_str: bfloat16` for faster inference.",
                        priority="medium",
                    )
                )

    return suggestions


def _evaluation_suggestions(
    entry: ConfigEntry, data: dict
) -> list[OptimizationSuggestion]:
    suggestions: list[OptimizationSuggestion] = []
    model = data.get("model", {})

    if isinstance(model, dict):
        if not model.get("trust_remote_code"):
            suggestions.append(
                OptimizationSuggestion(
                    config_path=entry.path,
                    category="compatibility",
                    title="Enable trust_remote_code",
                    suggestion=(
                        "Some models require `trust_remote_code: True` to load. "
                        "Consider enabling it for broader compatibility."
                    ),
                    priority="low",
                )
            )

    return suggestions


def _load_yaml(path: str) -> dict | None:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None
