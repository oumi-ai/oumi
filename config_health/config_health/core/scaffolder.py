"""Config scaffolding — generate new configs from templates.

Smart scaffold: auto-detects VLM models, queries the Oumi registry,
and generates configs with the right collator, LoRA targets, and batch size.
"""

from __future__ import annotations

import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_SCAFFOLDS_DIR = Path(__file__).parent.parent / "scaffolds"

# Task type to template file mapping
_TASK_TEMPLATES = {
    "training": "training.yaml.j2",
    "sft": "training.yaml.j2",
    "inference": "inference.yaml.j2",
    "evaluation": "evaluation.yaml.j2",
    "job": "job.yaml.j2",
}


def get_available_tasks() -> list[str]:
    """Return available scaffold task types."""
    return sorted(set(_TASK_TEMPLATES.keys()))


def scaffold_config(
    model_name: str,
    task_type: str,
    output_dir: str | None = None,
    *,
    use_lora: bool = True,
    dataset_name: str = "yahma/alpaca-cleaned",
    batch_size: int = 4,
) -> str:
    """Generate a config YAML from a template.

    Auto-detects VLM models and adjusts settings accordingly.
    Returns the generated YAML string, or file path if output_dir is set.
    """
    template_name = _TASK_TEMPLATES.get(task_type)
    if not template_name:
        raise ValueError(
            f"Unknown task type: {task_type}. Available: {get_available_tasks()}"
        )

    env = Environment(
        loader=FileSystemLoader(str(_SCAFFOLDS_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template(template_name)

    # Auto-detect model properties
    model_info = _detect_model_info(model_name)

    # Derive short model name for output dirs
    model_short = model_name.split("/")[-1].lower().replace("-", "_")

    # Adjust settings based on detection
    effective_batch_size = batch_size
    if model_info.get("params_b", 0) >= 30:
        effective_batch_size = min(batch_size, 1)
    elif model_info.get("params_b", 0) >= 7:
        effective_batch_size = min(batch_size, 2)

    context = {
        "model_name": model_name,
        "model_short_name": model_short,
        "task_type": task_type,
        "use_lora": use_lora,
        "dataset_name": dataset_name,
        "batch_size": effective_batch_size,
        # Auto-detected fields
        "is_vlm": model_info.get("is_vlm", False),
        "trust_remote_code": model_info.get("needs_trust_remote_code", False),
        "model_type": model_info.get("model_type", ""),
        "lora_targets": model_info.get("lora_targets", []),
        "collator_name": model_info.get("collator_name", ""),
        "params_b": model_info.get("params_b", 0),
    }

    rendered = template.render(**context)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{task_type}.yaml"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w") as f:
            f.write(rendered)
        return output_path

    return rendered


def _detect_model_info(model_name: str) -> dict:
    """Auto-detect model properties from HF and Oumi registry."""
    info: dict = {
        "is_vlm": False,
        "needs_trust_remote_code": False,
        "model_type": "",
        "lora_targets": [],
        "collator_name": "",
        "params_b": 0.0,
    }

    if "/" not in model_name:
        return info

    try:
        import transformers

        # Load HF config
        config = transformers.AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
        model_type = getattr(config, "model_type", "")
        info["model_type"] = model_type
        info["is_vlm"] = (
            hasattr(config, "text_config") and config.text_config is not None
        )

        # Check if trust_remote_code is needed
        try:
            transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=False)
        except Exception as e:
            if "trust_remote_code" in str(e).lower():
                info["needs_trust_remote_code"] = True

        # Estimate params from inner config
        inner = config.text_config if info["is_vlm"] and config.text_config else config
        hidden = getattr(inner, "hidden_size", 0) or 0
        layers = getattr(inner, "num_hidden_layers", 0) or 0
        vocab = getattr(inner, "vocab_size", 0) or 0
        inter = getattr(inner, "intermediate_size", 0) or 0
        if hidden and layers and vocab:
            embedding = vocab * hidden
            mlp = 3 * hidden * inter if inter else 4 * hidden * hidden
            attn = 4 * hidden * hidden  # approximate
            params = embedding + layers * (attn + mlp + 2 * hidden) + hidden
            info["params_b"] = round(params / 1e9, 1)

        # Check Oumi registry for VLM-specific settings
        try:
            from oumi.core.configs.internal.supported_models import get_all_models_map

            oumi_map = get_all_models_map()
            model_info = oumi_map.get(model_type)
            if model_info:
                cfg = model_info.config
                if cfg.visual_config is not None:
                    info["is_vlm"] = True
                    info["collator_name"] = "vision_language_sft"
        except Exception:
            pass

        # Detect LoRA targets from architecture
        if info.get("params_b", 0) > 0:
            try:
                import torch

                inner_config = (
                    config.text_config
                    if info["is_vlm"] and config.text_config
                    else config
                )
                with torch.device("meta"):
                    model = transformers.AutoModelForCausalLM.from_config(
                        inner_config, trust_remote_code=True
                    )
                module_names = set()
                for name, _ in model.named_modules():
                    if "." in name:
                        module_names.add(name.split(".")[-1])
                # Pick standard LoRA targets that exist
                standard = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
                info["lora_targets"] = [t for t in standard if t in module_names]
                del model
            except Exception:
                info["lora_targets"] = ["q_proj", "k_proj", "v_proj", "o_proj"]

    except Exception:
        pass

    return info
