"""Config scaffolding — generate new configs from templates."""

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

    Returns the generated YAML string.
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

    # Derive short model name for output dirs
    model_short = model_name.split("/")[-1].lower().replace("-", "_")

    context = {
        "model_name": model_name,
        "model_short_name": model_short,
        "task_type": task_type,
        "use_lora": use_lora,
        "dataset_name": dataset_name,
        "batch_size": batch_size,
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
