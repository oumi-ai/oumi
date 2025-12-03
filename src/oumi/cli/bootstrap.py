# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bootstrap configuration files for new open source models."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import typer
from huggingface_hub import HfApi, model_info
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Model size thresholds (in billions of parameters)
SMALL_MODEL_THRESHOLD = 4  # <= 4B: single GPU, no FSDP
MEDIUM_MODEL_THRESHOLD = 10  # <= 10B: FSDP full fine-tuning or LoRA single GPU
LARGE_MODEL_THRESHOLD = 40  # <= 40B: FSDP + LoRA
# > 40B: FSDP + Q-LoRA

# Known transformer layer classes for different model architectures
# These class names are from transformers library and may change between versions
# Tested with transformers>=4.45.0
TRANSFORMER_LAYER_CLASSES = {
    "llama": "LlamaDecoderLayer",
    "mistral": "MistralDecoderLayer",
    "qwen": "Qwen2DecoderLayer",
    "qwen2": "Qwen2DecoderLayer",
    "qwen3": "Qwen3DecoderLayer",
    "phi": "PhiDecoderLayer",
    "phi3": "Phi3DecoderLayer",
    "phi4": "Phi3DecoderLayer",  # Phi-4 uses same layer class as Phi-3
    "gemma": "GemmaDecoderLayer",
    "gemma2": "Gemma2DecoderLayer",
    "gemma3": "Gemma3DecoderLayer",
    "falcon": "FalconDecoderLayer",
    "starcoder": "GPTBigCodeBlock",
    "starcoder2": "Starcoder2DecoderLayer",
    "gpt2": "GPT2Block",
    "bloom": "BloomBlock",
    "opt": "OPTDecoderLayer",
    "mpt": "MPTBlock",
    "ministral": "MistralDecoderLayer",
    "deepseek": "DeepseekV2DecoderLayer",
    "command": "CohereDecoderLayer",  # Cohere Command-R
    "cohere": "CohereDecoderLayer",
}

# LoRA target modules by architecture
# More modules = more capacity but higher memory usage
LORA_TARGET_MODULES = {
    "default": ["q_proj", "k_proj", "v_proj"],
    "llama": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "mistral": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "ministral": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "qwen": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "qwen2": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "qwen3": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "phi": ["q_proj", "k_proj", "v_proj", "dense"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "phi4": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "gemma": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "gemma2": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "deepseek": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}

# Minimum transformers versions for specific model families
# Models not listed here should work with transformers>=4.45.0
MIN_TRANSFORMERS_VERSIONS = {
    "qwen3": "4.51.0",
    "gemma3": "4.51.0",
    "phi4": "4.48.0",
    "deepseek": "4.46.0",
    "starcoder2": "4.45.0",
}


@dataclass(frozen=True)
class ModelInfo:
    """Information about a model extracted from HuggingFace."""

    model_id: str
    model_name: str
    size_billions: float
    context_length: int
    architecture: str
    is_instruct: bool
    is_base: bool
    is_reasoning: bool
    license: str
    downloads: int
    likes: int


def parse_model_size(model_id: str, safetensors_params: Optional[int] = None) -> float:
    """Extract model size in billions from model ID or safetensors metadata."""
    # First try to use safetensors parameter count if available
    if safetensors_params:
        return safetensors_params / 1e9

    # Otherwise parse from model ID
    model_id_lower = model_id.lower()

    # Check for MoE models first (8x7b pattern)
    moe_match = re.search(r"(\d+)x(\d+)b", model_id_lower)
    if moe_match:
        # MoE model - return total for config purposes
        return float(moe_match.group(1)) * float(moe_match.group(2))

    # Common patterns: 7b, 7B, 7-b, 7_b, 70b, 0.5b, 1.5b, etc.
    size_match = re.search(r"(\d+\.?\d*)[-_]?b(?:illion)?", model_id_lower)
    if size_match:
        return float(size_match.group(1))

    # Default to medium if can't parse
    return 7.0


def parse_context_length(config: dict) -> int:
    """Extract context length from model config."""
    for key in [
        "max_position_embeddings",
        "n_positions",
        "max_seq_len",
        "seq_length",
        "context_length",
    ]:
        if key in config:
            return config[key]
    return 4096  # Default


def detect_architecture(model_id: str, config: dict) -> str:
    """Detect model architecture from model ID or config."""
    model_id_lower = model_id.lower()

    # Sort architecture names by length (longest first) to ensure
    # more specific names like "qwen3" are matched before "qwen"
    sorted_arch_names = sorted(TRANSFORMER_LAYER_CLASSES.keys(), key=len, reverse=True)

    # Check model ID first
    for arch_name in sorted_arch_names:
        if arch_name in model_id_lower:
            return arch_name

    # Check config architectures
    if "architectures" in config and config["architectures"]:
        arch = config["architectures"][0].lower()
        for arch_name in sorted_arch_names:
            if arch_name in arch:
                return arch_name

    return "llama"  # Default to llama as most common


def get_transformer_layer_class(architecture: str) -> str:
    """Get the transformer layer class for FSDP wrapping."""
    return TRANSFORMER_LAYER_CLASSES.get(architecture, "LlamaDecoderLayer")


def get_lora_target_modules(architecture: str) -> list[str]:
    """Get appropriate LoRA target modules for the architecture."""
    return LORA_TARGET_MODULES.get(architecture, LORA_TARGET_MODULES["default"])


def get_min_transformers_version(architecture: str) -> Optional[str]:
    """Get minimum transformers version required for this architecture."""
    return MIN_TRANSFORMERS_VERSIONS.get(architecture)


def fetch_model_info(model_id: str) -> ModelInfo:
    """Fetch model information from HuggingFace Hub."""
    try:
        info = model_info(model_id)

        # Get config if available
        config = {}
        if hasattr(info, "config") and info.config:
            config = info.config

        # Get safetensors params if available
        safetensors_params = None
        if hasattr(info, "safetensors") and info.safetensors:
            safetensors_params = info.safetensors.get("total", None)

        size = parse_model_size(model_id, safetensors_params)
        context_length = parse_context_length(config)
        architecture = detect_architecture(model_id, config)

        model_id_lower = model_id.lower()
        is_instruct = any(
            x in model_id_lower for x in ["instruct", "chat", "it", "sft"]
        )
        is_reasoning = "reasoning" in model_id_lower
        is_base = "base" in model_id_lower or (not is_instruct and not is_reasoning)

        return ModelInfo(
            model_id=model_id,
            model_name=model_id.split("/")[-1],
            size_billions=size,
            context_length=context_length,
            architecture=architecture,
            is_instruct=is_instruct,
            is_base=is_base,
            is_reasoning=is_reasoning,
            license=info.license if hasattr(info, "license") else "unknown",
            downloads=info.downloads if hasattr(info, "downloads") else 0,
            likes=info.likes if hasattr(info, "likes") else 0,
        )
    except Exception as e:
        console.print(f"[red]Error fetching model info for {model_id}: {e}[/red]")
        raise typer.Exit(1)


def fetch_collection_models(collection_url: str) -> list[str]:
    """Fetch model IDs from a HuggingFace collection."""
    # Extract collection ID from URL
    # Format: https://huggingface.co/collections/org/name-hash
    match = re.search(r"collections/([^/]+/[^/\s]+)", collection_url)
    if not match:
        console.print(f"[red]Invalid collection URL: {collection_url}[/red]")
        raise typer.Exit(1)

    collection_id = match.group(1)
    console.print(f"[blue]Fetching collection: {collection_id}[/blue]")

    try:
        api = HfApi()
        collection = api.get_collection(collection_id)

        model_ids = []
        for item in collection.items:
            if item.item_type == "model":
                model_ids.append(item.item_id)

        return model_ids
    except Exception as e:
        console.print(f"[red]Error fetching collection: {e}[/red]")
        raise typer.Exit(1)


def select_best_variants(models: list[ModelInfo], max_configs: int = 6) -> list[tuple]:
    """Select the most useful model variants and config types.

    Returns list of (ModelInfo, config_type) tuples where config_type is one of:
    - 'full': Full fine-tuning
    - 'lora': LoRA fine-tuning
    - 'qlora': Quantized LoRA
    """
    selected = []

    # Group by size
    size_groups: dict[str, list[ModelInfo]] = {}
    for model in models:
        size_key = f"{model.size_billions:.0f}B"
        if size_key not in size_groups:
            size_groups[size_key] = []
        size_groups[size_key].append(model)

    # For each size, prefer Instruct > Reasoning > Base
    for size_key, group in sorted(size_groups.items()):
        # Sort by preference: instruct first, then reasoning, then base
        group.sort(key=lambda m: (not m.is_instruct, not m.is_reasoning, not m.is_base))
        best_model = group[0]  # Take the best variant

        size = best_model.size_billions

        if size <= SMALL_MODEL_THRESHOLD:
            # Small models: full fine-tuning only
            selected.append((best_model, "full"))
        elif size <= MEDIUM_MODEL_THRESHOLD:
            # Medium models: full with FSDP, and LoRA option
            selected.append((best_model, "full"))
            selected.append((best_model, "lora"))
        elif size <= LARGE_MODEL_THRESHOLD:
            # Large models: LoRA with FSDP
            selected.append((best_model, "lora"))
        else:
            # Very large models: Q-LoRA with FSDP
            selected.append((best_model, "qlora"))

    # Limit total configs
    return selected[:max_configs]


def generate_config_yaml(
    model: ModelInfo, config_type: str, output_dir: Path
) -> tuple[str, str]:
    """Generate a YAML config file for the model.

    Returns (filename, content).
    """
    size = model.size_billions
    arch = model.architecture
    transformer_layer = get_transformer_layer_class(arch)
    lora_modules = get_lora_target_modules(arch)
    min_transformers = get_min_transformers_version(arch)

    # Determine training parameters based on size and config type
    if config_type == "full":
        use_peft = False
        use_fsdp = size > SMALL_MODEL_THRESHOLD
        if size <= SMALL_MODEL_THRESHOLD:
            batch_size = 4
            grad_accum = 4
            lr = 2e-5
        else:
            batch_size = 2
            grad_accum = 8
            lr = 5e-6
    elif config_type == "lora":
        use_peft = True
        use_fsdp = size > MEDIUM_MODEL_THRESHOLD
        batch_size = 2
        grad_accum = 8
        lr = 3e-4
    else:  # qlora
        use_peft = True
        use_fsdp = True
        batch_size = 2
        grad_accum = 8
        lr = 3e-4

    # Determine context length for training (cap at 32k for efficiency)
    model_max_length = min(model.context_length, 32768)

    # Build config sections
    config_type_label = {
        "full": "Full fine-tuning (FFT)",
        "lora": "LoRA fine-tuning",
        "qlora": "Q-LoRA fine-tuning",
    }[config_type]

    hardware_reqs = {
        "full": (
            "1x GPU with 24GB+ VRAM"
            if size <= SMALL_MODEL_THRESHOLD
            else "8x GPUs with 80GB VRAM each (A100/H100)"
        ),
        "lora": (
            "1x GPU with 24GB+ VRAM"
            if size <= MEDIUM_MODEL_THRESHOLD
            else "8x GPUs with 80GB VRAM each"
        ),
        "qlora": "8x GPUs with 80GB VRAM each (A100/H100)",
    }[config_type]

    # Generate safe output directory name
    safe_name = model.model_name.lower().replace("-", "_").replace(".", "_")
    output_subdir = f"output/{safe_name}.{config_type}"

    # Build YAML content
    yaml_lines = [
        f"# {config_type_label} config for {model.model_name}.",
        "#",
        "# Requirements:",
        "#   - Log into WandB (`wandb login`) or disable `enable_wandb`",
        f"#   - Hardware: {hardware_reqs}",
    ]

    if min_transformers:
        yaml_lines.append(f"#   - transformers>={min_transformers}")

    if "llama" in model.model_id.lower() or "meta" in model.model_id.lower():
        yaml_lines.append(
            f"#   - Request access: https://huggingface.co/{model.model_id}"
        )

    yaml_lines.extend(
        [
            "#",
            "# Usage:",
        ]
    )

    # Determine the config path for usage
    org_name = model.model_id.split("/")[0].lower().replace("-", "_")
    size_label = f"{model.size_billions:.0f}b" if model.size_billions >= 1 else "small"
    config_filename = "train.yaml"
    config_path = (
        f"configs/recipes/{org_name}/sft/{size_label}_{config_type}/{config_filename}"
    )

    if use_fsdp:
        yaml_lines.append(
            f"#   oumi distributed torchrun -m oumi train -c {config_path}"
        )
    else:
        yaml_lines.append(f"#   oumi train -c {config_path}")

    yaml_lines.extend(
        [
            "#",
            "# See Also:",
            "#   - Documentation: https://oumi.ai/docs/en/latest/user_guides/train/train.html",
            "#   - Config class: oumi.core.configs.TrainingConfig",
            "#   - Config source: https://github.com/oumi-ai/oumi/blob/main/src/oumi/core/configs/training_config.py",
            "#   - Other training configs: configs/**/*train.yaml",
            "",
            "model:",
            f'  model_name: "{model.model_id}"',
            f"  model_max_length: {model_max_length}",
            '  torch_dtype_str: "bfloat16"',
            '  attn_implementation: "sdpa"',
            "  trust_remote_code: True",
            "",
            "data:",
            "  train:",
            "    datasets:",
            '      - dataset_name: "yahma/alpaca-cleaned"  # 51,760 examples',
            "",
            "training:",
            '  trainer_type: "TRL_SFT"',
        ]
    )

    if use_peft:
        yaml_lines.append("  use_peft: True")

    yaml_lines.extend(
        [
            "  save_steps: 200",
            "  num_train_epochs: 1",
            f"  per_device_train_batch_size: {batch_size}",
            f"  gradient_accumulation_steps: {grad_accum}",
            "  max_grad_norm: null",
            "",
            "  enable_gradient_checkpointing: True",
            "  gradient_checkpointing_kwargs:",
            "    use_reentrant: False",
            "  ddp_find_unused_parameters: False",
            '  optimizer: "adamw_torch_fused"',
            f"  learning_rate: {lr}",
            '  lr_scheduler_type: "cosine"',
        ]
    )

    if use_peft:
        yaml_lines.extend(
            [
                "  warmup_steps: 100",
                "  weight_decay: 0.01",
            ]
        )
    else:
        if size > SMALL_MODEL_THRESHOLD:
            yaml_lines.append("  warmup_ratio: 0.1")

    yaml_lines.extend(
        [
            "  compile: False",
            "",
            '  dataloader_num_workers: "auto"',
            "  dataloader_prefetch_factor: 32",
            "",
            "  logging_steps: 100",
            "  empty_device_cache_steps: 50",
            f'  output_dir: "{output_subdir}"',
            "  include_performance_metrics: True",
            "  enable_wandb: True",
        ]
    )

    # Add PEFT section if needed
    if use_peft:
        yaml_lines.extend(
            [
                "",
                "peft:",
                "  lora_r: 16",
                "  lora_alpha: 32",
                "  lora_target_modules:",
            ]
        )
        for module in lora_modules:
            yaml_lines.append(f'    - "{module}"')

        if config_type == "qlora":
            yaml_lines.extend(
                [
                    "  q_lora: True",
                    '  bnb_4bit_quant_type: "nf4"',
                    '  bnb_4bit_compute_dtype: "bfloat16"',
                    '  bnb_4bit_quant_storage: "bfloat16"',
                ]
            )

    # Add FSDP section if needed
    if use_fsdp:
        yaml_lines.extend(
            [
                "",
                "fsdp:",
                "  enable_fsdp: True",
                "  forward_prefetch: True",
            ]
        )

        if config_type == "full" and size > MEDIUM_MODEL_THRESHOLD:
            yaml_lines.append("  cpu_offload: True")

        yaml_lines.extend(
            [
                '  sharding_strategy: "FULL_SHARD"',
                '  auto_wrap_policy: "TRANSFORMER_BASED_WRAP"',
                f'  transformer_layer_cls: "{transformer_layer}"',
            ]
        )

    yaml_lines.append("")  # Trailing newline

    return config_filename, "\n".join(yaml_lines)


def generate_readme(
    models: list[ModelInfo],
    configs: list[tuple[ModelInfo, str]],
    output_dir: Path,
    collection_url: Optional[str] = None,
) -> str:
    """Generate README.md content for the generated configs."""
    # Get org name from first model
    org_name = models[0].model_id.split("/")[0]
    model_family = models[0].model_name.split("-")[0]

    lines = [
        f"# {model_family} Model Configs",
        "",
        f"Training configurations for [{org_name}](https://huggingface.co/{org_name}) {model_family} models.",
        "",
    ]

    if collection_url:
        lines.extend(
            [
                "## Source",
                "",
                f"- Collection: [{collection_url}]({collection_url})",
                "",
            ]
        )

    # Model overview table
    lines.extend(
        [
            "## Models",
            "",
            "| Model | Size | Context | License | Type |",
            "|-------|------|---------|---------|------|",
        ]
    )

    for model in sorted({m for m, _ in configs}, key=lambda m: m.size_billions):
        model_type = (
            "Instruct"
            if model.is_instruct
            else ("Reasoning" if model.is_reasoning else "Base")
        )
        lines.append(
            f"| [{model.model_name}](https://huggingface.co/{model.model_id}) "
            f"| {model.size_billions:.1f}B | {model.context_length:,} | {model.license} | {model_type} |"
        )

    lines.extend(
        [
            "",
            "## Configurations",
            "",
        ]
    )

    # Group configs by type
    full_configs = [(m, t) for m, t in configs if t == "full"]
    lora_configs = [(m, t) for m, t in configs if t == "lora"]
    qlora_configs = [(m, t) for m, t in configs if t == "qlora"]

    org_dir = org_name.lower().replace("-", "_")

    if full_configs:
        lines.extend(
            [
                "### Full Fine-Tuning",
                "",
                "Full parameter fine-tuning for maximum quality. Requires more GPU memory.",
                "",
            ]
        )
        for model, _ in full_configs:
            size_label = (
                f"{model.size_billions:.0f}b" if model.size_billions >= 1 else "small"
            )
            config_path = f"configs/recipes/{org_dir}/sft/{size_label}_full/train.yaml"
            use_fsdp = model.size_billions > SMALL_MODEL_THRESHOLD
            cmd = (
                f"oumi distributed torchrun -m oumi train -c {config_path}"
                if use_fsdp
                else f"oumi train -c {config_path}"
            )
            lines.extend(
                [
                    f"**{model.model_name}** ({model.size_billions:.1f}B)",
                    "```bash",
                    cmd,
                    "```",
                    "",
                ]
            )

    if lora_configs:
        lines.extend(
            [
                "### LoRA Fine-Tuning",
                "",
                "Parameter-efficient fine-tuning. Lower memory requirements, faster training.",
                "",
            ]
        )
        for model, _ in lora_configs:
            size_label = (
                f"{model.size_billions:.0f}b" if model.size_billions >= 1 else "small"
            )
            config_path = f"configs/recipes/{org_dir}/sft/{size_label}_lora/train.yaml"
            use_fsdp = model.size_billions > MEDIUM_MODEL_THRESHOLD
            cmd = (
                f"oumi distributed torchrun -m oumi train -c {config_path}"
                if use_fsdp
                else f"oumi train -c {config_path}"
            )
            lines.extend(
                [
                    f"**{model.model_name}** ({model.size_billions:.1f}B)",
                    "```bash",
                    cmd,
                    "```",
                    "",
                ]
            )

    if qlora_configs:
        lines.extend(
            [
                "### Q-LoRA Fine-Tuning",
                "",
                "Quantized LoRA for very large models. Minimum memory requirements.",
                "",
            ]
        )
        for model, _ in qlora_configs:
            size_label = (
                f"{model.size_billions:.0f}b" if model.size_billions >= 1 else "small"
            )
            config_path = f"configs/recipes/{org_dir}/sft/{size_label}_qlora/train.yaml"
            cmd = f"oumi distributed torchrun -m oumi train -c {config_path}"
            lines.extend(
                [
                    f"**{model.model_name}** ({model.size_billions:.1f}B)",
                    "```bash",
                    cmd,
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## Hardware Requirements",
            "",
            "| Config Type | Model Size | Recommended Hardware |",
            "|-------------|------------|---------------------|",
            "| Full | ≤4B | 1x GPU (24GB+ VRAM) |",
            "| Full | 7-10B | 8x GPUs (80GB each) |",
            "| LoRA | ≤10B | 1x GPU (24GB+ VRAM) |",
            "| LoRA | 14-40B | 8x GPUs (80GB each) |",
            "| Q-LoRA | >40B | 8x GPUs (80GB each) |",
            "",
            "## Customization",
            "",
            "You can customize these configs by:",
            "",
            "1. **Dataset**: Replace `yahma/alpaca-cleaned` with your dataset",
            "2. **Batch size**: Adjust `per_device_train_batch_size` and `gradient_accumulation_steps`",
            "3. **Learning rate**: Tune `learning_rate` for your use case",
            "4. **LoRA rank**: Adjust `lora_r` (higher = more capacity, more memory)",
            "",
            "## Documentation",
            "",
            "- [Training Guide](https://oumi.ai/docs/en/latest/user_guides/train/train.html)",
            "- [FSDP Guide](https://oumi.ai/docs/en/latest/user_guides/train/distributed.html)",
            "- [PEFT Guide](https://oumi.ai/docs/en/latest/user_guides/train/peft.html)",
            "",
        ]
    )

    return "\n".join(lines)


def bootstrap(
    url: Annotated[
        str,
        typer.Argument(
            help="HuggingFace model URL or collection URL "
            "(e.g., https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512 "
            "or https://huggingface.co/collections/mistralai/ministral-3)"
        ),
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output directory for configs (default: configs/recipes/<org>)",
        ),
    ] = None,
    max_configs: Annotated[
        int,
        typer.Option(
            "--max-configs",
            "-m",
            help="Maximum number of config files to generate",
        ),
    ] = 6,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-n",
            help="Preview what would be generated without writing files",
        ),
    ] = False,
):
    """Bootstrap configuration files for new open source models.

    Takes a HuggingFace model page URL or collection URL and generates
    appropriate training configs based on model size:

    - Small models (≤4B): Full fine-tuning, single GPU
    - Medium models (≤10B): Full fine-tuning with FSDP, or LoRA
    - Large models (≤40B): LoRA with FSDP
    - Very large models (>40B): Q-LoRA with FSDP

    Examples:
        # Single model
        oumi bootstrap https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512

        # Collection of models
        oumi bootstrap https://huggingface.co/collections/mistralai/ministral-3

        # Custom output directory
        oumi bootstrap https://huggingface.co/Qwen/Qwen3-8B -o configs/recipes/qwen3

        # Preview without writing
        oumi bootstrap https://huggingface.co/Qwen/Qwen3-8B --dry-run
    """
    console.print(Panel.fit("Oumi Config Bootstrap", style="bold green"))

    # Determine if URL is a collection or single model
    is_collection = "/collections/" in url

    if is_collection:
        console.print("[blue]Fetching models from collection...[/blue]")
        model_ids = fetch_collection_models(url)
        collection_url = url
    else:
        # Extract model ID from URL
        match = re.search(r"huggingface\.co/([^/]+/[^/\s?]+)", url)
        if not match:
            console.print(f"[red]Invalid HuggingFace URL: {url}[/red]")
            raise typer.Exit(1)
        model_ids = [match.group(1)]
        collection_url = None

    console.print(f"[green]Found {len(model_ids)} model(s)[/green]")

    # Fetch info for all models
    models = []
    for model_id in model_ids:
        console.print(f"  Fetching info: {model_id}")
        try:
            info = fetch_model_info(model_id)
            models.append(info)
        except Exception:
            console.print(f"  [yellow]Skipping {model_id} (failed to fetch)[/yellow]")
            continue

    if not models:
        console.print("[red]No valid models found.[/red]")
        raise typer.Exit(1)

    # Display model info
    table = Table(title="Models Found")
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="green", justify="right")
    table.add_column("Context", style="blue", justify="right")
    table.add_column("Type", style="magenta")
    table.add_column("Downloads", justify="right")

    for m in sorted(models, key=lambda x: x.size_billions):
        model_type = (
            "Instruct" if m.is_instruct else ("Reasoning" if m.is_reasoning else "Base")
        )
        table.add_row(
            m.model_name,
            f"{m.size_billions:.1f}B",
            f"{m.context_length:,}",
            model_type,
            f"{m.downloads:,}",
        )

    console.print(table)

    # Select best variants and config types
    configs = select_best_variants(models, max_configs)

    console.print(f"\n[blue]Generating {len(configs)} config(s)...[/blue]")

    # Determine output directory
    org_name = models[0].model_id.split("/")[0].lower().replace("-", "_")
    if output_dir is None:
        output_dir = Path("configs/recipes") / org_name

    # Generate configs
    generated_files = []
    for model, config_type in configs:
        size_label = (
            f"{model.size_billions:.0f}b" if model.size_billions >= 1 else "small"
        )
        config_dir = output_dir / "sft" / f"{size_label}_{config_type}"

        filename, content = generate_config_yaml(model, config_type, config_dir)

        if dry_run:
            console.print(f"  [dim]Would create:[/dim] {config_dir / filename}")
            console.print(
                Panel(content, title=str(config_dir / filename), expand=False)
            )
        else:
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / filename
            config_path.write_text(content)
            generated_files.append(config_path)
            console.print(f"  [green]Created:[/green] {config_path}")

    # Generate README
    readme_content = generate_readme(models, configs, output_dir, collection_url)
    readme_path = output_dir / "README.md"

    if dry_run:
        console.print(f"  [dim]Would create:[/dim] {readme_path}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        readme_path.write_text(readme_content)
        generated_files.append(readme_path)
        console.print(f"  [green]Created:[/green] {readme_path}")

    # Summary
    console.print()
    if dry_run:
        console.print("[yellow]Dry run complete. No files were written.[/yellow]")
    else:
        console.print(
            f"[green]Bootstrap complete! Generated {len(generated_files)} files.[/green]"
        )
        console.print("\nNext steps:")
        console.print(f"  1. Review the generated configs in {output_dir}")
        console.print("  2. Customize the dataset and hyperparameters as needed")
        console.print(f"  3. Run training with the commands in {readme_path}")
