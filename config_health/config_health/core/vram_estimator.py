"""VRAM estimation for training configs — no GPU needed.

Estimates GPU memory requirements from model architecture (via HF AutoConfig)
and training parameters (batch size, dtype, LoRA, FSDP, gradient checkpointing).

Two estimates per config:
  - **as-configured**: VRAM with current settings
  - **minimal**: VRAM with batch_size=1, gradient checkpointing on, shortest seq_len

Memory components:
  - Model weights (dtype-dependent)
  - Optimizer states (Adam: 8 bytes/trainable param for momentum + variance)
  - Gradients (same dtype as trainable params)
  - Activations (depends on batch_size, seq_len, num_layers)
  - CUDA overhead (~500 MB constant)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from config_health.core.models import REMOTE_ENGINES, ConfigEntry, ConfigType

_GB = 1024**3
_CUDA_OVERHEAD_GB = 0.5  # ~500 MB for CUDA context, kernels, etc.

# Bytes per parameter for each dtype
_DTYPE_BYTES: dict[str, float] = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "fp4": 0.5,
    "nf4": 0.5,
    "mxfp4": 0.5,
}


@dataclass
class VRAMEstimate:
    """VRAM breakdown for a training config."""

    # Parameter counts
    total_params: int = 0
    trainable_params: int = 0

    # Memory breakdown (GB)
    model_memory_gb: float = 0.0
    optimizer_memory_gb: float = 0.0
    gradient_memory_gb: float = 0.0
    activation_memory_gb: float = 0.0
    overhead_gb: float = _CUDA_OVERHEAD_GB
    total_vram_gb: float = 0.0

    # Minimal setup
    minimal_activation_gb: float = 0.0
    minimal_total_vram_gb: float = 0.0

    # Config context
    batch_size: int = 1
    seq_len: int = 2048
    dtype_bytes: float = 2.0
    gradient_checkpointing: bool = False
    is_peft: bool = False
    peft_rank: int = 0
    is_quantized: bool = False
    quant_bytes: float = 2.0
    num_gpus: int = 1

    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def total_params_b(self) -> float:
        return self.total_params / 1e9

    @property
    def trainable_params_b(self) -> float:
        return self.trainable_params / 1e9

    def summary(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        parts = [f"{self.total_vram_gb:.1f} GB"]
        if self.minimal_total_vram_gb != self.total_vram_gb:
            parts.append(f"(min: {self.minimal_total_vram_gb:.1f} GB)")
        parts.append(f"[{self.total_params_b:.1f}B params]")
        return " ".join(parts)


def estimate_vram(entry: ConfigEntry) -> VRAMEstimate:
    """Estimate VRAM requirements for a config. No GPU needed."""
    if entry.config_type != ConfigType.TRAINING:
        return VRAMEstimate(error="Not a training config")

    data = _load_yaml(entry.abs_path)
    if data is None:
        return VRAMEstimate(error="Could not load YAML")

    if not entry.model_name:
        return VRAMEstimate(error="No model_name")

    # Skip remote engines
    if entry.engine and entry.engine in REMOTE_ENGINES:
        return VRAMEstimate(error="Remote engine")

    # Get model architecture info from HF config
    arch = _get_model_arch(entry.model_name)
    if arch is None:
        return VRAMEstimate(error=f"Cannot load HF config: {entry.model_name}")

    # Extract training params from YAML
    training = data.get("training", {}) or {}
    model_cfg = data.get("model", {}) or {}
    peft_cfg = data.get("peft", {}) or {}
    fsdp_cfg = data.get("fsdp", {}) or {}
    ds_cfg = data.get("deepspeed", {}) or {}

    # Dtype
    dtype_str = model_cfg.get("torch_dtype_str", "bfloat16")
    dtype_bytes = _DTYPE_BYTES.get(dtype_str, 2.0)

    # Batch size and sequence length
    batch_size = training.get("per_device_train_batch_size", 1)
    if not isinstance(batch_size, int) or batch_size < 1:
        batch_size = 1
    seq_len = model_cfg.get("model_max_length", 0)
    if not isinstance(seq_len, int) or seq_len < 1:
        # Fall back to model's native context window from AutoConfig
        seq_len = arch.max_position_embeddings if arch.max_position_embeddings > 0 else 2048

    # Detect attention implementation for activation memory estimation
    attn_impl = model_cfg.get("attn_implementation", "")
    model_kwargs = model_cfg.get("model_kwargs", {}) or {}
    if not attn_impl:
        attn_impl = model_kwargs.get("attn_implementation", "")

    # LoRA / PEFT
    is_peft = bool(training.get("use_peft", False))
    peft_rank = peft_cfg.get("lora_r", 16) if is_peft else 0
    q_lora = peft_cfg.get("q_lora", False) if is_peft else False

    # Quantization
    is_quantized = q_lora
    quant_bytes = dtype_bytes
    quant_config = model_kwargs.get("quantization_config", {}) or {}
    quant_method = quant_config.get("quant_method", "")
    if quant_method in ("bnb", "bitsandbytes", "awq", "gptq"):
        is_quantized = True
        load_in_4bit = quant_config.get("load_in_4bit", quant_config.get("bits") == 4)
        quant_bytes = 0.5 if load_in_4bit else 1.0
    elif quant_method in ("mxfp4", "fp4", "nf4"):
        is_quantized = True
        quant_bytes = 0.5

    # Gradient checkpointing
    grad_ckpt = bool(training.get("enable_gradient_checkpointing", False))

    # FSDP / multi-GPU
    fsdp_enabled = bool(fsdp_cfg.get("enable_fsdp", False)) if isinstance(fsdp_cfg, dict) else False
    ds_enabled = bool(ds_cfg.get("enable_deepspeed", False)) if isinstance(ds_cfg, dict) else False
    num_gpus = 1
    if fsdp_enabled or ds_enabled:
        # Infer from job config or default to 2
        num_gpus = _infer_num_gpus(data, entry)

    # --- Compute estimates ---
    est = VRAMEstimate(
        total_params=arch.total_params,
        batch_size=batch_size,
        seq_len=seq_len,
        dtype_bytes=dtype_bytes,
        gradient_checkpointing=grad_ckpt,
        is_peft=is_peft,
        peft_rank=peft_rank,
        is_quantized=is_quantized,
        quant_bytes=quant_bytes,
        num_gpus=num_gpus,
    )

    # Model memory
    if is_quantized:
        est.model_memory_gb = (arch.total_params * quant_bytes) / _GB
        est.notes.append(f"Quantized: {quant_bytes * 8:.0f}-bit")
    else:
        est.model_memory_gb = (arch.total_params * dtype_bytes) / _GB

    # Trainable parameters
    if is_peft:
        # LoRA: rank * (in + out) for each target module
        target_modules = peft_cfg.get("lora_target_modules", [])
        est.trainable_params = _estimate_lora_params(
            arch, peft_rank, target_modules
        )
        est.notes.append(f"LoRA r={peft_rank}, {len(target_modules)} targets")
    else:
        est.trainable_params = arch.total_params
        est.notes.append("Full finetune")

    # Optimizer memory (Adam: fp32 momentum + fp32 variance = 8 bytes/param)
    est.optimizer_memory_gb = (est.trainable_params * 8) / _GB

    # Gradient memory
    est.gradient_memory_gb = (est.trainable_params * dtype_bytes) / _GB

    # Activation memory
    est.activation_memory_gb = _estimate_activation_memory(
        arch, batch_size, seq_len, dtype_bytes, grad_ckpt, attn_impl
    )

    # Minimal activation memory (batch_size=1, grad checkpointing on)
    est.minimal_activation_gb = _estimate_activation_memory(
        arch, 1, min(seq_len, 512), dtype_bytes, True, attn_impl
    )

    # FSDP sharding
    if num_gpus > 1:
        est.notes.append(f"FSDP/DS across {num_gpus} GPUs")
        # Sharding divides model, optimizer, and gradient memory
        per_gpu_factor = 1.0 / num_gpus
        est.model_memory_gb *= per_gpu_factor
        est.optimizer_memory_gb *= per_gpu_factor
        est.gradient_memory_gb *= per_gpu_factor
        # Add gather overhead: ~1 layer's parameters temporarily
        layer_params = arch.total_params / max(arch.num_layers, 1)
        gather_overhead = (layer_params * dtype_bytes * 2) / _GB  # forward + backward
        est.activation_memory_gb += gather_overhead
        est.minimal_activation_gb += gather_overhead

    # Totals
    est.total_vram_gb = (
        est.model_memory_gb
        + est.optimizer_memory_gb
        + est.gradient_memory_gb
        + est.activation_memory_gb
        + est.overhead_gb
    )

    est.minimal_total_vram_gb = (
        est.model_memory_gb
        + est.optimizer_memory_gb
        + est.gradient_memory_gb
        + est.minimal_activation_gb
        + est.overhead_gb
    )

    # Round
    est.total_vram_gb = round(est.total_vram_gb, 1)
    est.minimal_total_vram_gb = round(est.minimal_total_vram_gb, 1)

    return est


# ── Architecture extraction ─────────────────────────────────────────


@dataclass
class _ModelArch:
    """Key architecture dimensions extracted from HF config."""

    total_params: int
    hidden_size: int
    num_layers: int
    intermediate_size: int
    vocab_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    max_position_embeddings: int = 0  # native context window from HF config
    is_moe: bool = False
    num_experts: int = 1
    experts_per_tok: int = 1


def _get_model_arch(model_name: str) -> _ModelArch | None:
    """Load HF config and extract architecture dimensions."""
    try:
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Handle composite configs (e.g., Llama 4 VLM)
        if hasattr(config, "text_config") and config.text_config is not None:
            config = config.text_config

        hidden = getattr(config, "hidden_size", 0) or 0
        layers = getattr(config, "num_hidden_layers", 0) or 0
        inter = getattr(config, "intermediate_size", 0) or 0
        vocab = getattr(config, "vocab_size", 0) or 0
        heads = getattr(config, "num_attention_heads", 0) or 0
        kv_heads = getattr(config, "num_key_value_heads", None) or heads
        head_dim = hidden // heads if heads else 0
        max_pos_emb = getattr(config, "max_position_embeddings", 0) or 0

        # MoE detection
        num_experts = getattr(config, "num_local_experts", None) or getattr(config, "num_experts", None) or 1
        experts_per_tok = getattr(config, "num_experts_per_tok", None) or 1
        is_moe = num_experts > 1

        if not all([hidden, layers, vocab]):
            return None

        # Estimate total parameters
        total_params = _estimate_total_params(
            hidden, layers, inter, vocab, heads, kv_heads, head_dim, num_experts
        )

        return _ModelArch(
            total_params=total_params,
            hidden_size=hidden,
            num_layers=layers,
            intermediate_size=inter,
            vocab_size=vocab,
            num_attention_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_pos_emb,
            is_moe=is_moe,
            num_experts=num_experts,
            experts_per_tok=experts_per_tok,
        )
    except Exception:
        return None


def _estimate_total_params(
    hidden: int,
    layers: int,
    inter: int,
    vocab: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    num_experts: int,
) -> int:
    """Estimate total parameters from architecture dimensions."""
    # Embedding (LM head is typically tied to embeddings — shared tensor, counted once)
    embedding = vocab * hidden

    # Per-layer attention: Q + K + V + O projections
    # Q: hidden -> hidden, K: hidden -> kv_dim, V: hidden -> kv_dim, O: hidden -> hidden
    kv_dim = kv_heads * head_dim if (kv_heads and head_dim) else hidden
    attn_per_layer = (
        hidden * hidden  # Q
        + hidden * kv_dim  # K
        + hidden * kv_dim  # V
        + hidden * hidden  # O
    )

    # Per-layer MLP (LLaMA-style: gate + up + down)
    if inter:
        mlp_per_layer = 3 * hidden * inter  # gate_proj, up_proj, down_proj
    else:
        mlp_per_layer = 4 * hidden * hidden  # fallback: standard 4x MLP

    # For MoE: multiply MLP by num_experts, add router
    if num_experts > 1:
        mlp_per_layer = mlp_per_layer * num_experts + hidden * num_experts  # router

    # Layer norms (2 per layer: pre-attn, pre-mlp)
    norm_per_layer = 2 * hidden

    total = (
        embedding  # token embeddings (LM head typically tied, shares this tensor)
        + layers * (attn_per_layer + mlp_per_layer + norm_per_layer)
        + hidden  # final layer norm
    )
    return total


def _estimate_lora_params(
    arch: _ModelArch, rank: int, target_modules: list[str]
) -> int:
    """Estimate LoRA trainable parameters."""
    if not target_modules or rank <= 0:
        return 0

    # Map module names to their dimensions
    kv_dim = (arch.num_kv_heads or arch.num_attention_heads) * (arch.head_dim or 0) if arch.head_dim else arch.hidden_size
    module_dims: dict[str, tuple[int, int]] = {
        "q_proj": (arch.hidden_size, arch.hidden_size),
        "k_proj": (arch.hidden_size, kv_dim),
        "v_proj": (arch.hidden_size, kv_dim),
        "o_proj": (arch.hidden_size, arch.hidden_size),
        "gate_proj": (arch.hidden_size, arch.intermediate_size),
        "up_proj": (arch.hidden_size, arch.intermediate_size),
        "down_proj": (arch.intermediate_size, arch.hidden_size),
        # Common alternative names
        "qkv_proj": (arch.hidden_size, arch.hidden_size + 2 * kv_dim),
        "out_proj": (arch.hidden_size, arch.hidden_size),
        "fc1": (arch.hidden_size, arch.intermediate_size),
        "fc2": (arch.intermediate_size, arch.hidden_size),
        "dense": (arch.hidden_size, arch.hidden_size),
    }

    total = 0
    for name in target_modules:
        if name == "all-linear":
            # Approximate: all projection layers
            for _, (in_f, out_f) in module_dims.items():
                total += rank * (in_f + out_f) * arch.num_layers
            return total

        in_f, out_f = module_dims.get(name, (arch.hidden_size, arch.hidden_size))
        total += rank * (in_f + out_f) * arch.num_layers

    return total


def _estimate_activation_memory(
    arch: _ModelArch,
    batch_size: int,
    seq_len: int,
    dtype_bytes: float,
    gradient_checkpointing: bool,
    attn_implementation: str = "",
) -> float:
    """Estimate activation memory in GB.

    Uses the literature-based formula: batch * seq * hidden * 34 * dtype per layer,
    which accounts for all intermediate tensors across attention, MLP, norms, and
    residuals. Flash Attention / SDPA reduces the attention component from O(seq^2)
    to O(seq), saving 10-20% for long sequences.

    References:
    - https://www.propelrc.com/llm-vram-calculator/
    - https://modal.com/blog/how-much-vram-need-fine-tuning
    """
    uses_efficient_attn = attn_implementation in ("sdpa", "flash_attention_2")

    if uses_efficient_attn:
        # Flash/SDPA: attention memory is O(seq_len), not O(seq_len^2)
        # Per-layer: batch * seq * hidden * 34 (literature formula) minus the
        # quadratic attention component, replaced with linear
        per_layer = batch_size * seq_len * arch.hidden_size * 34 * dtype_bytes
    else:
        # Naive attention: includes O(seq^2) attention scores
        # Use the component-based formula with quadratic attention
        attn_scores = batch_size * arch.num_attention_heads * seq_len * seq_len * dtype_bytes
        linear_components = batch_size * seq_len * arch.hidden_size * 34 * dtype_bytes
        # The factor-34 formula already includes a linear attention term;
        # add the extra quadratic cost on top if not using efficient attention
        kv_dim = (arch.num_kv_heads or arch.num_attention_heads) * (arch.head_dim or 0) if arch.head_dim else arch.hidden_size
        linear_attn_in_formula = batch_size * seq_len * (arch.hidden_size + 2 * kv_dim) * dtype_bytes
        per_layer = linear_components + attn_scores - linear_attn_in_formula

    if gradient_checkpointing:
        # Literature: ~35% reduction in activation memory with gradient checkpointing
        effective_layers = arch.num_layers * 0.65
    else:
        effective_layers = arch.num_layers

    return (per_layer * effective_layers) / _GB


def _infer_num_gpus(data: dict, entry: ConfigEntry) -> int:
    """Try to infer GPU count from config or filename."""
    # Check resources.accelerators in job-like configs
    resources = data.get("resources", {})
    if isinstance(resources, dict):
        accel = resources.get("accelerators", "")
        if isinstance(accel, str) and ":" in accel:
            try:
                return int(accel.split(":")[-1])
            except ValueError:
                pass

    # Filename hints
    path = entry.path.lower()
    for n in (8, 4, 2):
        if f"{n}gpu" in path or f"{n}_gpu" in path or f"{n}x" in path:
            return n

    # Default for FSDP/DeepSpeed configs
    return 2


def _load_yaml(path: str) -> dict | None:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None
