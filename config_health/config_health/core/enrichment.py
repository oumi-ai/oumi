"""Metadata enrichment — add model info and complexity scores to config entries."""

from __future__ import annotations

import re

from config_health.core.models import (
    REMOTE_ENGINES,
    ConfigEntry,
    ConfigType,
    ModelMetadata,
)
from config_health.core.scanner import load_yaml_cached


def enrich_entries(entries: list[ConfigEntry]) -> None:
    """Enrich config entries with model metadata and complexity scores.

    Modifies entries in-place. Model metadata requires network access for
    HF model configs (small downloads, cached per-run).
    """
    for entry in entries:
        entry.complexity = _compute_complexity(entry)
        if entry.model_name and not entry.model_meta:
            entry.model_meta = _resolve_model_metadata(entry.model_name, entry.engine)


# ── Complexity score ──────────────────────────────────────────────


_COMPLEXITY_FEATURES = [
    ("peft", lambda d: bool((d.get("training") or {}).get("use_peft"))),
    ("fsdp", lambda d: bool((d.get("fsdp") or {}).get("enable_fsdp"))),
    ("deepspeed", lambda d: bool((d.get("deepspeed") or {}).get("enable_deepspeed"))),
    ("quantization", lambda d: bool(
        (d.get("peft") or {}).get("q_lora")
        or ((d.get("model") or {}).get("model_kwargs") or {}).get("quantization_config")
    )),
    ("grad_checkpointing", lambda d: bool((d.get("training") or {}).get("enable_gradient_checkpointing"))),
    ("mixed_precision", lambda d: bool((d.get("training") or {}).get("mixed_precision_dtype"))),
    ("flash_attention", lambda d: (d.get("model") or {}).get("attn_implementation") in ("sdpa", "flash_attention_2")),
    ("compile", lambda d: bool((d.get("training") or {}).get("compile"))),
]


def _compute_complexity(entry: ConfigEntry) -> int:
    """Count the number of advanced features enabled in a config."""
    data = load_yaml_cached(entry.abs_path)
    if not data or not isinstance(data, dict):
        return 0
    return sum(1 for _, check in _COMPLEXITY_FEATURES if check(data))


# ── Model metadata ────────────────────────────────────────────────


_metadata_cache: dict[str, ModelMetadata | None] = {}


def _resolve_model_metadata(
    model_name: str, engine: str | None
) -> ModelMetadata | None:
    """Resolve model metadata from HF. Cached per model_name."""
    if model_name in _metadata_cache:
        return _metadata_cache[model_name]

    # Skip remote engines and local paths
    if engine and engine in REMOTE_ENGINES:
        _metadata_cache[model_name] = None
        return None
    if model_name.startswith(("./", "/", "~", "output/", "checkpoint")):
        _metadata_cache[model_name] = None
        return None
    if "/" not in model_name:
        # Try to extract size from name like "gpt2" or "smollm-135m"
        meta = ModelMetadata()
        size = _extract_size_from_name(model_name)
        if size:
            meta.params_b = size
        _metadata_cache[model_name] = meta
        return meta

    try:
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Get model_type (top-level, not text_config)
        model_type = getattr(config, "model_type", "")
        is_vlm = hasattr(config, "text_config") and config.text_config is not None

        # Get the inner config for parameter counting
        inner = config.text_config if is_vlm and config.text_config else config

        hidden = getattr(inner, "hidden_size", 0) or 0
        layers = getattr(inner, "num_hidden_layers", 0) or 0
        inter = getattr(inner, "intermediate_size", 0) or 0
        vocab = getattr(inner, "vocab_size", 0) or 0
        num_experts = getattr(inner, "num_local_experts", None) or getattr(inner, "num_experts", None) or 1

        # Estimate params
        params = 0
        if hidden and layers and vocab:
            embedding = vocab * hidden
            heads = getattr(inner, "num_attention_heads", 0) or 0
            kv_heads = getattr(inner, "num_key_value_heads", None) or heads
            head_dim = hidden // heads if heads else 0
            kv_dim = kv_heads * head_dim if (kv_heads and head_dim) else hidden

            attn_per_layer = hidden * hidden + hidden * kv_dim * 2 + hidden * hidden
            mlp_per_layer = 3 * hidden * inter if inter else 4 * hidden * hidden
            if num_experts > 1:
                mlp_per_layer = mlp_per_layer * num_experts + hidden * num_experts
            norm_per_layer = 2 * hidden

            params = embedding + layers * (attn_per_layer + mlp_per_layer + norm_per_layer) + hidden

        # Try to get architecture class name
        arch_name = type(config).__name__.replace("Config", "")

        meta = ModelMetadata(
            params_b=round(params / 1e9, 2) if params else 0.0,
            architecture=arch_name,
            model_type=model_type,
            is_vlm=is_vlm,
        )

        _metadata_cache[model_name] = meta
        return meta
    except Exception:
        # Fallback: try to extract size from name
        meta = ModelMetadata()
        size = _extract_size_from_name(model_name)
        if size:
            meta.params_b = size
        _metadata_cache[model_name] = meta
        return meta


def _extract_size_from_name(name: str) -> float:
    """Extract model size in billions from a name like 'Llama-3.1-8B' or '135M'."""
    # Match patterns like "8B", "70B", "0.5B", "135M"
    m = re.search(r"(\d+\.?\d*)\s*[Bb]\b", name)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*[Mm]\b", name)
    if m:
        return round(float(m.group(1)) / 1000, 2)
    return 0.0


def clear_metadata_cache() -> None:
    """Clear the model metadata cache."""
    _metadata_cache.clear()
    _hub_siblings_cache.clear()


# ── HF Hub sibling discovery ──────────────────────────────────────

_hub_siblings_cache: dict[str, list[tuple[str, float]]] = {}


def discover_hub_sizes(model_name: str) -> list[tuple[str, float]]:
    """Find all official size variants of a model on HuggingFace Hub.

    Given "meta-llama/Llama-3.1-8B-Instruct", returns:
      [("meta-llama/Llama-3.1-8B", 8.0), ("meta-llama/Llama-3.1-70B", 70.0), ...]

    Cached per model_name (deduplicates across configs sharing the same base).
    """
    if model_name in _hub_siblings_cache:
        return _hub_siblings_cache[model_name]

    if "/" not in model_name:
        _hub_siblings_cache[model_name] = []
        return []

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        org = model_name.split("/")[0]
        name = model_name.split("/")[1]

        # Extract base name: "Llama-3.1" from "Llama-3.1-8B-Instruct"
        base = re.sub(r"[-_]\d+\.?\d*[BbMm].*$", "", name)
        if not base or len(base) < 3:
            _hub_siblings_cache[model_name] = []
            return []

        results = api.list_models(author=org, search=base, limit=50)
        siblings: list[tuple[str, float]] = []
        for m in results:
            if not m.id.startswith(org + "/"):
                continue
            m_name = m.id.split("/")[1]
            # Must share the exact base prefix (avoid Qwen3 matching Qwen3.5)
            m_base = re.sub(r"[-_]\d+\.?\d*[BbMm].*$", "", m_name)
            if m_base != base:
                continue
            size_match = re.search(r"(\d+\.?\d*)\s*[Bb]\b", m.id)
            if size_match:
                size_b = float(size_match.group(1))
                siblings.append((m.id, size_b))

        # Deduplicate by size (keep shortest name per size)
        seen: dict[float, str] = {}
        for mid, size in sorted(siblings, key=lambda x: len(x[0])):
            if size not in seen:
                seen[size] = mid
        result = sorted([(mid, s) for s, mid in seen.items()], key=lambda x: x[1])
        _hub_siblings_cache[model_name] = result
        return result
    except Exception:
        _hub_siblings_cache[model_name] = []
        return []
