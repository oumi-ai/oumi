"""Coverage analysis — find gaps in model × config-type matrix."""

from __future__ import annotations

from config_health.core.models import (
    ArchCoverageEntry,
    ConfigEntry,
    ConfigType,
    CoverageGap,
    REMOTE_ENGINES,
)

# Expected config types per category
_RECIPE_EXPECTED = {"training", "inference", "evaluation"}
_API_EXPECTED = {"inference", "evaluation"}


def analyze_coverage(entries: list[ConfigEntry]) -> list[CoverageGap]:
    """Find model families missing expected config types."""
    # Group by (category, model_family)
    family_types: dict[tuple[str, str], set[str]] = {}
    for entry in entries:
        if not entry.model_family:
            continue
        if entry.config_type == ConfigType.UNKNOWN:
            continue
        # Normalize: treat job configs separately, they're cross-cutting
        if entry.config_type == ConfigType.JOB:
            continue
        key = (entry.category, entry.model_family)
        family_types.setdefault(key, set()).add(entry.config_type.value)

    gaps: list[CoverageGap] = []
    for (category, family), existing in sorted(family_types.items()):
        expected = _get_expected_types(category)
        if not expected:
            continue
        missing = expected - existing
        if missing:
            gaps.append(
                CoverageGap(
                    model_family=family,
                    missing_types=sorted(missing),
                    existing_types=sorted(existing),
                    category=category,
                )
            )

    return gaps


def build_coverage_matrix(
    entries: list[ConfigEntry],
) -> dict[str, dict[str, list[ConfigEntry]]]:
    """Build model_family -> config_type -> [entries] matrix.

    Only includes recipe and API configs.
    """
    matrix: dict[str, dict[str, list[ConfigEntry]]] = {}
    for entry in entries:
        if not entry.model_family:
            continue
        if entry.category not in ("recipes", "apis"):
            continue
        family = entry.model_family
        ctype = entry.config_type.value
        matrix.setdefault(family, {}).setdefault(ctype, []).append(entry)
    return matrix


def _get_expected_types(category: str) -> set[str]:
    if category == "recipes":
        return _RECIPE_EXPECTED
    if category == "apis":
        return _API_EXPECTED
    return set()


# ── Architecture coverage ──────────────────────────────────────────


# Cache: model_name -> model_type (persists across calls within a run)
_model_type_cache: dict[str, str | None] = {}


def clear_model_type_cache() -> None:
    """Clear the model_type resolution cache. Call on rescan."""
    _model_type_cache.clear()


def _get_oumi_registry() -> dict[str, dict]:
    """Get Oumi's internal model registry: model_type -> {model_class, tested, is_vlm}.

    Returns empty dict if oumi is not installed.
    """
    try:
        from oumi.core.configs.internal.supported_models import get_all_models_map

        result: dict[str, dict] = {}
        for model_type, info in get_all_models_map().items():
            is_vlm = (
                hasattr(info.config, "visual_config")
                and info.config.visual_config is not None
            )
            result[model_type] = {
                "model_class": info.model_class.__name__,
                "tested": getattr(info, "tested", False),
                "is_vlm": is_vlm,
            }
        return result
    except Exception:
        return {}


def get_supported_architectures() -> dict[str, dict]:
    """Get all supported model_type values from both transformers and Oumi.

    Merges:
    - transformers.AutoModelForCausalLM (137 causal LM architectures)
    - transformers.AutoModelForVision2Seq / AutoModelForImageTextToText (VLMs)
    - Oumi's internal registry (adds tested status, VLM flags, correct model class)

    Returns dict: model_type -> {model_class, is_vlm, in_oumi, oumi_tested}
    """
    result: dict[str, dict] = {}

    try:
        import transformers

        # Causal LM architectures
        for config_cls, model_cls in transformers.AutoModelForCausalLM._model_mapping.items():
            model_type = getattr(config_cls, "model_type", None)
            if model_type:
                result[model_type] = {
                    "model_class": model_cls.__name__,
                    "is_vlm": False,
                    "in_oumi": False,
                    "oumi_tested": False,
                }

        # Vision2Seq / ImageTextToText architectures
        for auto_cls_name in ("AutoModelForVision2Seq", "AutoModelForImageTextToText"):
            auto_cls = getattr(transformers, auto_cls_name, None)
            if auto_cls and hasattr(auto_cls, "_model_mapping"):
                for config_cls, model_cls in auto_cls._model_mapping.items():
                    model_type = getattr(config_cls, "model_type", None)
                    if model_type and model_type not in result:
                        result[model_type] = {
                            "model_class": model_cls.__name__,
                            "is_vlm": True,
                            "in_oumi": False,
                            "oumi_tested": False,
                        }
    except Exception:
        pass

    # Overlay Oumi registry info
    oumi_reg = _get_oumi_registry()
    for model_type, oumi_info in oumi_reg.items():
        if model_type in result:
            result[model_type]["in_oumi"] = True
            result[model_type]["oumi_tested"] = oumi_info["tested"]
            if oumi_info["is_vlm"]:
                result[model_type]["is_vlm"] = True
            # Use Oumi's model class if it differs (Oumi knows the correct one)
            result[model_type]["model_class"] = oumi_info["model_class"]
        else:
            # Oumi-only model type (not in standard transformers mappings)
            result[model_type] = {
                "model_class": oumi_info["model_class"],
                "is_vlm": oumi_info["is_vlm"],
                "in_oumi": True,
                "oumi_tested": oumi_info["tested"],
            }

    return dict(sorted(result.items()))


def resolve_model_type(model_name: str) -> str | None:
    """Resolve an HF model name to its model_type. Cached per-run.

    Returns the top-level model_type (not text_config's type for VLMs),
    since Oumi's registry uses the top-level type.
    """
    if model_name in _model_type_cache:
        return _model_type_cache[model_name]

    # Skip local paths, GGUF, and names without a slash (not HF repo IDs)
    if (
        model_name.startswith(("./", "/", "~", "output/", "checkpoint"))
        or "gguf" in model_name.lower()
        or "/" not in model_name
    ):
        _model_type_cache[model_name] = None
        return None

    try:
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
        model_type = getattr(config, "model_type", None)
    except Exception:
        model_type = None

    _model_type_cache[model_name] = model_type
    return model_type


def build_arch_coverage(
    entries: list[ConfigEntry],
) -> tuple[list[ArchCoverageEntry], list[ArchCoverageEntry]]:
    """Build architecture-level coverage: covered vs uncovered model_types.

    Merges transformers + Oumi registry to show:
    - Which architectures have configs (covered)
    - Which are supported but have no configs (uncovered)
    - VLM vs LLM distinction
    - Oumi registry status (registered, tested)
    """
    all_archs = get_supported_architectures()

    # Resolve entries to model_type and group
    arch_data: dict[str, dict] = {}
    for entry in entries:
        if not entry.model_name:
            continue
        if entry.engine and entry.engine in REMOTE_ENGINES:
            continue
        if entry.model_name.startswith(("./", "/", "~")):
            continue
        if entry.config_type in (ConfigType.UNKNOWN, ConfigType.JOB):
            continue

        model_type = resolve_model_type(entry.model_name)
        if not model_type:
            continue

        if model_type not in arch_data:
            arch_data[model_type] = {
                "config_types": set(),
                "model_names": set(),
                "count": 0,
            }
        arch_data[model_type]["config_types"].add(entry.config_type.value)
        arch_data[model_type]["model_names"].add(entry.model_name)
        arch_data[model_type]["count"] += 1

    covered: list[ArchCoverageEntry] = []
    uncovered: list[ArchCoverageEntry] = []

    for model_type, arch_info in all_archs.items():
        data = arch_data.get(model_type)
        entry = ArchCoverageEntry(
            model_type=model_type,
            model_class=arch_info["model_class"],
            is_vlm=arch_info["is_vlm"],
            in_oumi_registry=arch_info["in_oumi"],
            oumi_tested=arch_info["oumi_tested"],
        )
        if data:
            entry.config_types = sorted(data["config_types"])
            entry.model_names = sorted(data["model_names"])
            entry.config_count = data["count"]
            covered.append(entry)
        else:
            uncovered.append(entry)

    # Also add any model_types that appeared in configs but aren't in the
    # transformers/oumi registry (custom or very new models)
    for model_type, data in arch_data.items():
        if model_type not in all_archs:
            covered.append(ArchCoverageEntry(
                model_type=model_type,
                model_class="(unknown)",
                config_types=sorted(data["config_types"]),
                model_names=sorted(data["model_names"]),
                config_count=data["count"],
            ))

    covered.sort(key=lambda e: -e.config_count)
    return covered, uncovered
