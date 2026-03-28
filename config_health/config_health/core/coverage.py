"""Coverage analysis — find gaps in model × config-type matrix."""

from __future__ import annotations

from config_health.core.models import (
    ConfigEntry,
    ConfigType,
    CoverageGap,
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
