"""C3: Auto-fix known config issues.

Fixes:
- Wrong LoRA target modules (uses architecture-detected correct names)
- FSDP transformer_layer_cls using FQN format (suggests short class name)
"""

from __future__ import annotations

import re

import yaml

from config_health.core.models import CheckResult, CheckStatus, HealthReport


def apply_fixes(report: HealthReport, repo_root: str) -> int:
    """Apply auto-fixes for known issues. Returns number of fixes applied."""
    fix_count = 0

    # Collect fixable failures
    lora_failures: dict[str, CheckResult] = {}
    fsdp_failures: dict[str, CheckResult] = {}

    for result in report.check_results:
        if result.status != CheckStatus.FAIL:
            continue
        if result.check_name == "lora_targets" and result.details:
            lora_failures[result.config_path] = result
        elif result.check_name == "fsdp_layer_cls":
            fsdp_failures[result.config_path] = result

    # Fix wrong LoRA targets
    for path, result in lora_failures.items():
        entry = next((e for e in report.entries if e.path == path), None)
        if entry and _fix_lora_targets(entry.abs_path, result):
            fix_count += 1

    # Fix FSDP layer class FQN -> short name
    for path, result in fsdp_failures.items():
        entry = next((e for e in report.entries if e.path == path), None)
        if entry and _fix_fsdp_layer_cls(entry.abs_path, result):
            fix_count += 1

    return fix_count


def _parse_list_from_text(text: str, prefix: str) -> list[str]:
    """Parse a list of strings from text like "Available: ['a', 'b']"."""
    if prefix not in text:
        return []
    part = text.split(prefix)[1].strip()
    # Parse manually: extract quoted strings from bracket-list format
    return re.findall(r"'([^']+)'", part)


def _fix_lora_targets(abs_path: str, result: CheckResult) -> bool:
    """Replace wrong LoRA target modules with available ones from the model."""
    try:
        with open(abs_path) as f:
            content = f.read()
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            return False

        peft = data.get("peft", {})
        if not isinstance(peft, dict):
            return False

        targets = peft.get("lora_target_modules", [])
        if not targets:
            return False

        available = _parse_list_from_text(result.details or "", "Available:")
        if not available:
            return False

        # Map common wrong names to correct ones
        standard_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        correct_targets = [t for t in standard_targets if t in available]

        if not correct_targets:
            return False

        # Only fix if the current targets are clearly wrong
        wrong = [t for t in targets if t not in available]
        if not wrong:
            return False

        peft["lora_target_modules"] = correct_targets
        data["peft"] = peft

        with open(abs_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        return True
    except Exception:
        return False


def _fix_fsdp_layer_cls(abs_path: str, result: CheckResult) -> bool:
    """Fix FSDP transformer_layer_cls: replace with detected class from arch."""
    try:
        with open(abs_path) as f:
            content = f.read()
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            return False

        fsdp = data.get("fsdp", {})
        if not isinstance(fsdp, dict):
            return False

        layer_cls = fsdp.get("transformer_layer_cls", "")
        if not layer_cls:
            return False

        detected = _parse_list_from_text(result.message, "Detected:")
        if not detected:
            return False

        new_cls = ",".join(detected)
        fsdp["transformer_layer_cls"] = new_cls
        data["fsdp"] = fsdp

        with open(abs_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        return True
    except Exception:
        return False
