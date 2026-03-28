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
    """Replace wrong LoRA target modules with available ones from the model.

    Uses string replacement to preserve YAML formatting and comments.
    """
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

        # Use string replacement to preserve formatting
        new_content = _replace_yaml_list(
            content, "lora_target_modules", correct_targets
        )
        if new_content == content:
            return False

        with open(abs_path, "w") as f:
            f.write(new_content)

        return True
    except Exception:
        return False


def _fix_fsdp_layer_cls(abs_path: str, result: CheckResult) -> bool:
    """Fix FSDP transformer_layer_cls: replace with detected class from arch.

    Uses string replacement to preserve YAML formatting and comments.
    """
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

        # Use string replacement to preserve formatting
        new_content = _replace_yaml_value(content, "transformer_layer_cls", new_cls)
        if new_content == content:
            return False

        with open(abs_path, "w") as f:
            f.write(new_content)

        return True
    except Exception:
        return False


def _replace_yaml_value(content: str, key: str, new_value: str) -> str:
    """Replace a scalar YAML value while preserving surrounding formatting."""
    pattern = re.compile(
        rf"^(\s*{re.escape(key)}\s*:\s*)(.+)$", re.MULTILINE
    )
    return pattern.sub(rf"\g<1>{new_value}", content, count=1)


def _replace_yaml_list(content: str, key: str, new_items: list[str]) -> str:
    """Replace a YAML list value while preserving surrounding formatting.

    Handles both inline `[a, b]` and block `- a\\n- b` list styles.
    """
    lines = content.split("\n")
    new_lines: list[str] = []
    i = 0
    found = False
    while i < len(lines):
        line = lines[i]
        # Match the key line
        match = re.match(rf"^(\s*){re.escape(key)}\s*:", line)
        if match and not found:
            found = True
            indent = match.group(1)
            # Check if inline list: `key: [a, b, c]`
            if "[" in line:
                new_list = "[" + ", ".join(new_items) + "]"
                new_lines.append(f"{indent}{key}: {new_list}")
                i += 1
                continue
            # Block list: emit key, then items
            new_lines.append(f"{indent}{key}:")
            item_indent = indent + "  "
            for item in new_items:
                new_lines.append(f"{item_indent}- {item}")
            # Skip old list items
            i += 1
            while i < len(lines) and re.match(rf"^\s+-\s+", lines[i]):
                i += 1
            continue
        new_lines.append(line)
        i += 1
    return "\n".join(new_lines)
