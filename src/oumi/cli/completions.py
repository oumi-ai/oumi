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

"""Shell completion functions for the Oumi CLI.

This module provides autocompletion support for recipe names and config paths,
enabling tab completion in bash, zsh, fish, and PowerShell shells.

Usage:
    To enable shell completions, run one of:
        oumi --install-completion  # Install completion for current shell
        oumi --show-completion     # Show completion script

Example completions:
    oumi train -c lla<TAB>           # Completes to llama3.1-8b, llama3.2-1b, ...
    oumi train -c configs/rec<TAB>   # Completes config file paths
"""

from functools import lru_cache
from pathlib import Path

from oumi.cli.alias import _ALIASES, AliasType


@lru_cache(maxsize=1)
def _get_all_aliases() -> list[str]:
    """Get a cached list of all recipe aliases.

    Returns:
        Sorted list of all available recipe aliases.
    """
    return sorted(_ALIASES.keys())


@lru_cache(maxsize=8)
def _get_aliases_for_type(alias_type: AliasType) -> list[str]:
    """Get a cached list of aliases for a specific command type.

    Args:
        alias_type: The type of command (train, infer, eval, etc.)

    Returns:
        Sorted list of aliases that support the given command type.
    """
    return sorted([k for k, v in _ALIASES.items() if alias_type in v])


def _fuzzy_match(pattern: str, text: str) -> bool:
    """Check if pattern fuzzy-matches text.

    Supports both prefix matching and fuzzy character matching.
    For example, "l38" matches "llama3.1-8b".

    Args:
        pattern: The search pattern (what user has typed).
        text: The text to match against.

    Returns:
        True if pattern fuzzy-matches text.
    """
    if not pattern:
        return True

    pattern_lower = pattern.lower()
    text_lower = text.lower()

    # Exact prefix match
    if text_lower.startswith(pattern_lower):
        return True

    # Fuzzy match: check if all pattern characters appear in order
    text_idx = 0
    for char in pattern_lower:
        # Find next occurrence of character in text
        while text_idx < len(text_lower) and text_lower[text_idx] != char:
            text_idx += 1
        if text_idx >= len(text_lower):
            return False
        text_idx += 1

    return True


def _score_match(pattern: str, text: str) -> int:
    """Score a match for sorting purposes.

    Lower scores are better matches.

    Args:
        pattern: The search pattern.
        text: The text that matched.

    Returns:
        Score indicating match quality (lower is better).
    """
    pattern_lower = pattern.lower()
    text_lower = text.lower()

    # Exact match
    if text_lower == pattern_lower:
        return 0

    # Prefix match - score by how much is left after prefix
    if text_lower.startswith(pattern_lower):
        return 1 + (len(text) - len(pattern))

    # Contains match - score by position
    idx = text_lower.find(pattern_lower)
    if idx >= 0:
        return 100 + idx

    # Fuzzy match - score by gaps between characters
    score = 200
    text_idx = 0
    for char in pattern_lower:
        while text_idx < len(text_lower) and text_lower[text_idx] != char:
            text_idx += 1
            score += 1
        text_idx += 1

    return score


def create_config_completer(alias_type: AliasType):
    """Create a completion function for config options.

    Creates a completion function that:
    1. Completes recipe aliases (e.g., "llama3.1-8b")
    2. Completes config file paths (e.g., "configs/recipes/...")
    3. Supports fuzzy matching

    Args:
        alias_type: The type of command to get aliases for.

    Returns:
        Completion function suitable for use with typer.Option(autocompletion=...).
    """

    def complete_config(incomplete: str) -> list[tuple[str, str]]:
        """Complete config names and paths.

        Args:
            incomplete: The partial text typed by the user.

        Returns:
            List of (completion, help_text) tuples.
        """
        completions: list[tuple[str, str]] = []

        # Check if user is typing a path
        if incomplete.startswith(("configs/", "./", "/", "~")):
            # Path completion mode
            completions.extend(_complete_config_paths(incomplete))
        else:
            # Recipe alias completion mode
            aliases = _get_aliases_for_type(alias_type)

            # Get matching aliases with fuzzy matching
            matches = []
            for alias in aliases:
                if _fuzzy_match(incomplete, alias):
                    score = _score_match(incomplete, alias)
                    # Get the config path for help text
                    config_path = _ALIASES.get(alias, {}).get(alias_type, "")
                    help_text = _get_alias_help_text(alias, config_path)
                    matches.append((score, alias, help_text))

            # Sort by score and return
            matches.sort(key=lambda x: x[0])
            completions.extend((alias, help_text) for _, alias, help_text in matches)

            # Also include path completions if user might be typing a path
            if not completions or len(incomplete) <= 2:
                path_completions = _complete_config_paths(incomplete)
                completions.extend(path_completions)

        return completions

    return complete_config


def _get_alias_help_text(alias: str, config_path: str) -> str:
    """Generate help text for an alias completion.

    Args:
        alias: The recipe alias name.
        config_path: The path to the config file.

    Returns:
        Help text describing the alias.
    """
    # Extract model family and type from alias
    parts = alias.split("-")

    # Common patterns: model-size, model-size-adapter
    if len(parts) >= 2:
        model = parts[0]
        size = parts[1] if len(parts) > 1 else ""
        adapter = parts[-1] if len(parts) > 2 and parts[-1] in ("lora", "qlora") else ""

        if adapter:
            return f"{model.title()} {size} ({adapter.upper()})"
        return f"{model.title()} {size}"

    return alias


def _complete_config_paths(incomplete: str) -> list[tuple[str, str]]:
    """Complete file paths for config files.

    Args:
        incomplete: The partial path typed by the user.

    Returns:
        List of (path, help_text) tuples for matching paths.
    """
    completions: list[tuple[str, str]] = []

    # Expand user home directory
    if incomplete.startswith("~"):
        expanded = Path(incomplete).expanduser()
        base_path = expanded.parent if not expanded.is_dir() else expanded
    elif incomplete.startswith("/"):
        base_path = Path(incomplete).parent if incomplete else Path("/")
    else:
        # Relative path
        if incomplete:
            path = Path(incomplete)
            base_path = path.parent if not path.is_dir() else path
        else:
            base_path = Path()

    # Get the prefix to match
    if incomplete.endswith("/"):
        prefix = ""
        search_path = Path(incomplete)
    else:
        prefix = Path(incomplete).name
        search_path = base_path

    try:
        if search_path.exists() and search_path.is_dir():
            for item in sorted(search_path.iterdir()):
                name = item.name

                # Skip hidden files
                if name.startswith("."):
                    continue

                # Check prefix match
                if prefix and not name.lower().startswith(prefix.lower()):
                    continue

                # Build completion path
                if incomplete.endswith("/"):
                    completion = incomplete + name
                elif "/" in incomplete:
                    completion = str(search_path / name)
                else:
                    completion = name

                # Add trailing slash for directories
                if item.is_dir():
                    completion += "/"
                    help_text = "Directory"
                elif name.endswith(".yaml") or name.endswith(".yml"):
                    help_text = "Config file"
                else:
                    continue  # Skip non-yaml files

                completions.append((completion, help_text))
    except (PermissionError, OSError):
        pass

    return completions


def complete_train_config(incomplete: str) -> list[tuple[str, str]]:
    """Complete training config names and paths.

    Args:
        incomplete: The partial text typed by the user.

    Returns:
        List of (completion, help_text) tuples.
    """
    completer = create_config_completer(AliasType.TRAIN)
    return completer(incomplete)


def complete_infer_config(incomplete: str) -> list[tuple[str, str]]:
    """Complete inference config names and paths.

    Args:
        incomplete: The partial text typed by the user.

    Returns:
        List of (completion, help_text) tuples.
    """
    completer = create_config_completer(AliasType.INFER)
    return completer(incomplete)


def complete_eval_config(incomplete: str) -> list[tuple[str, str]]:
    """Complete evaluation config names and paths.

    Args:
        incomplete: The partial text typed by the user.

    Returns:
        List of (completion, help_text) tuples.
    """
    completer = create_config_completer(AliasType.EVAL)
    return completer(incomplete)


def complete_tune_config(incomplete: str) -> list[tuple[str, str]]:
    """Complete tuning config names and paths.

    Args:
        incomplete: The partial text typed by the user.

    Returns:
        List of (completion, help_text) tuples.
    """
    completer = create_config_completer(AliasType.TUNE)
    return completer(incomplete)


def complete_quantize_config(incomplete: str) -> list[tuple[str, str]]:
    """Complete quantization config names and paths.

    Args:
        incomplete: The partial text typed by the user.

    Returns:
        List of (completion, help_text) tuples.
    """
    completer = create_config_completer(AliasType.QUANTIZE)
    return completer(incomplete)


def complete_judge_config(incomplete: str) -> list[tuple[str, str]]:
    """Complete judge config names and paths.

    Args:
        incomplete: The partial text typed by the user.

    Returns:
        List of (completion, help_text) tuples.
    """
    completer = create_config_completer(AliasType.JUDGE)
    return completer(incomplete)


def complete_analyze_config(incomplete: str) -> list[tuple[str, str]]:
    """Complete analyze config names and paths.

    Args:
        incomplete: The partial text typed by the user.

    Returns:
        List of (completion, help_text) tuples.
    """
    completer = create_config_completer(AliasType.ANALYZE)
    return completer(incomplete)


def complete_job_config(incomplete: str) -> list[tuple[str, str]]:
    """Complete job config names and paths.

    Args:
        incomplete: The partial text typed by the user.

    Returns:
        List of (completion, help_text) tuples.
    """
    completer = create_config_completer(AliasType.JOB)
    return completer(incomplete)


def complete_synth_config(incomplete: str) -> list[tuple[str, str]]:
    """Complete synth config names and paths.

    Args:
        incomplete: The partial text typed by the user.

    Returns:
        List of (completion, help_text) tuples.
    """
    completer = create_config_completer(AliasType.SYNTH)
    return completer(incomplete)
