import pytest

from oumi.cli.alias import _ALIASES, AliasType, try_get_config_name_for_alias
from oumi.utils.io_utils import get_oumi_root_directory

# get_oumi_root_directory() returns src/oumi/, go up two levels to get repo root
_REPO_ROOT = get_oumi_root_directory()
_CONFIG_ROOT = _REPO_ROOT.parent.parent / "configs"
_OUMI_PREFIX = "oumi://configs"


def test_alias_all_entries():
    for alias in _ALIASES:
        for alias_type in _ALIASES[alias]:
            config_path = try_get_config_name_for_alias(alias, alias_type)
            assert config_path == _ALIASES[alias][alias_type], (
                f"Alias '{alias}' with type '{alias_type}' did not resolve correctly."
                f" Expected: {config_path}, Actual: {_ALIASES[alias][alias_type]}"
            )


def test_alias_not_found():
    alias = "non_existent_alias"
    alias_type = AliasType.TRAIN
    config_path = try_get_config_name_for_alias(alias, alias_type)
    assert config_path == alias, (
        f"Expected the original alias '{alias}' to be returned."
    )


def test_alias_type_not_found():
    alias = "llama4-scout"
    config_path = try_get_config_name_for_alias(alias, AliasType.EVAL)
    assert config_path == alias, (
        f"Expected the original alias '{alias}' to be returned."
    )


def test_alias_configs_exist():
    """Verify all aliased config files actually exist on disk."""

    if not _CONFIG_ROOT.exists():
        pytest.skip(
            "configs directory not found - this can happen with pip installed oumi"
        )

    missing_configs = []

    for alias in _ALIASES:
        for alias_type in _ALIASES[alias]:
            config_path = _ALIASES[alias][alias_type]

            # Convert oumi:// path to actual file path
            if config_path.startswith(_OUMI_PREFIX):
                relative_path = config_path[len(_OUMI_PREFIX) :].lstrip("/")
                actual_path = _CONFIG_ROOT / relative_path

                if not actual_path.exists():
                    missing_configs.append(
                        f"  - {alias} ({alias_type.value}): {config_path}"
                    )

    assert not missing_configs, (
        "The following aliased config files do not exist:\n"
        "Config root: " + str(_CONFIG_ROOT) + "\n"
        "Repo root: " + str(_REPO_ROOT) + "\n" + "\n".join(missing_configs)
    )


def test_no_duplicate_aliases():
    """Detect duplicate alias definitions in source code."""
    import ast

    source_path = _REPO_ROOT / "cli/alias.py"
    tree = ast.parse(source_path.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_ALIASES":
                    if isinstance(node.value, ast.Dict):
                        keys = [
                            k.value
                            for k in node.value.keys
                            if isinstance(k, ast.Constant)
                        ]
                        duplicates = [k for k in keys if keys.count(k) > 1]
                        assert not duplicates, (
                            f"Duplicate aliases found: {set(duplicates)}"
                        )
