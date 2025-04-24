from oumi.cli.alias import _ALIASES, AliasType, try_get_config_name_for_alias


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
