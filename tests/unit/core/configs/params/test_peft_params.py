import re

from oumi.core.configs.params.peft_params import PeftParams


def test_to_lora_no_scope_passes_list_through():
    """Without scope, target_modules list is passed as-is to LoraConfig."""
    params = PeftParams(
        lora_target_modules=["q_proj", "v_proj"],
    )
    config = params.to_lora()
    assert config.target_modules == {"q_proj", "v_proj"}


def test_to_lora_no_scope_none_modules():
    """Without scope or modules, target_modules is None."""
    params = PeftParams()
    config = params.to_lora()
    assert config.target_modules is None


def test_to_lora_all_linear_without_scope():
    """all-linear shorthand works without scope."""
    params = PeftParams(lora_target_modules=["all-linear"])
    config = params.to_lora()
    assert config.target_modules == "all-linear"


def test_to_lora_scope_builds_regex():
    """Scope + modules produces a regex string targeting only that scope."""
    params = PeftParams(
        lora_target_modules_scope="language_model",
        lora_target_modules=["q_proj", "k_proj", "v_proj"],
    )
    config = params.to_lora()

    # Should be a string (regex), not a list/set
    assert isinstance(config.target_modules, str)

    # Should match language_model paths
    assert re.fullmatch(
        config.target_modules,
        "model.language_model.layers.0.self_attn.q_proj",
    )
    assert re.fullmatch(
        config.target_modules,
        "model.language_model.layers.15.self_attn.v_proj",
    )

    # Should NOT match vision_tower paths
    assert not re.fullmatch(
        config.target_modules,
        "model.vision_tower.encoder.layers.0.self_attn.q_proj",
    )

    # Should NOT match audio_tower paths
    assert not re.fullmatch(
        config.target_modules,
        "model.audio_tower.layers.0.self_attn.q_proj",
    )


def test_to_lora_scope_vision_tower():
    """Vision tower scope with .linear suffix targets only vision layers."""
    params = PeftParams(
        lora_target_modules_scope="vision_tower",
        lora_target_modules=["q_proj.linear", "v_proj.linear"],
    )
    config = params.to_lora()

    assert isinstance(config.target_modules, str)

    # Should match vision_tower paths with .linear
    assert re.fullmatch(
        config.target_modules,
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear",
    )

    # Should NOT match language_model paths
    assert not re.fullmatch(
        config.target_modules,
        "model.language_model.layers.0.self_attn.q_proj",
    )

    # Should NOT match vision_tower without .linear suffix
    assert not re.fullmatch(
        config.target_modules,
        "model.vision_tower.encoder.layers.0.self_attn.q_proj",
    )


def test_to_lora_scope_with_empty_modules_no_regex():
    """Scope is ignored when target_modules is empty or None."""
    params_none = PeftParams(
        lora_target_modules_scope="language_model",
        lora_target_modules=None,
    )
    config = params_none.to_lora()
    assert config.target_modules is None

    params_empty = PeftParams(
        lora_target_modules_scope="language_model",
        lora_target_modules=[],
    )
    config = params_empty.to_lora()
    # Empty list is falsy, so scope branch is skipped
    assert config.target_modules == set()


def test_to_lora_scope_escapes_special_chars():
    """Special regex characters in scope or module names are escaped."""
    params = PeftParams(
        lora_target_modules_scope="model.sub",
        lora_target_modules=["proj.linear"],
    )
    config = params.to_lora()

    assert isinstance(config.target_modules, str)

    # The dots in scope and module name should be literal, not regex wildcards
    assert re.fullmatch(
        config.target_modules,
        "foo.model.sub.layers.0.proj.linear",
    )
    # A path where the dot is replaced by another char should NOT match
    assert not re.fullmatch(
        config.target_modules,
        "foo.modelXsub.layers.0.projXlinear",
    )


def test_to_lora_scope_single_module():
    """Scope works with a single module in the list."""
    params = PeftParams(
        lora_target_modules_scope="language_model",
        lora_target_modules=["q_proj"],
    )
    config = params.to_lora()

    assert isinstance(config.target_modules, str)
    assert re.fullmatch(
        config.target_modules,
        "model.language_model.layers.0.self_attn.q_proj",
    )
