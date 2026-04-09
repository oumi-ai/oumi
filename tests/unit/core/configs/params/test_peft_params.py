import re

from oumi.core.configs.params.peft_params import PeftParams


def test_to_lora_plain_list_passes_through():
    """Plain module names (no regex) are passed as-is to LoraConfig."""
    params = PeftParams(
        lora_target_modules=["q_proj", "v_proj"],
    )
    config = params.to_lora()
    assert config.target_modules == {"q_proj", "v_proj"}


def test_to_lora_none_modules():
    """Without modules, target_modules is None."""
    params = PeftParams()
    config = params.to_lora()
    assert config.target_modules is None


def test_to_lora_all_linear():
    """all-linear shorthand works."""
    params = PeftParams(lora_target_modules=["all-linear"])
    config = params.to_lora()
    assert config.target_modules == "all-linear"


def test_to_lora_regex_items_joined_into_string():
    """Items with regex metacharacters are joined into a single regex string."""
    params = PeftParams(
        lora_target_modules=[
            ".*language_model.*q_proj",
            ".*language_model.*v_proj",
        ],
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


def test_to_lora_regex_vision_tower():
    """Regex targeting vision tower with .linear suffix."""
    params = PeftParams(
        lora_target_modules=[
            ".*vision_tower.*q_proj\\.linear",
            ".*vision_tower.*v_proj\\.linear",
        ],
    )
    config = params.to_lora()

    assert isinstance(config.target_modules, str)

    assert re.fullmatch(
        config.target_modules,
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear",
    )
    assert not re.fullmatch(
        config.target_modules,
        "model.language_model.layers.0.self_attn.q_proj",
    )


def test_to_lora_empty_list_no_regex():
    """Empty list passes through without regex detection."""
    params = PeftParams(lora_target_modules=[])
    config = params.to_lora()
    assert config.target_modules == set()


def test_to_lora_regex_single_item():
    """Single regex item works."""
    params = PeftParams(
        lora_target_modules=[".*language_model.*q_proj"],
    )
    config = params.to_lora()

    assert isinstance(config.target_modules, str)
    assert re.fullmatch(
        config.target_modules,
        "model.language_model.layers.0.self_attn.q_proj",
    )


def test_to_lora_mixed_plain_and_regex_detected():
    """If any item has regex chars, all items are joined as regex."""
    params = PeftParams(
        lora_target_modules=[
            ".*language_model.*q_proj",
            "gate_proj",  # plain, but will be treated as regex too
        ],
    )
    config = params.to_lora()

    # Should be joined as regex string
    assert isinstance(config.target_modules, str)

    # The regex item matches scoped
    assert re.fullmatch(
        config.target_modules,
        "model.language_model.layers.0.self_attn.q_proj",
    )
    # The plain item "gate_proj" as regex matches literally
    assert re.fullmatch(
        config.target_modules,
        "gate_proj",
    )
