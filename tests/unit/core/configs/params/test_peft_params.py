import re

from oumi.core.configs.params.peft_params import PeftParams


def test_to_lora_plain_list_passes_through():
    """Plain module names are passed to LoraConfig as a list (suffix match)."""
    params = PeftParams(lora_target_modules=["q_proj", "v_proj"])
    config = params.to_lora()
    assert config.target_modules == {"q_proj", "v_proj"}


def test_to_lora_dotted_names_not_regex_by_default():
    """Without the regex flag, dotted entries like "self_attn.q_proj" are passed
    through as literal names for PEFT to suffix-match, not compiled to a regex."""
    params = PeftParams(lora_target_modules=["self_attn.q_proj"])
    config = params.to_lora()
    assert config.target_modules == {"self_attn.q_proj"}


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


def test_to_lora_empty_list():
    """Empty list targets no modules."""
    params = PeftParams(lora_target_modules=[])
    config = params.to_lora()
    assert config.target_modules == set()


def test_to_lora_regex_flag_joins_items_into_string():
    """With the regex flag set, items are joined into one fullmatch regex string."""
    params = PeftParams(
        lora_target_modules=[
            ".*language_model.*q_proj",
            ".*language_model.*v_proj",
        ],
        lora_target_modules_regex=True,
    )
    config = params.to_lora()

    assert isinstance(config.target_modules, str)
    assert re.fullmatch(
        config.target_modules,
        "model.language_model.layers.0.self_attn.q_proj",
    )
    assert re.fullmatch(
        config.target_modules,
        "model.language_model.layers.15.self_attn.v_proj",
    )
    assert not re.fullmatch(
        config.target_modules,
        "model.vision_tower.encoder.layers.0.self_attn.q_proj",
    )
    assert not re.fullmatch(
        config.target_modules,
        "model.audio_tower.layers.0.self_attn.q_proj",
    )


def test_to_lora_regex_flag_single_item():
    """A single regex item is joined into a string under the flag."""
    params = PeftParams(
        lora_target_modules=[".*language_model.*q_proj"],
        lora_target_modules_regex=True,
    )
    config = params.to_lora()

    assert isinstance(config.target_modules, str)
    assert re.fullmatch(
        config.target_modules,
        "model.language_model.layers.0.self_attn.q_proj",
    )


def test_to_lora_regex_flag_vision_tower():
    """Regex targeting the vision tower with a ".linear" suffix."""
    params = PeftParams(
        lora_target_modules=[
            ".*vision_tower.*q_proj\\.linear",
            ".*vision_tower.*v_proj\\.linear",
        ],
        lora_target_modules_regex=True,
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


def test_to_lora_all_linear_ignores_regex_flag():
    """all-linear stays a passthrough string even when the regex flag is set."""
    params = PeftParams(
        lora_target_modules=["all-linear"],
        lora_target_modules_regex=True,
    )
    config = params.to_lora()
    assert config.target_modules == "all-linear"
