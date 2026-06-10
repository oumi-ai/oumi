import re

from oumi.core.configs.params.peft_params import PeftParams


def test_to_lora_plain_list_passes_through():
    """Plain module names are passed to LoraConfig as a list (suffix match)."""
    params = PeftParams(lora_target_modules=["q_proj", "v_proj"])
    config = params.to_lora()
    assert config.target_modules == {"q_proj", "v_proj"}
    assert config.exclude_modules is None


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


def test_to_lora_exclude_modules_joined_into_regex():
    """Exclude patterns are joined into one regex (matched by re.fullmatch, the same
    way PEFT applies exclude_modules), while the target list keeps its plain names."""
    params = PeftParams(
        lora_target_modules=["q_proj", "v_proj"],
        lora_exclude_modules=[".*vision_tower.*", ".*audio_tower.*"],
    )
    config = params.to_lora()

    assert config.target_modules == {"q_proj", "v_proj"}
    assert isinstance(config.exclude_modules, str)
    # Tower projections are excluded; the language model is not.
    assert re.fullmatch(config.exclude_modules, "model.vision_tower.layers.0.q_proj")
    assert re.fullmatch(config.exclude_modules, "model.audio_tower.layers.3.v_proj")
    assert not re.fullmatch(
        config.exclude_modules, "model.language_model.layers.0.self_attn.q_proj"
    )
