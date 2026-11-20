from types import MappingProxyType

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

from oumi.verl_grpo.config import merge_verl_config
from oumi.verl_grpo.errors import UnknownVerlKeyError


def test_merge_with_no_layers_is_the_defaults_plus_empty_reward_kwargs():
    merged = merge_verl_config()
    assert merged.trainer.use_v1 is True
    assert merged.trainer.v1.trainer_mode == "sync"
    reward_kwargs = merged.reward.custom_reward_function.reward_kwargs
    assert OmegaConf.to_container(reward_kwargs) == {}


def test_unknown_key_raises_with_its_dotted_name():
    with pytest.raises(UnknownVerlKeyError) as error:
        merge_verl_config({"actor_rollout_ref": {"actor": {"ppo_mini_batch_sise": 8}}})
    assert error.value.key == "actor_rollout_ref.actor.ppo_mini_batch_sise"
    assert "verl 0.9.1" in str(error.value)
    assert str(error.value) == (
        "verl 0.9.1 has no setting 'actor_rollout_ref.actor.ppo_mini_batch_sise'"
    )


def test_a_dotted_top_level_key_is_unknown():
    with pytest.raises(UnknownVerlKeyError, match="trainer.total_epochs"):
        merge_verl_config({"trainer.total_epochs": 3})


def test_a_scalar_where_verl_has_a_map_raises_value_error():
    with pytest.raises(ValueError):
        merge_verl_config({"trainer": 1})


@pytest.mark.parametrize(
    "path",
    [
        "data.apply_chat_template_kwargs",
        "actor_rollout_ref.model.override_config",
        "reward.custom_reward_function.reward_kwargs",
    ],
)
def test_free_form_maps_accept_new_nested_keys(path):
    layer = OmegaConf.to_container(OmegaConf.from_dotlist([f"{path}.new_key.nested=1"]))
    assert isinstance(layer, dict)
    layer = {str(key): value for key, value in layer.items()}
    assert OmegaConf.select(merge_verl_config(layer), f"{path}.new_key.nested") == 1


def test_later_layers_win_and_lists_replace():
    merged = merge_verl_config(
        {"trainer": {"logger": ["console"]}}, {"trainer": {"logger": ["file"]}}
    )
    assert list(merged.trainer.logger) == ["file"]


def test_merge_is_idempotent():
    once = merge_verl_config({"trainer": {"total_epochs": 2}})
    assert merge_verl_config(once) == once


def test_a_read_only_layer_gives_a_writable_result():
    layer = OmegaConf.create({"trainer": {"total_epochs": 2}})
    OmegaConf.set_readonly(layer, True)
    merged = merge_verl_config(layer)
    merged.trainer.total_epochs = 3
    assert merged.trainer.total_epochs == 3


def test_interpolations_stay_unresolved():
    merged = merge_verl_config()
    assert OmegaConf.is_interpolation(merged.data, "tool_config_path")


def test_the_result_is_struct():
    merged = merge_verl_config()
    with pytest.raises(ConfigAttributeError):
        merged.trainer.not_a_key = 1


def test_each_call_returns_an_independent_config():
    first = merge_verl_config()
    first.trainer.total_epochs = 99
    assert merge_verl_config().trainer.total_epochs == 30


def _optimizer_layer(value):
    return {
        "actor_rollout_ref": {
            "actor": {"optim": {"override_optimizer_config": value}},
        }
    }


def _optimizer_overrides(config):
    return OmegaConf.to_container(
        config.actor_rollout_ref.actor.optim.override_optimizer_config
    )


def test_a_map_set_on_a_null_default_accepts_keys_from_a_later_layer():
    merged = merge_verl_config(_optimizer_layer({"a": 1}), _optimizer_layer({"b": 2}))
    assert _optimizer_overrides(merged) == {"a": 1, "b": 2}


def test_a_stored_result_accepts_an_override_of_a_null_default_map():
    stored = merge_verl_config(_optimizer_layer({"a": 1}))
    merged = merge_verl_config(stored, _optimizer_layer({"c": 3}))
    assert _optimizer_overrides(merged) == {"a": 1, "c": 3}


def test_a_map_over_a_scalar_default_raises_value_error():
    with pytest.raises(ValueError, match="trainer.total_epochs"):
        merge_verl_config({"trainer": {"total_epochs": {"typo": 1}}})


def test_a_list_where_verl_has_a_map_raises_value_error():
    with pytest.raises(ValueError, match="trainer"):
        merge_verl_config({"trainer": [1, 2]})


def test_a_top_level_mapping_that_is_not_a_dict_is_accepted():
    merged = merge_verl_config(MappingProxyType({"trainer": {"total_epochs": 4}}))
    assert merged.trainer.total_epochs == 4


def test_an_interpolation_may_alias_a_whole_section():
    merged = merge_verl_config(
        {"critic": {"optim": "${actor_rollout_ref.actor.optim}"}}
    )
    assert OmegaConf.is_interpolation(merged.critic, "optim")
