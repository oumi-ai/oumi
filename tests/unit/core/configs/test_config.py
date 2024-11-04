import os
import tempfile

from omegaconf import OmegaConf

from oumi.core.configs import (
    AutoWrapPolicy,
    BackwardPrefetch,
    DatasetParams,
    FSDPParams,
    ShardingStrategy,
    StateDictType,
    TrainingConfig,
)
from oumi.core.configs.params.training_params import TrainingParams
from oumi.core.distributed import prepare_accelerate_fsdp_run


def test_config_serialization():
    with tempfile.TemporaryDirectory() as folder:
        original_config = TrainingConfig()
        dataset_params = DatasetParams(dataset_name="my_test_dataset")
        original_config.data.train.datasets = [dataset_params]
        original_config.model.model_name = "my_test_model"
        filename = os.path.join(folder, "test_config.yaml")
        original_config.to_yaml(filename)

        assert os.path.exists(filename)

        loaded_config = TrainingConfig.from_yaml(filename)
        assert loaded_config.model.model_name == "my_test_model"
        assert len(loaded_config.data.train.datasets) == 1
        assert loaded_config.data.train.datasets[0].dataset_name == "my_test_dataset"
        assert original_config == loaded_config


def test_config_equality():
    config_a = TrainingConfig()
    config_b = TrainingConfig()
    assert config_a == config_b

    config_a.model.model_name = "test_model"
    assert config_a != config_b


def test_config_override():
    low_priority_config = TrainingConfig()
    low_priority_config.model.model_name = "model_low_priority"

    high_priority_config = TrainingConfig()
    high_priority_config.model.model_name = "model_high_priority"

    # Override with CLI arguments if provided
    merged_config = OmegaConf.merge(low_priority_config, high_priority_config)
    assert merged_config.model.model_name == "model_high_priority"
    assert merged_config == high_priority_config
    assert merged_config != low_priority_config


def test_get_accelerate_env_vars_default():
    config = TrainingConfig()
    env_vars = prepare_accelerate_fsdp_run(config)
    assert env_vars == {
        "ACCELERATE_DYNAMO_BACKEND": "NO",
        "ACCELERATE_DYNAMO_MODE": "default",
        "ACCELERATE_DYNAMO_USE_FULLGRAPH": "False",
        "ACCELERATE_DYNAMO_USE_DYNAMIC": "False",
        "FSDP_USE_ORIG_PARAMS": "true",
        "FSDP_CPU_RAM_EFFICIENT_LOADING": "true",
        "ACCELERATE_USE_FSDP": "false",
        "FSDP_SHARDING_STRATEGY": "FULL_SHARD",
        "FSDP_OFFLOAD_PARAMS": "false",
        "FSDP_BACKWARD_PREFETCH": "BACKWARD_PRE",
        "FSDP_FORWARD_PREFETCH": "false",
        "FSDP_STATE_DICT_TYPE": "FULL_STATE_DICT",
        "FSDP_AUTO_WRAP_POLICY": "SIZE_BASED_WRAP",
        "FSDP_MIN_NUM_PARAMS": "100000",
        "FSDP_SYNC_MODULE_STATES": "true",
        "FSDP_ACTIVATION_CHECKPOINTING": "false",
    }


def test_get_accelerate_env_vars():
    fsdp_params = FSDPParams(
        enable_fsdp=True,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        cpu_offload=True,
        mixed_precision="bf16",
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
        forward_prefetch=True,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        auto_wrap_policy=AutoWrapPolicy.TRANSFORMER_BASED_WRAP,
        transformer_layer_cls="LlamaDecoderLayer",
        sync_module_states=False,
        min_num_params=100000000,
    )
    config = TrainingConfig(
        fsdp=fsdp_params, training=TrainingParams(enable_gradient_checkpointing=True)
    )

    env_vars = prepare_accelerate_fsdp_run(config)
    assert env_vars == {
        "ACCELERATE_DYNAMO_BACKEND": "NO",
        "ACCELERATE_DYNAMO_MODE": "default",
        "ACCELERATE_DYNAMO_USE_FULLGRAPH": "False",
        "ACCELERATE_DYNAMO_USE_DYNAMIC": "False",
        "FSDP_USE_ORIG_PARAMS": "true",
        "FSDP_CPU_RAM_EFFICIENT_LOADING": "true",
        "ACCELERATE_USE_FSDP": "true",
        "FSDP_SHARDING_STRATEGY": "HYBRID_SHARD",
        "FSDP_OFFLOAD_PARAMS": "true",
        "ACCELERATE_MIXED_PRECISION": "bf16",
        "FSDP_BACKWARD_PREFETCH": "BACKWARD_POST",
        "FSDP_FORWARD_PREFETCH": "true",
        "FSDP_STATE_DICT_TYPE": "SHARDED_STATE_DICT",
        "FSDP_AUTO_WRAP_POLICY": "TRANSFORMER_BASED_WRAP",
        "FSDP_TRANSFORMER_CLS_TO_WRAP": "LlamaDecoderLayer",
        "FSDP_MIN_NUM_PARAMS": "100000000",
        "FSDP_SYNC_MODULE_STATES": "false",
        "FSDP_ACTIVATION_CHECKPOINTING": "true",
    }
