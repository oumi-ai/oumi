from unittest.mock import patch

from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    ModelParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
)


def test_training_config_processor_kwargs():
    """Test that json, regex, and choice parameters are mutually exclusive."""
    config = TrainingConfig(
        model=ModelParams(
            model_name="llava-hf/llava-1.5-7b-hf",
            processor_kwargs={"num_patches": 16},
        ),
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="llava_collator",
                datasets=[
                    DatasetParams(
                        dataset_name="merve/vqav2-small",
                        split="train",
                        dataset_kwargs={"processor_name": "llava-hf/llava-1.5-7b-hf"},
                    ),
                    DatasetParams(
                        dataset_name="HuggingFaceH4/llava-instruct-mix-vsft",
                        split="train",
                        dataset_kwargs={
                            "processor_name": "llava-hf/llava-1.5-7b-hf",
                            "processor_kwargs": {"num_patches": 32, "foo": "bar"},
                        },
                    ),
                    DatasetParams(
                        dataset_name="HuggingFaceM4/the_cauldron",
                        split="train",
                        subset="ocrvqa",
                        dataset_kwargs={
                            "processor_name": "microsoft/Phi-3-vision-128k-instruct"
                        },
                    ),
                    DatasetParams(
                        dataset_name="HuggingFaceM4/the_cauldron",
                        split="test",
                        subset="ocrvqa",
                        dataset_kwargs={
                            "processor_name": "llava-hf/llava-1.5-7b-hf",
                            "processor_kwargs": {},
                        },
                    ),
                ],
            ),
            validation=DatasetSplitParams(
                collator_name="llava_collator",
                datasets=[
                    DatasetParams(
                        dataset_name="merve/vqav2-small",
                        split="validation",
                        dataset_kwargs={"processor_name": "llava-hf/llava-1.5-7b-hf"},
                    )
                ],
            ),
            test=DatasetSplitParams(
                collator_name="llava_collator",
                datasets=[
                    DatasetParams(
                        dataset_name="HuggingFaceH4/llava-instruct-mix-vsft",
                        split="test",
                        dataset_kwargs={"processor_name": "llava-hf/llava-1.5-7b-hf"},
                    )
                ],
            ),
        ),
    )
    assert len(config.data.train.datasets) == 4
    assert config.data.train.datasets[0] == DatasetParams(
        dataset_name="merve/vqav2-small",
        split="train",
        dataset_kwargs={
            "processor_name": "llava-hf/llava-1.5-7b-hf",
            "processor_kwargs": {
                "num_patches": 16,
            },
        },
    )
    assert config.data.train.datasets[1] == DatasetParams(
        dataset_name="HuggingFaceH4/llava-instruct-mix-vsft",
        split="train",
        dataset_kwargs={
            "processor_name": "llava-hf/llava-1.5-7b-hf",
            "processor_kwargs": {"num_patches": 32, "foo": "bar"},
        },
    )
    assert config.data.train.datasets[2] == DatasetParams(
        dataset_name="HuggingFaceM4/the_cauldron",
        split="train",
        subset="ocrvqa",
        dataset_kwargs={"processor_name": "microsoft/Phi-3-vision-128k-instruct"},
    )
    assert config.data.train.datasets[3] == DatasetParams(
        dataset_name="HuggingFaceM4/the_cauldron",
        split="test",
        subset="ocrvqa",
        dataset_kwargs={
            "processor_name": "llava-hf/llava-1.5-7b-hf",
            "processor_kwargs": {},
        },
    )

    assert len(config.data.validation.datasets) == 1
    assert config.data.validation.datasets[0] == DatasetParams(
        dataset_name="merve/vqav2-small",
        split="validation",
        dataset_kwargs={
            "processor_name": "llava-hf/llava-1.5-7b-hf",
            "processor_kwargs": {
                "num_patches": 16,
            },
        },
    )

    assert len(config.data.test.datasets) == 1
    assert config.data.test.datasets[0] == DatasetParams(
        dataset_name="HuggingFaceH4/llava-instruct-mix-vsft",
        split="test",
        dataset_kwargs={
            "processor_name": "llava-hf/llava-1.5-7b-hf",
            "processor_kwargs": {
                "num_patches": 16,
            },
        },
    )


def _verl_grpo_config(model: ModelParams) -> TrainingConfig:
    return TrainingConfig(
        model=model,
        data=DataParams(
            validation=DatasetSplitParams(
                datasets=[DatasetParams(dataset_name="text_sft_jsonl")]
            )
        ),
        training=TrainingParams(trainer_type=TrainerType.VERL_GRPO),
    )


@patch("oumi.core.configs.training_config.logger")
def test_verl_grpo_warns_text_only_is_ignored(mock_logger):
    _verl_grpo_config(ModelParams(model_name="Qwen/Qwen3.5-4B", text_only=True))

    messages = [c.args[0] for c in mock_logger.warning.call_args_list]
    assert any("model.text_only has no effect for VERL_GRPO" in m for m in messages)


@patch("oumi.core.configs.training_config.logger")
def test_verl_grpo_warns_freeze_layers_is_ignored(mock_logger):
    _verl_grpo_config(
        ModelParams(model_name="Qwen/Qwen3.5-4B", freeze_layers=["model.visual"])
    )

    messages = [c.args[0] for c in mock_logger.warning.call_args_list]
    assert any("model.freeze_layers" in m and "VERL_GRPO" in m for m in messages)


@patch("oumi.core.configs.training_config.logger")
def test_verl_grpo_no_warning_without_ignored_model_options(mock_logger):
    _verl_grpo_config(ModelParams(model_name="Qwen/Qwen3.5-4B"))

    messages = [c.args[0] for c in mock_logger.warning.call_args_list]
    assert not any("VERL_GRPO" in m for m in messages)


@patch("oumi.core.configs.training_config.logger")
def test_text_only_does_not_warn_for_hf_trainers(mock_logger):
    TrainingConfig(
        model=ModelParams(model_name="Qwen/Qwen3.5-4B", text_only=True),
        training=TrainingParams(trainer_type=TrainerType.TRL_SFT),
    )

    messages = [c.args[0] for c in mock_logger.warning.call_args_list]
    assert not any("text_only" in m for m in messages)
