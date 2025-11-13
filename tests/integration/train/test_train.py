import pathlib
import tempfile

import pytest

from oumi import train
from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    ModelParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
)
from oumi.core.configs.params.gkd_params import GkdParams


def test_train_basic():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        output_training_dir = str(pathlib.Path(output_temp_dir) / "train")
        config: TrainingConfig = TrainingConfig(
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="yahma/alpaca-cleaned",
                        )
                    ],
                ),
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
                model_max_length=1024,
                trust_remote_code=True,
                tokenizer_pad_token="<|endoftext|>",
            ),
            training=TrainingParams(
                trainer_type=TrainerType.TRL_SFT,
                max_steps=3,
                logging_steps=3,
                log_model_summary=True,
                enable_wandb=False,
                enable_tensorboard=False,
                enable_mlflow=False,
                output_dir=output_training_dir,
                try_resume_from_last_checkpoint=True,
                save_final_model=True,
            ),
        )

        train(config)


def test_train_unregistered_metrics_function():
    with pytest.raises(KeyError) as exception_info:
        with tempfile.TemporaryDirectory() as output_temp_dir:
            output_training_dir = str(pathlib.Path(output_temp_dir) / "train")
            config: TrainingConfig = TrainingConfig(
                data=DataParams(
                    train=DatasetSplitParams(
                        datasets=[
                            DatasetParams(
                                dataset_name="yahma/alpaca-cleaned",
                            )
                        ],
                    ),
                ),
                model=ModelParams(
                    model_name="openai-community/gpt2",
                    model_max_length=1024,
                    trust_remote_code=True,
                    tokenizer_pad_token="<|endoftext|>",
                ),
                training=TrainingParams(
                    trainer_type=TrainerType.TRL_SFT,
                    metrics_function="unregistered_function_name",
                    max_steps=2,
                    logging_steps=2,
                    log_model_summary=True,
                    enable_wandb=False,
                    enable_tensorboard=False,
                    enable_mlflow=False,
                    output_dir=output_training_dir,
                    try_resume_from_last_checkpoint=True,
                    save_final_model=False,
                ),
            )

            train(config)
    assert "unregistered_function_name" in str(exception_info.value)


def test_train_pack_with_pretraining_dataset():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        config: TrainingConfig = TrainingConfig(
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="debug_pretraining",
                            dataset_kwargs={"dataset_size": 25, "seq_length": 128},
                        )
                    ],
                    stream=True,
                    pack=True,
                ),
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
                # The true max length is 1024, but a lower value works. This is done to
                # reduce test runtime.
                model_max_length=128,
                trust_remote_code=True,
                tokenizer_pad_token="<|endoftext|>",
            ),
            training=TrainingParams(
                trainer_type=TrainerType.TRL_SFT,
                max_steps=1,
                logging_steps=1,
                enable_wandb=False,
                enable_tensorboard=False,
                enable_mlflow=False,
                output_dir=output_temp_dir,
            ),
        )

        train(config)


def test_train_pack_with_sft_dataset():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        config: TrainingConfig = TrainingConfig(
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="debug_sft",
                            dataset_kwargs={"dataset_size": 10},
                        )
                    ],
                    stream=False,
                    pack=True,
                ),
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
                # The true max length is 1024, but a lower value works. This is done to
                # reduce test runtime.
                model_max_length=128,
                trust_remote_code=True,
                tokenizer_pad_token="<|endoftext|>",
            ),
            training=TrainingParams(
                trainer_type=TrainerType.TRL_SFT,
                max_steps=1,
                logging_steps=1,
                enable_wandb=False,
                enable_tensorboard=False,
                enable_mlflow=False,
                output_dir=output_temp_dir,
            ),
        )

        train(config)


def test_train_dpo():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        output_training_dir = str(pathlib.Path(output_temp_dir) / "train")
        config: TrainingConfig = TrainingConfig(
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="debug_dpo",
                        )
                    ],
                ),
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
                model_max_length=1024,
                trust_remote_code=True,
                tokenizer_pad_token="<|endoftext|>",
            ),
            training=TrainingParams(
                per_device_train_batch_size=1,
                trainer_type=TrainerType.TRL_DPO,
                max_steps=3,
                logging_steps=3,
                log_model_summary=True,
                enable_wandb=False,
                enable_tensorboard=False,
                enable_mlflow=False,
                output_dir=output_training_dir,
                try_resume_from_last_checkpoint=False,
                save_final_model=True,
            ),
        )

        train(config)


def test_train_kto():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        output_training_dir = str(pathlib.Path(output_temp_dir) / "train")
        config: TrainingConfig = TrainingConfig(
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="debug_kto",
                        )
                    ],
                ),
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
                model_max_length=1024,
                trust_remote_code=True,
                tokenizer_pad_token="<|endoftext|>",
            ),
            training=TrainingParams(
                per_device_train_batch_size=2,
                trainer_type=TrainerType.TRL_KTO,
                max_steps=3,
                logging_steps=3,
                log_model_summary=True,
                enable_wandb=False,
                enable_tensorboard=False,
                output_dir=output_training_dir,
                try_resume_from_last_checkpoint=False,
                save_final_model=True,
                trainer_kwargs={
                    "max_length": 512,
                    "max_prompt_length": 128,
                    "remove_unused_columns": False,
                    "desirable_weight": 0.8,
                },
            ),
        )

        train(config)


def test_train_gkd():
    """Test GKD (Generalized Knowledge Distillation) training workflow."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        output_training_dir = str(pathlib.Path(output_temp_dir) / "train")
        config: TrainingConfig = TrainingConfig(
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="debug_sft",
                            dataset_kwargs={
                                "dataset_size": 5,
                                "return_conversations": True,
                                "return_conversations_format": "dict",
                            },
                        )
                    ],
                ),
            ),
            model=ModelParams(
                # Small student model for fast testing
                model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
                model_max_length=512,
                trust_remote_code=True,
            ),
            training=TrainingParams(
                per_device_train_batch_size=1,
                trainer_type=TrainerType.TRL_GKD,
                max_steps=2,
                logging_steps=1,
                log_model_summary=False,
                enable_wandb=False,
                enable_tensorboard=False,
                enable_mlflow=False,
                output_dir=output_training_dir,
                try_resume_from_last_checkpoint=False,
                save_final_model=False,
                gkd=GkdParams(
                    # Small teacher model for fast testing
                    teacher_model_name_or_path="HuggingFaceTB/SmolLM2-360M-Instruct",
                    teacher_model_init_kwargs={
                        "attn_implementation": "sdpa",
                    },
                    temperature=0.9,
                    lmbda=0.5,
                    beta=0.5,
                    max_new_tokens=128,
                    disable_dropout=True,
                    seq_kd=False,
                ),
            ),
        )

        train(config)
