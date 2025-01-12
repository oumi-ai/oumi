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


def test_train_basic():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        output_training_dir = str(pathlib.Path(output_temp_dir) / "train")
        config: TrainingConfig = TrainingConfig(
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="debug_sft",
                        )
                    ],
                ),
            ),
            model=ModelParams(
                model_name="MlpEncoder",
                model_max_length=1024,
                trust_remote_code=True,
                tokenizer_name="gpt2",
                tokenizer_pad_token="<|endoftext|>",
                load_pretrained_weights=False,
            ),
            training=TrainingParams(
                trainer_type=TrainerType.TRL_SFT,
                max_steps=3,
                logging_steps=3,
                log_model_summary=True,
                enable_wandb=False,
                enable_tensorboard=False,
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
                                dataset_name="debug_sft",
                            )
                        ],
                    ),
                ),
                model=ModelParams(
                    model_name="MlpEncoder",
                    model_max_length=1024,
                    trust_remote_code=True,
                    tokenizer_name="gpt2",
                    tokenizer_pad_token="<|endoftext|>",
                    load_pretrained_weights=False,
                ),
                training=TrainingParams(
                    trainer_type=TrainerType.TRL_SFT,
                    metrics_function="unregistered_function_name",
                    max_steps=2,
                    logging_steps=2,
                    log_model_summary=True,
                    enable_wandb=False,
                    enable_tensorboard=False,
                    output_dir=output_training_dir,
                    try_resume_from_last_checkpoint=True,
                    save_final_model=False,
                ),
            )

            train(config)
    assert "unregistered_function_name" in str(exception_info.value)


def test_train_pack():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        config: TrainingConfig = TrainingConfig(
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="debug_pretraining",
                        )
                    ],
                    stream=True,
                    pack=True,
                    use_async_dataset=True,
                ),
            ),
            model=ModelParams(
                model_name="MlpEncoder",
                # The true max length is 1024, but a lower value works. This is done to
                # reduce test runtime.
                model_max_length=128,
                trust_remote_code=True,
                tokenizer_name="gpt2",
                tokenizer_pad_token="<|endoftext|>",
                load_pretrained_weights=False,
            ),
            training=TrainingParams(
                trainer_type=TrainerType.TRL_SFT,
                max_steps=1,
                logging_steps=1,
                enable_wandb=False,
                enable_tensorboard=False,
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
                model_name="MlpEncoder",
                model_max_length=1024,
                trust_remote_code=True,
                tokenizer_name="gpt2",
                tokenizer_pad_token="<|endoftext|>",
                load_pretrained_weights=False,
            ),
            training=TrainingParams(
                per_device_train_batch_size=1,
                trainer_type=TrainerType.TRL_DPO,
                max_steps=3,
                logging_steps=3,
                log_model_summary=True,
                enable_wandb=False,
                enable_tensorboard=False,
                output_dir=output_training_dir,
                try_resume_from_last_checkpoint=False,
                save_final_model=True,
            ),
        )

        train(config)
