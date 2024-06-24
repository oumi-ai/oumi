import tempfile

from lema import train
from lema.core.types import (
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
        config: TrainingConfig = TrainingConfig(
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="yahma/alpaca-cleaned",
                            preprocessing_function_name="alpaca",
                        )
                    ],
                    target_col="prompt",
                ),
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
                model_max_length=1024,
                trust_remote_code=True,
            ),
            training=TrainingParams(
                trainer_type=TrainerType.TRL_SFT,
                max_steps=3,
                logging_steps=3,
                log_model_summary=True,
                enable_wandb=False,
                enable_tensorboard=False,
                output_dir=output_temp_dir,
                try_resume_from_last_checkpoint=True,
            ),
        )

        train(config)


def test_train_custom():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        config: TrainingConfig = TrainingConfig(
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="yahma/alpaca-cleaned",
                            preprocessing_function_name="alpaca",
                        )
                    ],
                    target_col="prompt",
                ),
            ),
            model=ModelParams(
                model_name="learning-machines/sample",
                tokenizer_name="gpt2",
                model_max_length=1024,
                trust_remote_code=False,
            ),
            training=TrainingParams(
                trainer_type=TrainerType.TRL_SFT,
                max_steps=3,
                logging_steps=3,
                enable_wandb=False,
                enable_tensorboard=False,
                output_dir=output_temp_dir,
                include_performance_metrics=True,
            ),
        )

        train(config)


def test_train_pack():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        config: TrainingConfig = TrainingConfig(
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="Salesforce/wikitext",
                            subset="wikitext-2-raw-v1",
                        )
                    ],
                    stream=True,
                    pack=True,
                    target_col="text",
                ),
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
                # The true max length is 1024, but a lower value works. This is done to
                # reduce test runtime.
                model_max_length=128,
                trust_remote_code=True,
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
