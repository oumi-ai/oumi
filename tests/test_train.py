import tempfile

import pytest

from lema import train
from lema.core.types import (
    DataParams,
    ModelParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
)


def test_basic_train():
    output_temp_dir = tempfile.mkdtemp()

    config: TrainingConfig = TrainingConfig(
        data=DataParams(
            dataset_name="yahma/alpaca-cleaned",
            preprocessing_function_name="alpaca",
            text_col="prompt",
        ),
        model=ModelParams(
            model_name="openai-community/gpt2",
            trust_remote_code=True,
        ),
        training=TrainingParams(
            trainer_type=TrainerType.TRL_SFT,
            max_steps=3,
            logging_steps=3,
            enable_wandb=False,
            enable_tensorboard=False,
            output_dir=output_temp_dir,
        ),
    )

    train(config)


def test_custom_train():
    output_temp_dir = tempfile.mkdtemp()

    config: TrainingConfig = TrainingConfig(
        data=DataParams(
            dataset_name="yahma/alpaca-cleaned",
            preprocessing_function_name="alpaca",
            text_col="prompt",
        ),
        model=ModelParams(
            model_name="learning-machines/sample",
            tokenizer_name="gpt2",
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


# Currently takes a long time to run because packing is very slow.
# TODO: Change `skip` to `e2e` after #62 is fixed.
@pytest.mark.skip
def test_pack_train():
    output_temp_dir = tempfile.mkdtemp()

    config: TrainingConfig = TrainingConfig(
        data=DataParams(
            dataset_name="Salesforce/wikitext",
            dataset_config="wikitext-2-raw-v1",
            stream=True,
            pack=True,
            text_col="text",
        ),
        model=ModelParams(
            model_name="openai-community/gpt2",
            model_max_length=1024,
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
