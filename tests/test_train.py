import os
import pytest
import tempfile

from lema.core.types import DataParams
from lema.core.types import ModelParams
from lema.core.types import TrainingParams
from lema.core.types import TrainingConfig


from lema import train


def test_basic_train():
    output_temp_dir = tempfile.mkdtemp()

    config: TrainingConfig = TrainingConfig(
        data=DataParams(
            dataset_name="yahma/alpaca-cleaned",
            preprocessing_function_name="alpaca",
            trainer_kwargs={
                "dataset_text_field": "prompt",
            },
        ),
        model=ModelParams(
            model_name="openai-community/gpt2",
            trust_remote_code=True,
        ),
        training=TrainingParams(
            max_steps=5,
            logging_steps=5,
            enable_wandb=False,
            enable_tensorboard=False,
            output_dir=output_temp_dir,
        ),
    )

    train(config)
