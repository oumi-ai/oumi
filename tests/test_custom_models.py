import tempfile
import unittest

from transformers import Trainer

from lema import train
from lema.builders.data import build_dataset
from lema.builders.models import build_model, build_tokenizer
from lema.core.types import (
    DataParams,
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    ModelParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
)


def _get_default_config(output_temp_dir):
    return TrainingConfig(
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
            model_name="MlpEncoder",
            tokenizer_name="gpt2",
            model_max_length=128,
            trust_remote_code=False,
            load_pretrained_weights=False,
            model_kwargs={
                "input_dim": 50257,
                "output_dim": 50257,
            },  # vocab size of GPT2 tokenizer
        ),
        training=TrainingParams(
            trainer_type=TrainerType.HF,
            max_steps=3,
            logging_steps=1,
            enable_wandb=False,
            enable_tensorboard=False,
            output_dir=output_temp_dir,
            include_performance_metrics=False,
            include_alternative_mfu_metrics=True,
        ),
    )


def test_train_native_pt_model_from_api():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        config = _get_default_config(output_temp_dir)

        tokenizer = build_tokenizer(config.model)

        dataset = build_dataset(config, tokenizer, DatasetSplit.TRAIN)

        model = build_model(model_params=config.model)

        training_args = config.training.to_hf()

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()


@unittest.skip("Temporarily disabled. Failing potentially due to network timeout")
def test_train_native_pt_model_from_config():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        config = _get_default_config(output_temp_dir)

        train(config)
