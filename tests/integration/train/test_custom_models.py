import tempfile

from transformers import Trainer

from oumi import train
from oumi.builders.data import build_dataset_mixture
from oumi.builders.models import (
    build_model,
    build_tokenizer,
    is_image_text_llm,
)
from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    ModelParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
)
from oumi.core.configs.internal.supported_models import (
    is_custom_model,
)


def _get_default_config(output_temp_dir):
    return TrainingConfig(
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
            model_name="MlpEncoder",
            tokenizer_name="gpt2",
            model_max_length=128,
            trust_remote_code=False,
            load_pretrained_weights=False,
            model_kwargs={
                "input_dim": 50257,
                "output_dim": 50257,
            },  # vocab size of GPT2 tokenizer
            tokenizer_pad_token="<|endoftext|>",
        ),
        training=TrainingParams(
            trainer_type=TrainerType.HF,
            max_steps=3,
            logging_steps=1,
            enable_wandb=False,
            enable_tensorboard=False,
            enable_mlflow=False,
            output_dir=output_temp_dir,
            include_performance_metrics=False,
            include_alternative_mfu_metrics=True,
        ),
    )


def test_train_native_pt_model_from_api():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        config: TrainingConfig = _get_default_config(output_temp_dir)

        assert is_custom_model(config.model.model_name), f"ModelParams: {config.model}"
        assert not is_image_text_llm(config.model), f"ModelParams: {config.model}"

        tokenizer = build_tokenizer(config.model)

        dataset = build_dataset_mixture(
            config.data,
            tokenizer,
            DatasetSplit.TRAIN,
            seq_length=config.model.model_max_length,
        )

        model = build_model(model_params=config.model)

        training_args = config.training.to_hf()

        trainer = Trainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()


def test_train_native_pt_model_from_config():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        config = _get_default_config(output_temp_dir)

        assert is_custom_model(config.model.model_name), f"ModelParams: {config.model}"
        assert not is_image_text_llm(config.model), f"ModelParams: {config.model}"

        train(config)
