import tempfile

from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    ModelParams,
    TuningConfig,
    TuningParams,
)
from oumi.tune import tune


def test_tune():
    """Test tuning multiple hyperparameters simultaneously."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        config = TuningConfig(
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="debug_sft",
                            dataset_kwargs={"dataset_size": 10},
                        )
                    ]
                ),
                validation=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="debug_sft", dataset_kwargs={"dataset_size": 5}
                        )
                    ]
                ),
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
                model_max_length=1024,
                trust_remote_code=True,
                tokenizer_pad_token="<|endoftext|>",
            ),
            tuning=TuningParams(
                tunable_training_params={
                    "learning_rate": {
                        "type": "loguniform",
                        "low": 1e-5,
                        "high": 1e-4,
                    },
                    "per_device_train_batch_size": {
                        "type": "categorical",
                        "choices": [2, 4],
                    },
                },
                fixed_training_params={
                    "max_steps": 2,
                    "enable_wandb": False,
                    "enable_tensorboard": False,
                    "enable_mlflow": False,
                },
                tunable_peft_params={
                    "lora_r": {
                        "type": "categorical",
                        "choices": [4, 8],
                    },
                    "lora_alpha": {
                        "type": "categorical",
                        "choices": [16, 32],
                    },
                },
                output_dir=output_temp_dir,
                n_trials=2,
            ),
        )

        tune(config)


def test_tune_multi_objective():
    """Test multi-objective hyperparameter optimization."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        config = TuningConfig(
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="debug_sft",
                            dataset_kwargs={"dataset_size": 10},
                        )
                    ]
                ),
                validation=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="debug_sft", dataset_kwargs={"dataset_size": 5}
                        )
                    ]
                ),
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
                model_max_length=1024,
                trust_remote_code=True,
                tokenizer_pad_token="<|endoftext|>",
            ),
            tuning=TuningParams(
                tunable_training_params={
                    "learning_rate": {
                        "type": "categorical",
                        "choices": [1e-5, 5e-5],
                    },
                },
                fixed_training_params={
                    "max_steps": 2,
                    "enable_wandb": False,
                    "enable_tensorboard": False,
                    "enable_mlflow": False,
                },
                output_dir=output_temp_dir,
                evaluation_metrics=["eval_loss", "eval_mean_token_accuracy"],
                evaluation_direction=["minimize", "maximize"],
                n_trials=2,
            ),
        )

        tune(config)
