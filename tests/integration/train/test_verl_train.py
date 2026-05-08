import importlib.util
import pathlib
import tempfile
from unittest.mock import patch

import pytest
from datasets import Dataset

from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    ModelParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
)
from oumi.core.configs.params.grpo_params import GrpoParams
from oumi.core.trainers.verl_grpo_trainer import VerlGrpoTrainer
from oumi.utils.packaging import is_verl_v0_7_or_later
from tests.markers import requires_gpus

_verl_available = importlib.util.find_spec("verl") is not None

_MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def _make_countdown_dataset(size: int = 4) -> Dataset:
    """Creates a minimal countdown-style dataset in verl format."""
    records = []
    for i in range(size):
        target = 10 + i
        nums = [5 + i, 5]
        records.append(
            {
                "data_source": "countdown",
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            f"Using the numbers {nums}, create an equation that "
                            f"equals {target}."
                        ),
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": target, "numbers": nums},
                },
                "extra_info": {"split": "train", "index": i},
            }
        )
    return Dataset.from_list(records)


def _make_verl_training_config(output_dir: str, n_gpus: int = 1) -> TrainingConfig:
    """Creates a minimal VERL_GRPO TrainingConfig for testing."""
    return TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                datasets=[
                    DatasetParams(dataset_name="d1shs0ap/countdown"),
                ],
            ),
            validation=DatasetSplitParams(
                datasets=[
                    DatasetParams(dataset_name="d1shs0ap/countdown"),
                ],
            ),
        ),
        model=ModelParams(
            model_name=_MODEL_NAME,
            model_max_length=256,
            trust_remote_code=True,
        ),
        training=TrainingParams(
            trainer_type=TrainerType.VERL_GRPO,
            max_steps=1,
            logging_steps=1,
            enable_wandb=False,
            enable_tensorboard=False,
            enable_mlflow=False,
            output_dir=output_dir,
            save_final_model=False,
            reward_functions=["countdown"],
            grpo=GrpoParams(
                max_completion_length=32,
                max_prompt_length=128,
                temperature=0.7,
                use_vllm=False,
            ),
            # Override verl-specific settings for fast testing.
            verl_config_overrides={
                "trainer": {
                    "n_gpus_per_node": n_gpus,
                    "nnodes": 1,
                },
                "actor_rollout_ref": {
                    "rollout": {
                        "log_prob_micro_batch_size_per_gpu": 1,
                        "micro_batch_size_per_gpu": 1,
                    },
                    "actor": {
                        "ppo_micro_batch_size_per_gpu": 1,
                    },
                    "ref": {
                        "log_prob_micro_batch_size_per_gpu": 1,
                    },
                },
                "critic": {
                    "ppo_micro_batch_size_per_gpu": 1,
                },
                "data": {
                    "train_batch_size": 2,
                    "val_batch_size": 2,
                    "max_prompt_length": 128,
                    "max_response_length": 32,
                },
            },
        ),
    )


def _create_trainer_no_ray(
    config: TrainingConfig,
    train_ds: Dataset,
    eval_ds: Dataset,
    cache_dir: str | pathlib.Path | None = None,
) -> VerlGrpoTrainer:
    """Instantiates VerlGrpoTrainer with Ray/GPU setup mocked out."""
    from oumi.builders import build_reward_functions, build_tokenizer

    tokenizer = build_tokenizer(config.model)
    reward_fns = build_reward_functions(config.training)

    with patch.object(VerlGrpoTrainer, "_setup_verl_trainer"):
        trainer = VerlGrpoTrainer(
            processing_class=tokenizer,
            config=config,
            reward_funcs=reward_fns,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            cache_dir=cache_dir,
        )
    return trainer


@pytest.mark.skipif(not _verl_available, reason="verl is not installed")
class TestVerlConfigAndDatasetPipeline:
    """Tests verl config creation and dataset pipeline without requiring GPUs.

    These tests exercise the config translation (Oumi -> verl DictConfig) and
    dataset-to-parquet conversion, which are the most common sources of breakage
    when verl or oumi configs change.
    """

    def test_create_config(self):
        """Verifies Oumi TrainingConfig is correctly translated to verl DictConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_verl_training_config(tmpdir)
            trainer = _create_trainer_no_ray(
                config,
                _make_countdown_dataset(4),
                _make_countdown_dataset(2),
                cache_dir=pathlib.Path(tmpdir) / "cache",
            )
            verl_config = trainer._create_config()

            assert verl_config.algorithm.adv_estimator == "grpo"
            assert verl_config.actor_rollout_ref.model.path == _MODEL_NAME
            assert verl_config.actor_rollout_ref.rollout.name == "hf"
            assert verl_config.data.max_response_length == 32
            assert verl_config.trainer.n_gpus_per_node == 1
            assert verl_config.trainer.nnodes == 1
            assert verl_config.trainer.total_training_steps == 1

            if is_verl_v0_7_or_later():
                assert (
                    verl_config.reward.custom_reward_function.name == "countdown_reward"
                )

    def test_create_dataset_files(self):
        """Verifies datasets are properly converted to parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_verl_training_config(tmpdir)
            cache_dir = pathlib.Path(tmpdir) / "cache"
            trainer = _create_trainer_no_ray(
                config,
                _make_countdown_dataset(4),
                _make_countdown_dataset(2),
                cache_dir=cache_dir,
            )

            train_path = pathlib.Path(trainer._train_filepath)
            val_path = pathlib.Path(trainer._val_filepath)
            assert train_path.exists(), f"Train parquet not found: {train_path}"
            assert val_path.exists(), f"Val parquet not found: {val_path}"
            assert train_path.stat().st_size > 0
            assert val_path.stat().st_size > 0

    def test_create_config_with_vllm_rollout(self):
        """Verifies vLLM rollout config is set correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_verl_training_config(tmpdir)
            config.training.grpo.use_vllm = True
            config.training.grpo.vllm_gpu_memory_utilization = 0.85

            trainer = _create_trainer_no_ray(
                config,
                _make_countdown_dataset(4),
                _make_countdown_dataset(2),
                cache_dir=pathlib.Path(tmpdir) / "cache",
            )
            verl_config = trainer._create_config()

            assert verl_config.actor_rollout_ref.rollout.name == "vllm"
            assert verl_config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.85

    def test_conversation_dataset_pipeline(self):
        """Verifies conversation-format datasets are properly detected and converted."""
        from oumi.core.types.conversation import Conversation, Message, Role

        conversations = []
        for i in range(4):
            conv = Conversation(
                messages=[
                    Message(role=Role.USER, content=f"What is {i} + {i}?"),
                    Message(role=Role.ASSISTANT, content=str(i + i)),
                ]
            )
            conversations.append({"conversation_json": conv.to_json()})

        train_ds = Dataset.from_list(conversations)
        eval_ds = Dataset.from_list(conversations[:2])

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_verl_training_config(tmpdir)
            cache_dir = pathlib.Path(tmpdir) / "cache"
            trainer = _create_trainer_no_ray(
                config, train_ds, eval_ds, cache_dir=cache_dir
            )

            train_path = pathlib.Path(trainer._train_filepath)
            assert train_path.exists()

            import pandas as pd

            df = pd.read_parquet(train_path)
            assert "prompt" in df.columns
            assert "data_source" in df.columns
            assert len(df) == 4


@requires_gpus(count=1, min_gb=40.0)
@pytest.mark.skipif(not _verl_available, reason="verl is not installed")
def test_verl_grpo_train_1_step():
    """End-to-end verl GRPO training for 1 step on a single GPU.

    Uses Qwen2.5-0.5B with the countdown dataset and vLLM rollout.
    Requires ~40GB GPU (L40S/A100) due to FSDP actor + vLLM coexisting.
    """
    from oumi import train

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = str(pathlib.Path(tmpdir) / "train")
        config = TrainingConfig(
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[DatasetParams(dataset_name="d1shs0ap/countdown")],
                ),
                validation=DatasetSplitParams(
                    datasets=[DatasetParams(dataset_name="d1shs0ap/countdown")],
                ),
            ),
            model=ModelParams(
                model_name=_MODEL_NAME,
                model_max_length=512,
                trust_remote_code=True,
            ),
            training=TrainingParams(
                trainer_type=TrainerType.VERL_GRPO,
                max_steps=1,
                logging_steps=1,
                enable_wandb=False,
                enable_tensorboard=False,
                enable_mlflow=False,
                output_dir=output_dir,
                save_final_model=False,
                reward_functions=["countdown"],
                grpo=GrpoParams(
                    max_completion_length=64,
                    max_prompt_length=256,
                    num_generations=2,
                    temperature=0.7,
                    use_vllm=True,
                    vllm_gpu_memory_utilization=0.4,
                ),
                verl_config_overrides={
                    "trainer": {
                        "n_gpus_per_node": 1,
                        "nnodes": 1,
                        "val_before_train": False,
                    },
                    "actor_rollout_ref": {
                        "actor": {
                            "ppo_micro_batch_size_per_gpu": 8,
                            "strategy": "fsdp",
                        },
                        "ref": {
                            "log_prob_micro_batch_size_per_gpu": 8,
                        },
                        "rollout": {
                            "log_prob_micro_batch_size_per_gpu": 8,
                            "tensor_model_parallel_size": 1,
                            "n": 2,
                        },
                    },
                    "critic": {
                        "ppo_micro_batch_size_per_gpu": 8,
                        "strategy": "fsdp",
                    },
                    "data": {
                        "train_batch_size": 8,
                        "val_batch_size": 8,
                        "max_prompt_length": 256,
                        "max_response_length": 64,
                    },
                    "reward": {
                        "num_workers": 1,
                    },
                },
            ),
        )

        train(config)

        verl_output = pathlib.Path(output_dir) / "verl_output"
        assert verl_output.exists(), f"verl_output dir not created: {verl_output}"
        assert any(verl_output.iterdir()), f"verl_output dir is empty: {verl_output}"
