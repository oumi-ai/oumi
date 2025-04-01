"""Integration tests for the VERL PPO implementation in Oumi."""

import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from oumi.core.configs import TrainingConfig
from oumi.core.configs.params.training_params import TrainerType
from oumi.core.configs.params.verl_params import (
    AdvantageEstimator,
    RolloutEngine,
    TrainingStrategy,
)
from oumi.utils.io_utils import load_json
from oumi.utils.torch_utils import device_cleanup
from tests import get_configs_dir
from tests.e2e import get_e2e_test_output_dir, is_file_not_empty
from tests.markers import requires_gpus


@pytest.fixture
def verl_test_models_dir():
    """Returns the path to test models for VERL tests."""
    return Path("/tmp/verl_test_models")


@pytest.fixture
def small_test_config():
    """Returns a small test config for VERL PPO."""
    config = {
        "model": {
            "model_name": "hf-internal-testing/tiny-random-gpt2",
            "model_max_length": 32,
            "torch_dtype_str": "float32",
            "attn_implementation": "eager",
        },
        "data": {
            "train": {
                "datasets": [
                    {
                        "dataset_name": "Anthropic/hh-rlhf",
                        "split": "train",
                        "subset": "harmless-base",
                        "dataset_kwargs": {
                            "input_column": "input",
                            "output_column": "output",
                            "max_rows": 10,
                        },
                    }
                ]
            }
        },
        "training": {
            "trainer_type": "VERL_PPO",
            "save_steps": 5,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "verl_params": {
                "adv_estimator": AdvantageEstimator.GAE,
                "training_strategy": TrainingStrategy.FSDP,
                "rollout_engine": RolloutEngine.TRANSFORMERS,
                "n_gpus_per_node": 1,
                "nnodes": 1,
                "use_reward_model": False,
                "kl_ctrl_type": "fixed",
                "kl_coef": 0.001,
                "extra_args": {
                    "actor_rollout_ref": {
                        "model": {"enable_gradient_checkpointing": False},
                        "actor": {
                            "ppo_mini_batch_size": 2,
                            "ppo_micro_batch_size_per_gpu": 1,
                            "optim": {"lr": 1.0e-5},
                        },
                        "rollout": {
                            "name": "transformers",
                            "tensor_model_parallel_size": 1,
                        },
                    },
                    "critic": {
                        "ppo_micro_batch_size_per_gpu": 1,
                        "optim": {"lr": 1.0e-5},
                    },
                    "trainer": {
                        "project_name": "oumi_verl_test",
                        "experiment_name": "tiny_gpt2_test",
                        "logger": ["console"],
                        "total_epochs": 1,
                        "test_freq": 1,
                    },
                },
            },
            "max_steps": 5,
            "logging_steps": 1,
            "output_dir": str(get_e2e_test_output_dir() / "verl_ppo_test"),
            "enable_wandb": False,
            "enable_tensorboard": False,
        },
    }
    return OmegaConf.create(config)


@patch("ray.init")
@patch("verl.trainer.ppo.ray_trainer.RayPPOTrainer")
@requires_gpus()
def test_verl_ppo_trainer_integration(
    mock_ray_ppo_trainer, mock_ray_init, small_test_config
):
    """Test that the VERL PPO trainer initializes and runs correctly."""

    # Create mock instance for RayPPOTrainer
    mock_trainer_instance = mock_ray_ppo_trainer.return_value
    mock_trainer_instance.init_workers = lambda: None
    mock_trainer_instance.fit = lambda: None

    # Set up output directory
    output_dir = Path(small_test_config.training.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the test config to a temporary file
    config_path = output_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(OmegaConf.to_container(small_test_config), f)

    # Import the necessary modules here to avoid circular dependencies
    from oumi.core.configs import TrainingConfig
    from oumi.train import train

    # Convert OmegaConf to TrainingConfig
    config = TrainingConfig(**OmegaConf.to_container(small_test_config))

    try:
        # Run the training
        train(config)

        # Verify that RayPPOTrainer methods were called
        assert mock_ray_init.called, "Ray.init was not called"
        assert mock_ray_ppo_trainer.called, "RayPPOTrainer constructor was not called"
        assert mock_trainer_instance.init_workers.called, "init_workers was not called"
        assert mock_trainer_instance.fit.called, "fit was not called"

        # Verify initialization args
        trainer_args = mock_ray_ppo_trainer.call_args[1]
        assert "config" in trainer_args, "Missing config in RayPPOTrainer args"
        assert "tokenizer" in trainer_args, "Missing tokenizer in RayPPOTrainer args"
        assert "role_worker_mapping" in trainer_args, (
            "Missing role_worker_mapping in RayPPOTrainer args"
        )
        assert "ray_worker_group_cls" in trainer_args, (
            "Missing ray_worker_group_cls in RayPPOTrainer args"
        )

        verl_config = trainer_args["config"]
        assert verl_config["algorithm"]["adv_estimator"] == "gae", (
            "Incorrect adv_estimator"
        )
        assert verl_config["actor_rollout_ref"]["actor"]["strategy"] == "fsdp", (
            "Incorrect strategy"
        )

    finally:
        # Clean up
        device_cleanup()


@pytest.mark.xfail(reason="This test requires actual setup and runs with real models")
@requires_gpus(2)
def test_verl_ppo_train_with_real_models():
    """End-to-end test of VERL PPO training with real models.

    This test is marked as xfail because it requires actual setup and
    runs with real models. It's included as a reference for manual testing.
    """
    from oumi.train import train

    # Set up output directory
    output_dir = get_e2e_test_output_dir() / "verl_ppo_real_test"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the config
    config_path = get_configs_dir() / "examples" / "verl_ppo" / "train.yaml"

    # Update output directory
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config_dict["training"]["output_dir"] = str(output_dir)
    config_dict["training"]["max_steps"] = 5
    config_dict["training"]["verl_params"]["n_gpus_per_node"] = 2
    config_dict["training"]["enable_wandb"] = False
    config_dict["training"]["enable_tensorboard"] = False

    # Convert to TrainingConfig
    config = TrainingConfig(**config_dict)

    try:
        # Run the training
        train(config)

        # Verify outputs
        assert output_dir.exists(), "Output directory not created"
        checkpoint_dir = output_dir / "checkpoints"
        assert checkpoint_dir.exists(), "Checkpoints directory not created"

    finally:
        # Clean up
        device_cleanup()
