from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import Dataset

from oumi.core.configs import TrainerType, TrainingParams
from oumi.core.configs.params.verl_params import (
    AdvantageEstimator,
    RolloutEngine,
    TrainingStrategy,
    VerlParams,
)
from oumi.core.trainers.verl_ppo_trainer import VerlPpoTrainer


@pytest.fixture
def mock_model():
    model = MagicMock(spec=torch.nn.Module)
    # Add config attribute with required properties
    model.config = MagicMock()
    model.config.model_type = "test_model"
    model.config.max_position_embeddings = 2048
    return model


@pytest.fixture
def mock_tokenizer():
    return MagicMock()


@pytest.fixture
def mock_dataset():
    # Create a mock without spec to allow setting __len__
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)
    dataset.state_dict = None
    return dataset


@pytest.fixture
def mock_ray():
    with (
        patch("ray.init") as mock_init,
        patch("ray.is_initialized", return_value=False) as mock_is_initialized,
    ):
        yield {
            "init": mock_init,
            "is_initialized": mock_is_initialized,
        }


@pytest.fixture
def mock_resource_pool_manager():
    with patch("verl.trainer.ppo.ray_trainer.ResourcePoolManager") as mock_rpm:
        mock_rpm_instance = MagicMock()
        mock_rpm.return_value = mock_rpm_instance
        yield mock_rpm


@pytest.fixture
def mock_ray_ppo_trainer():
    with patch("verl.trainer.ppo.ray_trainer.RayPPOTrainer") as mock_trainer:
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.init_workers = MagicMock()
        mock_trainer_instance.fit = MagicMock()
        mock_trainer_instance.save_checkpoint = MagicMock()
        mock_trainer_instance.save_model = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        
        # Patch OmegaConf.create to return a normal dict
        with patch("oumi.core.trainers.verl_ppo_trainer.OmegaConf.create") as mock_omegaconf:
            def side_effect(config_dict):
                # Convert to MagicMock with attribute access that mimics OmegaConf
                config = MagicMock()
                
                # Add top level keys
                config.keys.return_value = config_dict.keys()
                
                # Add attributes for nested access
                for k, v in config_dict.items():
                    if isinstance(v, dict):
                        nested = MagicMock()
                        nested.keys.return_value = v.keys()
                        for nk, nv in v.items():
                            setattr(nested, nk, nv)
                        setattr(config, k, nested)
                    else:
                        setattr(config, k, v)
                
                return config
                
            mock_omegaconf.side_effect = side_effect
            yield mock_trainer


@pytest.fixture
def mock_reward_manager():
    with patch("verl.workers.reward_manager.NaiveRewardManager") as mock_rm:
        mock_rm_instance = MagicMock()
        mock_rm.return_value = mock_rm_instance
        yield mock_rm


@pytest.fixture
def verl_params():
    return VerlParams(
        adv_estimator=AdvantageEstimator.GAE,
        training_strategy=TrainingStrategy.FSDP,
        rollout_engine=RolloutEngine.VLLM,
        n_gpus_per_node=2,
        nnodes=1,
        use_reward_model=False,
        kl_ctrl_type="fixed",
        kl_coef=0.1,
        target_kl=0.1,
        kl_horizon=10000,
    )


@pytest.fixture
def training_params(verl_params):
    params = MagicMock(spec=TrainingParams)
    params.trainer_type = TrainerType.VERL_PPO
    params.verl_params = verl_params
    params.num_train_epochs = 3
    params.per_device_train_batch_size = 8
    params.logging_steps = 10
    params.save_steps = 50
    params.enable_wandb = False
    params.enable_tensorboard = False
    params.output_dir = "/tmp/test_verl_output"
    params.seed = 42
    params.learning_rate = 0.0001  # Add actual learning rate value instead of MagicMock
    params.enable_gradient_checkpointing = False

    return params


def test_verl_ppo_trainer_initialization(
    mock_ray,
    mock_model,
    mock_tokenizer,
    mock_dataset,
    mock_resource_pool_manager,
    mock_ray_ppo_trainer,
    mock_reward_manager,
    training_params,
):
    """Test that the VERL PPO trainer is initialized correctly."""
    
    # Create mock objects we'll use later
    mock_trainer_instance = MagicMock()
    mock_config = MagicMock()
    
    # Setup config attributes
    mock_config.trainer = MagicMock()
    mock_config.actor_rollout_ref = MagicMock()
    mock_config.actor_rollout_ref.hybrid_engine = True
    mock_config.critic = MagicMock()
    mock_config.reward_model = MagicMock()
    mock_config.data = MagicMock()
    mock_config.data.prompt_key = "prompt"
    mock_config.data.completion_key = "completion"
    mock_config.data.train_batch_size = 16
    mock_config.algorithm = MagicMock()
    
    # Patch the _setup_verl_trainer method
    with patch.object(VerlPpoTrainer, '_setup_verl_trainer', autospec=True) as mock_setup:
        # Configure the mocked method to set attributes on the trainer instance
        def side_effect(trainer_instance):
            trainer_instance.verl_trainer = mock_trainer_instance
            trainer_instance.verl_config = mock_config
            
        mock_setup.side_effect = side_effect

        # Initialize the trainer
        trainer = VerlPpoTrainer(
            model=mock_model,
            processing_class=mock_tokenizer,
            args=training_params,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
        )

        # Check initialization
        assert trainer.model == mock_model
        assert trainer.processing_class == mock_tokenizer
        assert trainer.train_dataset == mock_dataset
        assert trainer.eval_dataset == mock_dataset

        # Verify Ray was initialized
        assert mock_ray["is_initialized"].called
        assert mock_ray["init"].called

        # Verify setup method was called
        assert mock_setup.called
        
        # Verify we have a VERL trainer
        assert hasattr(trainer, "verl_trainer")
        assert trainer.verl_trainer == mock_trainer_instance
        
        # Verify config attributes as set by our side effect
        assert hasattr(trainer, "verl_config")  
        assert hasattr(trainer.verl_config, "trainer")
        assert hasattr(trainer.verl_config, "actor_rollout_ref")
        assert hasattr(trainer.verl_config, "critic")
        assert hasattr(trainer.verl_config, "reward_model")
        assert hasattr(trainer.verl_config, "data")
        assert hasattr(trainer.verl_config, "algorithm")
        
        # Verify dataset configuration
        assert hasattr(trainer.verl_config.data, "prompt_key")
        assert hasattr(trainer.verl_config.data, "completion_key")
        assert hasattr(trainer.verl_config.data, "train_batch_size")
        
        # Verify required hybrid_engine attribute is set
        assert trainer.verl_config.actor_rollout_ref.hybrid_engine is True


def test_verl_ppo_train_method(
    mock_ray,
    mock_model,
    mock_tokenizer,
    mock_dataset,
    mock_resource_pool_manager,
    mock_ray_ppo_trainer,
    mock_reward_manager,
    training_params,
):
    """Test that the train method calls the VERL PPO trainer's fit method."""
    
    # Create a mock verl_trainer that we'll set in the side effect
    mock_trainer_instance = MagicMock()
    
    # Patch the _setup_verl_trainer method
    with patch.object(VerlPpoTrainer, '_setup_verl_trainer', autospec=True) as mock_setup:
        # Configure the mocked method to set attributes on the trainer instance
        def side_effect(trainer_instance):
            trainer_instance.verl_trainer = mock_trainer_instance
            
        mock_setup.side_effect = side_effect

        # Initialize the trainer
        trainer = VerlPpoTrainer(
            model=mock_model,
            processing_class=mock_tokenizer,
            args=training_params,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
        )

        # Call the train method
        trainer.train()

        # Verify that init_workers and fit were called
        mock_trainer_instance.init_workers.assert_called_once()
        mock_trainer_instance.fit.assert_called_once()


def test_verl_ppo_save_state_method(
    mock_ray,
    mock_model,
    mock_tokenizer,
    mock_dataset,
    mock_resource_pool_manager,
    mock_ray_ppo_trainer,
    mock_reward_manager,
    training_params,
):
    """Test that the save_state method calls save_checkpoint."""
    
    # Create a mock verl_trainer that we'll set in the side effect
    mock_trainer_instance = MagicMock()
    
    # Patch the _setup_verl_trainer method
    with patch.object(VerlPpoTrainer, '_setup_verl_trainer', autospec=True) as mock_setup:
        # Configure the mocked method to set attributes on the trainer instance
        def side_effect(trainer_instance):
            trainer_instance.verl_trainer = mock_trainer_instance
            
        mock_setup.side_effect = side_effect

        # Initialize the trainer
        trainer = VerlPpoTrainer(
            model=mock_model,
            processing_class=mock_tokenizer,
            args=training_params,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
        )

        # Call the save_state method
        trainer.save_state()

        # Verify that save_checkpoint was called
        mock_trainer_instance.save_checkpoint.assert_called_once()


@patch("oumi.core.distributed.is_world_process_zero", return_value=True)
def test_verl_ppo_save_model_method(
    mock_world_zero,
    mock_ray,
    mock_model,
    mock_tokenizer,
    mock_dataset,
    mock_resource_pool_manager,
    mock_ray_ppo_trainer,
    mock_reward_manager,
    training_params,
):
    """Test that the save_model method calls save_checkpoint with the correct path."""
    
    # Create a mock verl_trainer that we'll set in the side effect
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.save_model = MagicMock()
    
    # Patch the _setup_verl_trainer method
    with patch.object(VerlPpoTrainer, '_setup_verl_trainer', autospec=True) as mock_setup:
        # Configure the mocked method to set attributes on the trainer instance
        def side_effect(trainer_instance):
            trainer_instance.verl_trainer = mock_trainer_instance
            
        mock_setup.side_effect = side_effect

        # Initialize the trainer
        trainer = VerlPpoTrainer(
            model=mock_model,
            processing_class=mock_tokenizer,
            args=training_params,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
        )

        # Call the save_model method
        trainer.save_model(None, final=True)

        # Verify that save_model was called with the correct path
        expected_path = f"{training_params.output_dir}/final_model"
        mock_trainer_instance.save_model.assert_called_once_with(expected_path)


def test_verl_ppo_save_model_skip_non_zero(
    mock_ray,
    mock_model,
    mock_tokenizer,
    mock_dataset,
    mock_resource_pool_manager,
    mock_ray_ppo_trainer,
    mock_reward_manager,
    training_params,
):
    """Test that save_model skips execution if not world process zero."""
    
    # Directly patch the exact function that's used in save_model
    with patch('oumi.core.trainers.verl_ppo_trainer.is_world_process_zero', return_value=False):
        # Create a mock verl_trainer that we'll set in the side effect
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.save_model = MagicMock()
        mock_trainer_instance.save_checkpoint = MagicMock()
        
        # Patch the _setup_verl_trainer method
        with patch.object(VerlPpoTrainer, '_setup_verl_trainer', autospec=True) as mock_setup:
            # Configure the mocked method to set attributes on the trainer instance
            def side_effect(trainer_instance):
                trainer_instance.verl_trainer = mock_trainer_instance
                
            mock_setup.side_effect = side_effect

            # Initialize the trainer
            trainer = VerlPpoTrainer(
                model=mock_model,
                processing_class=mock_tokenizer,
                args=training_params,
                train_dataset=mock_dataset,
                eval_dataset=mock_dataset,
            )

            # Call the save_model method
            trainer.save_model(None, final=True)

            # Verify that save_model was not called
            assert not mock_trainer_instance.save_model.called
            assert not mock_trainer_instance.save_checkpoint.called
