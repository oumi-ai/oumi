"""Unit tests for BaseModel save_pretrained and from_pretrained functionality."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from oumi.builders.models import build_model
from oumi.core.configs import ModelParams
from oumi.core.models.base_model import BaseModel
from oumi.models.mlp import MLPEncoder


def test_save_and_load_pretrained_mlp():
    """Test saving and loading a custom MLP model."""
    # Creating a model with specific parameters
    input_dim = 100
    hidden_dim = 64
    output_dim = 10

    model_params = ModelParams(
        model_name="MLPEncoder",
        load_pretrained_weights=False,
        model_kwargs={
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
        },
    )

    # Building the original model
    original_model = build_model(model_params)
    assert isinstance(original_model, MLPEncoder)
    assert isinstance(original_model, BaseModel)

    # Get original weights for comparison
    original_state_dict = original_model.state_dict()

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlp_checkpoint"

        # Saving the model
        original_model.save_pretrained(save_dir)

        # Verifying files were created
        assert (save_dir / "model.safetensors").exists()
        assert (save_dir / "config.json").exists()

        # Verify config content
        with open(save_dir / "config.json") as f:
            config = json.load(f)
        assert config["model_type"] == "MLPEncoder"
        assert config["init_kwargs"]["input_dim"] == input_dim
        assert config["init_kwargs"]["hidden_dim"] == hidden_dim
        assert config["init_kwargs"]["output_dim"] == output_dim

        # Loading the model using from_pretrained
        loaded_model = MLPEncoder.from_pretrained(save_dir)
        assert isinstance(loaded_model, MLPEncoder)

        # Verifying loaded weights match original
        loaded_state_dict = loaded_model.state_dict()
        assert set(original_state_dict.keys()) == set(loaded_state_dict.keys())

        for key in original_state_dict.keys():
            assert torch.allclose(original_state_dict[key], loaded_state_dict[key]), (
                f"Mismatch in parameter: {key}"
            )


def test_save_and_load_pretrained_via_build_model():
    """Test loading a pretrained custom model using build_model."""
    # Creating and saving a model
    input_dim = 50
    hidden_dim = 32
    output_dim = 5

    original_params = ModelParams(
        model_name="MLPEncoder",
        load_pretrained_weights=False,
        model_kwargs={
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
        },
    )

    model = build_model(original_params)
    assert isinstance(model, BaseModel)
    original_model = model
    original_state_dict = original_model.state_dict()

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlp_checkpoint"
        original_model.save_pretrained(save_dir)

        # Loading using build_model with custom_pretrained_dir
        load_params = ModelParams(
            model_name="MLPEncoder",
            load_pretrained_weights=True,
            custom_pretrained_dir=str(save_dir),
        )

        loaded_model = build_model(load_params)

        # Verifying loaded weights match
        loaded_state_dict = loaded_model.state_dict()
        assert set(original_state_dict.keys()) == set(loaded_state_dict.keys())

        for key in original_state_dict.keys():
            assert torch.allclose(original_state_dict[key], loaded_state_dict[key]), (
                f"Mismatch in parameter: {key}"
            )


def test_load_pretrained_with_override_kwargs():
    """Test loading with override_kwargs to modify initialization parameters."""
    # Creating and saving a model with original dimensions
    original_input_dim = 100
    original_hidden_dim = 64
    original_output_dim = 10

    original_model = MLPEncoder(
        input_dim=original_input_dim,
        hidden_dim=original_hidden_dim,
        output_dim=original_output_dim,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlp_checkpoint"
        original_model.save_pretrained(save_dir)

        # Loading with same dimensions - should work
        loaded_model = MLPEncoder.from_pretrained(
            save_dir,
            override_kwargs={
                "input_dim": original_input_dim,
                "hidden_dim": original_hidden_dim,
                "output_dim": original_output_dim,
            },
        )
        assert isinstance(loaded_model, MLPEncoder)


def test_inference_with_loaded_model():
    """Test that loaded model produces correct outputs."""
    input_dim = 50
    hidden_dim = 32
    output_dim = 10

    # Creating and saving a model
    original_model = MLPEncoder(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
    )

    # Creating test input
    batch_size = 2
    seq_length = 5
    test_input = torch.randint(0, input_dim, (batch_size, seq_length))

    # Getting original output
    original_model.eval()
    with torch.no_grad():
        original_output = original_model(input_ids=test_input)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlp_checkpoint"
        original_model.save_pretrained(save_dir)

        # Loading model
        loaded_model = MLPEncoder.from_pretrained(save_dir)
        loaded_model.eval()

        # Getting loaded model output
        with torch.no_grad():
            loaded_output = loaded_model(input_ids=test_input)

        # Outputs should be identical
        assert set(original_output.keys()) == set(loaded_output.keys())
        assert torch.allclose(
            original_output["logits"], loaded_output["logits"], atol=1e-6
        )


def test_save_pretrained_creates_directory():
    """Test that save_pretrained creates the directory if it doesn't exist."""
    model = MLPEncoder(input_dim=10, hidden_dim=8, output_dim=5)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "nested" / "path" / "to" / "model"
        assert not save_dir.exists()

        model.save_pretrained(save_dir)

        assert save_dir.exists()
        assert (save_dir / "model.safetensors").exists()
        assert (save_dir / "config.json").exists()


def test_from_pretrained_missing_weights_file():
    """Test that from_pretrained raises error when weights file is missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "empty_dir"
        save_dir.mkdir()

        with pytest.raises(
            FileNotFoundError, match="Pretrained weights file not found"
        ):
            MLPEncoder.from_pretrained(save_dir)


def test_from_pretrained_without_config():
    """Test loading a model when config file is missing but override_kwargs provided."""
    input_dim = 20
    hidden_dim = 16
    output_dim = 5

    # Creating and saving model
    model = MLPEncoder(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlp_checkpoint"
        model.save_pretrained(save_dir)

        # Removing config file
        (save_dir / "config.json").unlink()

        # Should work with override_kwargs
        loaded_model = MLPEncoder.from_pretrained(
            save_dir,
            override_kwargs={
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
            },
        )
        assert isinstance(loaded_model, MLPEncoder)


def test_from_pretrained_strict_mode():
    """Test strict mode when loading state dict."""
    input_dim = 30
    hidden_dim = 20
    output_dim = 10

    # Creating and saving a model
    model = MLPEncoder(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlp_checkpoint"
        model.save_pretrained(save_dir)

        # Loading with correct dimensions should work in strict mode
        loaded_model = MLPEncoder.from_pretrained(save_dir, strict=True)
        assert isinstance(loaded_model, MLPEncoder)


def test_build_model_load_pretrained_missing_custom_pretrained_dir():
    """Test build_model raises error when load_pretrained_weights=True."""
    params = ModelParams(
        model_name="MLPEncoder",
        load_pretrained_weights=True,
        # Missing custom_pretrained_dir
        model_kwargs={
            "input_dim": 10,
            "hidden_dim": 8,
            "output_dim": 5,
        },
    )

    with pytest.raises(ValueError, match="requires either"):
        build_model(params)


def test_save_pretrained_custom_filenames():
    """Test saving with custom filenames."""
    model = MLPEncoder(input_dim=10, hidden_dim=8, output_dim=5)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "custom_names"
        custom_weights = "custom_weights.safetensors"
        custom_config = "custom_config.json"

        model.save_pretrained(
            save_dir, weights_filename=custom_weights, config_filename=custom_config
        )

        assert (save_dir / custom_weights).exists()
        assert (save_dir / custom_config).exists()

        # Loading with custom filenames
        loaded_model = MLPEncoder.from_pretrained(
            save_dir, weights_filename=custom_weights, config_filename=custom_config
        )
        assert isinstance(loaded_model, MLPEncoder)


def test_save_pretrained_without_config():
    """Test saving without config file."""
    model = MLPEncoder(input_dim=10, hidden_dim=8, output_dim=5)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "no_config"

        model.save_pretrained(save_dir, save_config=False)

        assert (save_dir / "model.safetensors").exists()
        assert not (save_dir / "config.json").exists()
