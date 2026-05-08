"""Unit tests for BaseModel save_pretrained and from_pretrained."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from oumi.builders.models import build_model
from oumi.core.configs import ModelParams
from oumi.core.models.base_model import BaseModel
from oumi.models.cnn_classifier import CNNClassifier
from oumi.models.mlp import MLPEncoder


def test_save_and_load_pretrained_mlp():
    model_params = ModelParams(
        model_name="MLPEncoder",
        load_pretrained_weights=False,
        model_kwargs={"input_dim": 100, "hidden_dim": 64, "output_dim": 10},
    )
    original_model = build_model(model_params)
    assert isinstance(original_model, MLPEncoder)
    assert isinstance(original_model, BaseModel)

    original_state_dict = original_model.state_dict()

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlp_checkpoint"
        original_model.save_pretrained(save_dir)

        assert (save_dir / "model.safetensors").exists()
        assert (save_dir / "config.json").exists()

        with open(save_dir / "config.json") as f:
            config = json.load(f)
        assert config["model_type"] == "MLPEncoder"
        assert config["init_kwargs"]["input_dim"] == 100

        loaded_model = MLPEncoder.from_pretrained(save_dir)
        assert isinstance(loaded_model, MLPEncoder)

        loaded_state_dict = loaded_model.state_dict()
        assert set(original_state_dict.keys()) == set(loaded_state_dict.keys())
        for key in original_state_dict.keys():
            assert torch.allclose(original_state_dict[key], loaded_state_dict[key])


def test_save_and_load_pretrained_via_build_model():
    original_params = ModelParams(
        model_name="MLPEncoder",
        load_pretrained_weights=False,
        model_kwargs={"input_dim": 50, "hidden_dim": 32, "output_dim": 5},
    )
    original_model = build_model(original_params)
    original_state_dict = original_model.state_dict()

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlp_checkpoint"
        original_model.save_pretrained(save_dir)  # type: ignore[union-attr]

        load_params = ModelParams(
            model_name=str(save_dir),
            load_pretrained_weights=True,
        )
        loaded_model = build_model(load_params)

        loaded_state_dict = loaded_model.state_dict()
        assert set(original_state_dict.keys()) == set(loaded_state_dict.keys())
        for key in original_state_dict.keys():
            assert torch.allclose(original_state_dict[key], loaded_state_dict[key])


def test_load_pretrained_with_override_kwargs():
    original_model = MLPEncoder(input_dim=100, hidden_dim=64, output_dim=10)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlp_checkpoint"
        original_model.save_pretrained(save_dir)

        loaded_model = MLPEncoder.from_pretrained(
            save_dir,
            override_kwargs={"input_dim": 100, "hidden_dim": 64, "output_dim": 10},
        )
        assert isinstance(loaded_model, MLPEncoder)


def test_inference_with_loaded_model():
    original_model = MLPEncoder(input_dim=50, hidden_dim=32, output_dim=10)
    test_input = torch.randint(0, 50, (2, 5))

    original_model.train(False)
    with torch.no_grad():
        original_output = original_model(input_ids=test_input)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlp_checkpoint"
        original_model.save_pretrained(save_dir)

        loaded_model = MLPEncoder.from_pretrained(save_dir)
        loaded_model.train(False)

        with torch.no_grad():
            loaded_output = loaded_model(input_ids=test_input)

        assert torch.allclose(
            original_output["logits"], loaded_output["logits"], atol=1e-6
        )


def test_save_pretrained_creates_directory():
    model = MLPEncoder(input_dim=10, hidden_dim=8, output_dim=5)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "nested" / "path" / "to" / "model"
        assert not save_dir.exists()

        model.save_pretrained(save_dir)

        assert save_dir.exists()
        assert (save_dir / "model.safetensors").exists()
        assert (save_dir / "config.json").exists()


def test_from_pretrained_missing_weights_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "empty_dir"
        save_dir.mkdir()

        with pytest.raises(
            FileNotFoundError, match="Pretrained weights file not found"
        ):
            MLPEncoder.from_pretrained(save_dir)


def test_from_pretrained_without_config():
    model = MLPEncoder(input_dim=20, hidden_dim=16, output_dim=5)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlp_checkpoint"
        model.save_pretrained(save_dir)
        (save_dir / "config.json").unlink()

        loaded_model = MLPEncoder.from_pretrained(
            save_dir,
            override_kwargs={"input_dim": 20, "hidden_dim": 16, "output_dim": 5},
        )
        assert isinstance(loaded_model, MLPEncoder)


def test_from_pretrained_model_type_mismatch_raises():
    model = MLPEncoder(input_dim=20, hidden_dim=16, output_dim=5)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlp_checkpoint"
        model.save_pretrained(save_dir)

        with pytest.raises(ValueError, match="Model type mismatch"):
            CNNClassifier.from_pretrained(
                save_dir, override_kwargs={"image_width": 28, "image_height": 28}
            )


def test_from_pretrained_strict_mode():
    model = MLPEncoder(input_dim=30, hidden_dim=20, output_dim=10)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlp_checkpoint"
        model.save_pretrained(save_dir)

        loaded_model = MLPEncoder.from_pretrained(save_dir, strict=True)
        assert isinstance(loaded_model, MLPEncoder)


def test_build_model_load_pretrained_with_registry_name_fails():
    params = ModelParams(
        model_name="MLPEncoder",
        load_pretrained_weights=True,
        model_kwargs={"input_dim": 10, "hidden_dim": 8, "output_dim": 5},
    )

    with pytest.raises(ValueError, match="Cannot load pretrained custom model"):
        build_model(params)


def test_save_pretrained_custom_filenames():
    model = MLPEncoder(input_dim=10, hidden_dim=8, output_dim=5)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "custom_names"
        model.save_pretrained(
            save_dir,
            weights_filename="custom_weights.safetensors",
            config_filename="custom_config.json",
        )

        assert (save_dir / "custom_weights.safetensors").exists()
        assert (save_dir / "custom_config.json").exists()

        loaded_model = MLPEncoder.from_pretrained(
            save_dir,
            weights_filename="custom_weights.safetensors",
            config_filename="custom_config.json",
        )
        assert isinstance(loaded_model, MLPEncoder)


def test_save_pretrained_without_config():
    model = MLPEncoder(input_dim=10, hidden_dim=8, output_dim=5)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "no_config"
        model.save_pretrained(save_dir, save_config=False)

        assert (save_dir / "model.safetensors").exists()
        assert not (save_dir / "config.json").exists()
