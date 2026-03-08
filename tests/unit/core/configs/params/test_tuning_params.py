from pathlib import Path

import pytest

from oumi.core.configs.params.telemetry_params import TelemetryParams
from oumi.core.configs.params.training_params import TrainerType
from oumi.core.configs.params.tuning_params import TuningParams


def test_post_init_tunable_fixed_training_params():
    """Test verification for valid tunable and fixed training params."""
    tunable_params = {
        "learning_rate": {
            "type": "float",
            "low": "1e-5",
            "high": "1e-3",
        },
        "per_device_train_batch_size": {
            "type": "int",
            "low": "16",
            "high": "64",
        },
        "optimizer": {
            "type": "categorical",
            "choices": ["adamw_torch", "sgd", "adafactor"],
        },
    }
    params = TuningParams(
        tunable_training_params=tunable_params,
        fixed_training_params={"num_train_epochs": 10},
    )
    params.finalize_and_validate()

    assert params.tunable_training_params == tunable_params
    assert params.fixed_training_params == {"num_train_epochs": 10}


def test_post_init_tunable_fixed_peft_params():
    """Test verification for valid tunable and fixed peft params."""
    tunable_params = {
        "lora_r": {"type": "categorical", "choices": [4, 8, 16]},
        "lora_target_modules": {
            "type": "categorical",
            "choices": [["q_proj", "v_proj"], ["k_proj", "o_proj"]],
        },
        "lora_dropout": {
            "type": "float",
            "low": 0.0,
            "high": 0.5,
        },
    }
    params = TuningParams(
        tunable_peft_params=tunable_params,
        fixed_peft_params={"lora_bias": "none"},
    )
    params.finalize_and_validate()

    assert params.tunable_peft_params == tunable_params
    assert params.fixed_peft_params == {"lora_bias": "none"}


def test_post_init_non_existing_params():
    """Test that invalid training and peft params fail."""
    with pytest.raises(ValueError) as excinfo:
        tunable_params = {
            "some_wrong_named_param": {
                "type": "float",
                "low": "1e-5",
                "high": "1e-3",
            },
        }
        params = TuningParams(
            tunable_training_params=tunable_params,
            fixed_training_params={"num_train_epochs": 10},
        )
        params.finalize_and_validate()

    assert "Invalid tunable parameter: some_wrong_named_param" in str(excinfo.value)
    assert "Must be a valid `TrainingParams` field." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        tunable_params = {
            "invalid_peft_param": {"type": "categorical", "choices": [4, 8, 16]},
        }
        params = TuningParams(
            tunable_peft_params=tunable_params,
            fixed_peft_params={"lora_bias": "none"},
        )
        params.finalize_and_validate()

    assert "Invalid tunable parameter: invalid_peft_param" in str(excinfo.value)
    assert "Must be a valid `PeftParams` field." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        tunable_params = {
            "learning_rate": {
                "type": "float",
                "low": "1e-5",
                "high": "1e-3",
            },
        }
        params = TuningParams(
            tunable_training_params=tunable_params,
            fixed_training_params={"some_wrong_param": 10},
        )
        params.finalize_and_validate()

    assert "Invalid fixed parameter: some_wrong_param" in str(excinfo.value)
    assert "Must be a valid `TrainingParams` field." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        tunable_params = {
            "lora_r": {"type": "categorical", "choices": [4, 8, 16]},
        }
        params = TuningParams(
            tunable_peft_params=tunable_params,
            fixed_peft_params={"a_wrong_param": "none"},
        )
        params.finalize_and_validate()

    assert "Invalid fixed parameter: a_wrong_param" in str(excinfo.value)
    assert "Must be a valid `PeftParams` field." in str(excinfo.value)


def test_invalid_param_type():
    """Test that invalid ParamType values are rejected."""
    with pytest.raises(ValueError) as excinfo:
        params = TuningParams(
            tunable_training_params={
                "learning_rate": {
                    "type": "invalid_type",
                    "low": 1e-5,
                    "high": 1e-3,
                }
            }
        )
        params.finalize_and_validate()

    assert "Invalid type 'invalid_type'" in str(excinfo.value)
    assert "Must be one of:" in str(excinfo.value)


def test_categorical_missing_choices():
    """Test that categorical params require 'choices' key."""
    with pytest.raises(ValueError) as excinfo:
        params = TuningParams(
            tunable_training_params={
                "optimizer": {
                    "type": "categorical",
                }
            }
        )
        params.finalize_and_validate()

    assert "must have 'choices' key" in str(excinfo.value)


def test_categorical_empty_choices():
    """Test that categorical params require non-empty choices."""
    with pytest.raises(ValueError) as excinfo:
        params = TuningParams(
            tunable_training_params={
                "optimizer": {
                    "type": "categorical",
                    "choices": [],
                }
            }
        )
        params.finalize_and_validate()

    assert "must have non-empty choices list" in str(excinfo.value)


def test_numeric_missing_low_high():
    """Test that numeric params require 'low' and 'high' keys."""
    with pytest.raises(ValueError) as excinfo:
        params = TuningParams(
            tunable_training_params={
                "learning_rate": {
                    "type": "float",
                    "low": 1e-5,
                    # Missing 'high'
                }
            }
        )
        params.finalize_and_validate()

    assert "must have 'low' and 'high' keys" in str(excinfo.value)


def test_evaluation_direction_mismatch():
    """Test that evaluation_direction length must match evaluation_metrics."""
    with pytest.raises(ValueError) as excinfo:
        params = TuningParams(
            evaluation_metrics=["eval_loss", "accuracy"],
            evaluation_direction=["minimize", "maximize", "minimize"],  # Too many
        )
        params.finalize_and_validate()

    assert "Length of evaluation_metrics must match" in str(excinfo.value)


def test_evaluation_direction_single_broadcast():
    """Test that single evaluation_direction is broadcast to all metrics."""
    params = TuningParams(
        evaluation_metrics=["eval_loss", "accuracy", "f1"],
        evaluation_direction=["minimize"],
    )
    params.finalize_and_validate()

    assert params.evaluation_direction == ["minimize", "minimize", "minimize"]


def test_invalid_evaluation_direction():
    """Test that invalid evaluation directions are rejected."""
    with pytest.raises(ValueError) as excinfo:
        params = TuningParams(
            evaluation_metrics=["eval_loss"],
            evaluation_direction=["invalid"],
        )
        params.finalize_and_validate()

    assert "Invalid evaluation_direction: invalid" in str(excinfo.value)
    assert 'Choose either "minimize" or "maximize"' in str(excinfo.value)


def test_invalid_logging_strategy():
    """Test that invalid logging strategies are rejected."""
    with pytest.raises(ValueError) as excinfo:
        params = TuningParams(logging_strategy="invalid")
        params.finalize_and_validate()

    assert "Invalid logging_strategy: invalid" in str(excinfo.value)


def test_invalid_trainer_type():
    """Test that only TRL_SFT trainer is currently supported."""
    with pytest.raises(ValueError) as excinfo:
        params = TuningParams(trainer_type=TrainerType.TRL_DPO)
        params.finalize_and_validate()

    assert "Invalid trainer_type" in str(excinfo.value)


def test_logging_dir_default():
    """Test that logging_dir defaults to output_dir/logs."""
    params = TuningParams(output_dir="/custom/output")
    params.finalize_and_validate()

    assert params.logging_dir == "/custom/output/logs"


def test_logging_dir_custom():
    """Test that custom logging_dir is preserved."""
    params = TuningParams(output_dir="/custom/output", logging_dir="/custom/logs")
    params.finalize_and_validate()

    assert params.logging_dir == "/custom/logs"


def test_telemetry_dir_absolute():
    """Test telemetry_dir with absolute path."""
    params = TuningParams(
        output_dir="/output",
        telemetry=TelemetryParams(telemetry_dir="/absolute/telemetry"),
    )
    params.finalize_and_validate()

    assert params.telemetry_dir == Path("/absolute/telemetry")


def test_telemetry_dir_relative():
    """Test telemetry_dir with relative path."""
    params = TuningParams(
        output_dir="/output",
        telemetry=TelemetryParams(telemetry_dir="relative/telemetry"),
    )
    params.finalize_and_validate()

    assert params.telemetry_dir == Path("/output/relative/telemetry")
