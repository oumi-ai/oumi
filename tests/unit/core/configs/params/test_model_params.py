import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from oumi.core.configs.params.model_params import ModelParams


def test_post_init_adapter_model_present():
    params = ModelParams(model_name="base_model", adapter_model="adapter_model")
    params.finalize_and_validate()

    assert params.model_name == "base_model"
    assert params.adapter_model == "adapter_model"


def test_post_init_adapter_model_not_present(tmp_path: Path):
    # This is the expected config for FFT.
    params = ModelParams(model_name=str(tmp_path))
    params.finalize_and_validate()

    assert Path(params.model_name) == tmp_path
    assert params.adapter_model is None


@patch("oumi.core.configs.params.model_params.find_adapter_config_file")
def test_post_init_adapter_model_not_present_exception(
    mock_find_adapter_config_file, tmp_path: Path
):
    # This is the expected config for FFT.
    mock_find_adapter_config_file.side_effect = OSError("No adapter config found.")
    params = ModelParams(model_name=str(tmp_path))
    params.finalize_and_validate()

    assert Path(params.model_name) == tmp_path
    assert params.adapter_model is None
    mock_find_adapter_config_file.assert_called_with(str(tmp_path))


@patch("oumi.core.configs.params.model_params.logger")
def test_post_init_config_file_present(mock_logger, tmp_path: Path):
    with open(f"{tmp_path}/config.json", "w"):
        pass
    with open(f"{tmp_path}/adapter_config.json", "w"):
        pass

    params = ModelParams(model_name=str(tmp_path))
    params.finalize_and_validate()

    mock_logger.info.assert_called_with(
        "Setting `model_name` to test found in adapter config."
    )
    assert params.model_name == "test"
    assert params.adapter_model == str(tmp_path)


@patch("oumi.core.configs.params.model_params.logger")
def test_post_init_config_file_not_present(mock_logger, tmp_path: Path):
    with open(f"{tmp_path}/config.json", "w"):
        pass

    params = ModelParams(model_name=str(tmp_path))
    params.finalize_and_validate()

    mock_logger.info.assert_has_calls(
        [
            call("No adapter config found. Loading as base model."),
        ]
    )

    assert params.model_name == str(tmp_path)
    assert params.adapter_model is None


@patch("oumi.core.configs.params.model_params.logger")
def test_post_init_config_file_empty(mock_logger, tmp_path: Path):
    with open(f"{tmp_path}/config.json", "w"):
        pass
    with open(f"{tmp_path}/adapter_config.json", "w"):
        pass

    params = ModelParams(model_name=str(tmp_path))
    params.finalize_and_validate()

    mock_logger.info.assert_has_calls(
        [
            call("No base model found in adapter config. Loading as base model."),
        ]
    )

    assert params.model_name == str(tmp_path)
    assert params.adapter_model is None


def _get_invalid_field_name_lists() -> list[list[str]]:
    return [
        ["model_name", "tokenizer_name"],
        ["model", "tokenizer"],
        ["dataset_name", "target_column"],
        ["processor_name", "processor_kwargs"],
    ]


def _get_test_name_for_invalid_field_name_list(x):
    return ", ".join(x)


@pytest.mark.parametrize(
    "field_names", _get_invalid_field_name_lists(), ids=_get_test_name_for_invalid_field_name_list
)
def test_model_params_reserved_processor_kwargs(field_names: list[str], tmp_path: Path):
    # Get all the fields from ModelParams and only return those that are not
    # in the list of field names.
    all_fields = {f.name for f in dataclasses.fields(ModelParams)}
    allowed_names = all_fields - set(field_names)
    with pytest.raises(ValueError):
        ModelParams(
            model_name=str(tmp_path),
            processor_kwargs={field_name: "foo_value" for field_name in field_names},
        )


def test_new_hf_kernels_field_default(tmp_path):
    """Test that enable_hf_kernels field has correct default."""
    params = ModelParams(model_name=str(tmp_path))

    assert hasattr(params, "enable_hf_kernels")
    assert params.enable_hf_kernels is False


def test_hf_kernels_field_can_be_set(tmp_path):
    """Test that enable_hf_kernels field can be set."""
    params = ModelParams(model_name=str(tmp_path), enable_hf_kernels=True)

    assert params.enable_hf_kernels is True