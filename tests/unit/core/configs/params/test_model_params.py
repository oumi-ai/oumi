import dataclasses
from pathlib import Path
from unittest.mock import call, patch

import pytest

from oumi.core.configs.params.model_params import ModelParams
from oumi.exceptions import OumiConfigError


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

    assert Path(params.model_name) == tmp_path
    assert Path(params.adapter_model or "") == tmp_path
    mock_logger.info.assert_called_with(
        f"Found LoRA adapter at {tmp_path}, setting `adapter_model` to `model_name`."
    )


@patch("oumi.core.configs.params.model_params.logger")
def test_post_init_config_file_not_present(mock_logger, tmp_path: Path):
    with open(f"{tmp_path}/adapter_config.json", "w") as f:
        f.write('{"base_model_name_or_path": "base_model"}')

    params = ModelParams(model_name=str(tmp_path))
    params.finalize_and_validate()

    assert params.model_name == "base_model"
    assert Path(params.adapter_model or "") == tmp_path

    assert mock_logger.info.call_count == 2
    mock_logger.info.assert_has_calls(
        [
            call(
                f"Found LoRA adapter at {tmp_path}, setting `adapter_model` to "
                "`model_name`."
            ),
            call("Setting `model_name` to base_model found in adapter config."),
        ]
    )


@patch("oumi.core.configs.params.model_params.logger")
def test_post_init_config_file_empty(mock_logger, tmp_path: Path):
    with open(f"{tmp_path}/adapter_config.json", "w") as f:
        f.write("{}")

    params = ModelParams(model_name=str(tmp_path))
    with pytest.raises(
        OumiConfigError,
        match="`model_name` specifies an adapter model only,"
        " but the base model could not be found!",
    ):
        params.finalize_and_validate()


def _get_invalid_field_name_lists() -> list[list[str]]:
    all_fields: set[str] = {f.name for f in dataclasses.fields(ModelParams())}
    result = [[field_name] for field_name in all_fields]
    result.extend([["valid_kwarg", field_name] for field_name in all_fields][:3])
    return result


def _get_test_name_for_invalid_field_name_list(x):
    assert isinstance(x, list)
    return "--".join(x)


@pytest.mark.parametrize(
    "field_names",
    _get_invalid_field_name_lists(),
    ids=_get_test_name_for_invalid_field_name_list,
)
def test_model_params_reserved_processor_kwargs(field_names: list[str], tmp_path: Path):
    invalid_names = {f.name for f in dataclasses.fields(ModelParams())}.intersection(
        field_names
    )
    with pytest.raises(
        OumiConfigError,
        match=(
            "processor_kwargs attempts to override the following reserved fields: "
            f"{invalid_names}"
        ),
    ):
        ModelParams(
            model_name=str(tmp_path),
            processor_kwargs={field_name: "foo_value" for field_name in field_names},
        )


def test_chat_template_kwargs_custom_assignment():
    model_params = ModelParams(chat_template_kwargs={"enable_thinking": False})
    assert model_params.chat_template_kwargs is not None
    assert model_params.chat_template_kwargs["enable_thinking"] is False


@patch("oumi.core.configs.params.model_params.find_adapter_config_file")
def test_adapter_config_file_path_not_a_file(mock_find, tmp_path: Path):
    """Raises OumiConfigError when the reported adapter config path is missing.

    When `find_adapter_config_file` returns a path that does not exist on disk, the
    subsequent `open()` call in the adapter-config reading branch raises
    `FileNotFoundError`, which is wrapped as `OumiConfigError` with a
    "Failed to read adapter config" message.
    """
    mock_find.return_value = str(tmp_path / "ghost_adapter_config.json")

    params = ModelParams(model_name=str(tmp_path))
    with pytest.raises(
        OumiConfigError,
        match="Failed to read adapter config",
    ):
        params.finalize_and_validate()


@patch("oumi.core.configs.params.model_params.find_adapter_config_file")
def test_adapter_config_read_oserror(mock_find, tmp_path: Path):
    """Test OSError reading adapter_config.json is re-raised as OumiConfigError."""
    adapter_path = tmp_path / "adapter_config.json"
    adapter_path.write_text('{"base_model_name_or_path": "base_model"}')
    mock_find.return_value = str(adapter_path)

    params = ModelParams(model_name=str(tmp_path))
    with patch("builtins.open", side_effect=OSError("Permission denied")):
        with pytest.raises(
            OumiConfigError,
            match="Failed to read adapter config",
        ):
            params.finalize_and_validate()


def test_adapter_config_invalid_json(tmp_path: Path):
    """Test malformed JSON raises OumiConfigError with location info."""
    (tmp_path / "adapter_config.json").write_text("{not: valid json!!}")

    params = ModelParams(model_name=str(tmp_path))
    with pytest.raises(OumiConfigError, match="contains invalid JSON") as exc_info:
        params.finalize_and_validate()

    assert "line" in str(exc_info.value)
    assert "col" in str(exc_info.value)
