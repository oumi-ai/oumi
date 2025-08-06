import dataclasses
from pathlib import Path
from unittest.mock import call, patch, MagicMock

import pytest

from oumi.core.configs.params.model_params import (
    ModelParams, 
    _is_flash_attn_3_available,
    _resolve_flash_attention_implementation,
)
from oumi.core.types.exceptions import HardwareException


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
        ValueError,
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
        ValueError,
        match=(
            "processor_kwargs attempts to override the following reserved fields: "
            f"{invalid_names}"
        ),
    ):
        ModelParams(
            model_name=str(tmp_path),
            processor_kwargs={field_name: "foo_value" for field_name in field_names},
        )


class TestFlashAttentionDetection:
    """Test Flash Attention detection and resolution logic."""

    @patch("oumi.core.configs.params.model_params.flash_attn_interface")
    def test_is_flash_attn_3_available_with_fa3(self, mock_flash_attn_interface):
        """Test FA3 detection when Flash Attention 3 is available."""
        # Mock successful import and function access
        mock_flash_attn_interface.flash_attn_func = MagicMock()
        
        result = _is_flash_attn_3_available()
        
        assert result is True

    @patch("oumi.core.configs.params.model_params.flash_attn_interface", side_effect=ImportError("No module"))
    def test_is_flash_attn_3_available_no_fa3(self, mock_flash_attn_interface):
        """Test FA3 detection when Flash Attention 3 is not available."""
        result = _is_flash_attn_3_available()
        
        assert result is False

    @patch("oumi.core.configs.params.model_params._is_flash_attn_3_available")
    @patch("oumi.core.configs.params.model_params.is_flash_attn_2_available")
    def test_resolve_flash_attention_fa3_available(self, mock_fa2_available, mock_fa3_available):
        """Test attention resolution when FA3 is available."""
        mock_fa3_available.return_value = True
        mock_fa2_available.return_value = True
        
        result = _resolve_flash_attention_implementation("flash_attention")
        
        assert result == "flash_attention_2"  # HF expects this internally

    @patch("oumi.core.configs.params.model_params._is_flash_attn_3_available")
    @patch("oumi.core.configs.params.model_params.is_flash_attn_2_available") 
    def test_resolve_flash_attention_fa2_available(self, mock_fa2_available, mock_fa3_available):
        """Test attention resolution when only FA2 is available."""
        mock_fa3_available.return_value = False
        mock_fa2_available.return_value = True
        
        result = _resolve_flash_attention_implementation("flash_attention")
        
        assert result == "flash_attention_2"

    @patch("oumi.core.configs.params.model_params._is_flash_attn_3_available")
    @patch("oumi.core.configs.params.model_params.is_flash_attn_2_available")
    def test_resolve_flash_attention_fallback_to_sdpa(self, mock_fa2_available, mock_fa3_available):
        """Test attention resolution fallback to SDPA when no FA available."""
        mock_fa3_available.return_value = False
        mock_fa2_available.return_value = False
        
        result = _resolve_flash_attention_implementation("flash_attention")
        
        assert result == "sdpa"

    def test_resolve_flash_attention_passthrough_other_types(self):
        """Test that non-flash-attention types are passed through unchanged."""
        for attn_type in ["sdpa", "eager", None, "custom_attention"]:
            result = _resolve_flash_attention_implementation(attn_type)
            assert result == attn_type

    @patch("oumi.core.configs.params.model_params.logger")
    @patch("oumi.core.configs.params.model_params._is_flash_attn_3_available")
    def test_resolve_flash_attention_deprecation_warning(self, mock_fa3_available, mock_logger):
        """Test deprecation warning for flash_attention_2 syntax."""
        mock_fa3_available.return_value = False
        
        with patch("oumi.core.configs.params.model_params.is_flash_attn_2_available", return_value=True):
            _resolve_flash_attention_implementation("flash_attention_2")
        
        mock_logger.warning.assert_called_once()
        assert "deprecated" in mock_logger.warning.call_args[0][0]


class TestModelParamsAttentionResolution:
    """Test ModelParams attention resolution in validation."""

    @patch("oumi.core.configs.params.model_params._resolve_flash_attention_implementation")
    def test_finalize_and_validate_resolves_attention(self, mock_resolve, tmp_path):
        """Test that finalize_and_validate calls attention resolution."""
        mock_resolve.return_value = "flash_attention_2"
        
        params = ModelParams(
            model_name=str(tmp_path), 
            attn_implementation="flash_attention"
        )
        params.finalize_and_validate()
        
        mock_resolve.assert_called_once_with("flash_attention")
        assert params.attn_implementation == "flash_attention_2"

    @patch("oumi.core.configs.params.model_params._resolve_flash_attention_implementation")
    def test_finalize_and_validate_none_resolution_raises(self, mock_resolve, tmp_path):
        """Test that None resolution raises HardwareException."""
        mock_resolve.return_value = None
        
        params = ModelParams(
            model_name=str(tmp_path),
            attn_implementation="flash_attention"
        )
        
        with pytest.raises(HardwareException, match="not available"):
            params.finalize_and_validate()

    @patch("oumi.core.configs.params.model_params._resolve_flash_attention_implementation")  
    @patch.object(ModelParams, "_validate_flash_attention_3_requirements")
    @patch.object(ModelParams, "_is_using_flash_attention_3")
    def test_finalize_and_validate_calls_fa3_validation(
        self, mock_is_fa3, mock_validate_fa3, mock_resolve, tmp_path
    ):
        """Test that FA3 validation is called when using FA3."""
        mock_resolve.return_value = "flash_attention_2"
        mock_is_fa3.return_value = True
        
        params = ModelParams(
            model_name=str(tmp_path),
            attn_implementation="flash_attention"
        )
        params.finalize_and_validate()
        
        mock_validate_fa3.assert_called_once()

    def test_new_hf_kernels_field_default(self, tmp_path):
        """Test that enable_hf_kernels field has correct default."""
        params = ModelParams(model_name=str(tmp_path))
        
        assert hasattr(params, "enable_hf_kernels")
        assert params.enable_hf_kernels is False

    def test_hf_kernels_field_can_be_set(self, tmp_path):
        """Test that enable_hf_kernels field can be set."""
        params = ModelParams(
            model_name=str(tmp_path),
            enable_hf_kernels=True
        )
        
        assert params.enable_hf_kernels is True
