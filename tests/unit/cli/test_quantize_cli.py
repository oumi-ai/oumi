from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

import oumi
from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.quantize import quantize
from oumi.core.configs.quantization_config import (
    QuantizationAlgorithm,
    QuantizationBackend,
    QuantizationScheme,
)
from oumi.quantize.base import QuantizationResult

runner = CliRunner()


@pytest.fixture
def app():
    fake_app = typer.Typer()
    fake_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(quantize)
    yield fake_app


def _parsed_config() -> MagicMock:
    cfg = MagicMock()
    cfg.model = SimpleNamespace(model_name="Qwen/Qwen3-0.6B")
    cfg.scheme = QuantizationScheme.FP8_DYNAMIC
    cfg.algorithm = QuantizationAlgorithm.AUTO
    cfg.output_path = "quantized_model"
    return cfg


def _quantization_result() -> QuantizationResult:
    return QuantizationResult(
        quantized_size_bytes=1024,
        output_path="/tmp/quantized",
        backend=QuantizationBackend.LLM_COMPRESSOR,
        scheme=QuantizationScheme.FP8_DYNAMIC,
        format_type="safetensors",
        additional_info={"TestKey": "TestValue"},
    )


def _invoke_quantize_with_mocks(
    app, args: list[str], parsed_cfg: MagicMock | None = None
):
    if parsed_cfg is None:
        parsed_cfg = _parsed_config()

    with (
        patch(
            "oumi.cli.quantize.try_get_config_name_for_alias",
            side_effect=lambda v, *_: v,
        ) as mock_alias,
        patch(
            "oumi.cli.cli_utils.resolve_and_fetch_config", side_effect=lambda v: v
        ) as mock_fetch,
        patch("oumi.cli.cli_utils.configure_common_env_vars") as mock_configure_env,
        patch(
            "oumi.core.configs.QuantizationConfig.from_yaml_and_arg_list",
            return_value=parsed_cfg,
        ) as mock_from_yaml,
        patch.object(
            oumi, "quantize", return_value=_quantization_result()
        ) as mock_quantize,
        patch("oumi.utils.torch_utils.device_cleanup") as mock_device_cleanup,
        patch("oumi.telemetry.TelemetryManager.get_instance") as mock_tm_get,
    ):
        tm = MagicMock()
        mock_tm_get.return_value = tm
        result = runner.invoke(app, args)
        return (
            result,
            parsed_cfg,
            mock_from_yaml,
            mock_quantize,
            mock_device_cleanup,
            tm,
            mock_alias,
            mock_fetch,
            mock_configure_env,
        )


def test_list_schemes_callback_exits_zero(app):
    result = runner.invoke(app, ["--list-schemes"])
    assert result.exit_code == 0


def test_list_schemes_callback_false_noop(app):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--list-schemes" in result.stdout


def test_list_schemes_prints_all_schemes(app):
    result = runner.invoke(app, ["--list-schemes"])
    assert result.exit_code == 0
    for scheme in QuantizationScheme:
        assert scheme.value in result.stdout


def test_list_schemes_table_columns(app):
    result = runner.invoke(app, ["--list-schemes"])
    assert result.exit_code == 0
    for col in ["Scheme", "Backend", "Calibration", "Min GPU"]:
        assert col in result.stdout
    # Rich table header can wrap this combined header into separate tokens.
    assert "Default" in result.stdout
    assert "Algorithm" in result.stdout


def test_list_algorithms_callback_exits_zero(app):
    result = runner.invoke(app, ["--list-algorithms"])
    assert result.exit_code == 0


def test_list_algorithms_callback_false_noop(app):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--list-algorithms" in result.stdout


def test_list_algorithms_prints_all_algorithms(app):
    result = runner.invoke(app, ["--list-algorithms"])
    assert result.exit_code == 0
    for algo in QuantizationAlgorithm:
        assert algo.value in result.stdout


def test_list_algorithms_table_columns(app):
    result = runner.invoke(app, ["--list-algorithms"])
    assert result.exit_code == 0
    for col in ["Algorithm", "Calibration", "Default For Schemes", "Description"]:
        assert col in result.stdout


def test_config_required(app):
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    assert "Missing option '--config'" in result.stdout


def test_scheme_override_applied(app):
    (
        result,
        parsed_cfg,
        _mock_from_yaml,
        mock_quantize,
        _mock_device_cleanup,
        _tm,
        *_,
    ) = _invoke_quantize_with_mocks(
        app, ["--config", "config.yaml", "--scheme", "fp8_block"]
    )
    assert result.exit_code == 0
    assert parsed_cfg.scheme == "fp8_block"
    parsed_cfg.finalize_and_validate.assert_called_once()
    mock_quantize.assert_called_once_with(parsed_cfg)


def test_algorithm_override_applied(app):
    (
        result,
        parsed_cfg,
        _mock_from_yaml,
        mock_quantize,
        _mock_device_cleanup,
        _tm,
        *_,
    ) = _invoke_quantize_with_mocks(
        app, ["--config", "config.yaml", "--algorithm", "gptq"]
    )
    assert result.exit_code == 0
    assert parsed_cfg.algorithm == "gptq"
    parsed_cfg.finalize_and_validate.assert_called_once()
    mock_quantize.assert_called_once_with(parsed_cfg)


def test_output_path_override_applied(app):
    (
        result,
        parsed_cfg,
        _mock_from_yaml,
        mock_quantize,
        _mock_device_cleanup,
        _tm,
        *_,
    ) = _invoke_quantize_with_mocks(
        app, ["--config", "config.yaml", "--output_path", "/tmp/foo"]
    )
    assert result.exit_code == 0
    assert parsed_cfg.output_path == "/tmp/foo"
    mock_quantize.assert_called_once_with(parsed_cfg)


def test_model_name_dot_notation_override(app):
    args = ["--config", "config.yaml", "--model.model_name", "Qwen/Qwen3-0.6B"]
    (
        result,
        _parsed_cfg,
        mock_from_yaml,
        _mock_quantize,
        _mock_device_cleanup,
        _tm,
        *_,
    ) = _invoke_quantize_with_mocks(app, args)
    assert result.exit_code == 0
    call_args = mock_from_yaml.call_args.args
    assert call_args[0] == "config.yaml"
    assert "model.model_name=Qwen/Qwen3-0.6B" in call_args[1]


def test_verbose_prints_config(app):
    (
        result,
        parsed_cfg,
        _mock_from_yaml,
        _mock_quantize,
        _mock_device_cleanup,
        _tm,
        *_,
    ) = _invoke_quantize_with_mocks(app, ["--config", "config.yaml", "--verbose"])
    assert result.exit_code == 0
    parsed_cfg.print_config.assert_called_once()


def test_results_table_displayed(app):
    (
        result,
        _parsed_cfg,
        _mock_from_yaml,
        _mock_quantize,
        _mock_device_cleanup,
        _tm,
        *_,
    ) = _invoke_quantize_with_mocks(app, ["--config", "config.yaml"])
    assert result.exit_code == 0
    assert "Quantization Results" in result.stdout
    assert "Output Path" in result.stdout
    assert "Backend" in result.stdout
    assert "Scheme" in result.stdout
    assert "Format" in result.stdout
    assert "Quantized Size" in result.stdout


def test_null_result_exits_one(app):
    parsed_cfg = _parsed_config()
    with (
        patch(
            "oumi.cli.quantize.try_get_config_name_for_alias",
            side_effect=lambda v, *_: v,
        ),
        patch("oumi.cli.cli_utils.resolve_and_fetch_config", side_effect=lambda v: v),
        patch("oumi.cli.cli_utils.configure_common_env_vars"),
        patch(
            "oumi.core.configs.QuantizationConfig.from_yaml_and_arg_list",
            return_value=parsed_cfg,
        ),
        patch.object(oumi, "quantize", return_value=None),
        patch("oumi.utils.torch_utils.device_cleanup") as mock_device_cleanup,
        patch("oumi.telemetry.TelemetryManager.get_instance"),
    ):
        result = runner.invoke(app, ["--config", "config.yaml"])
    assert result.exit_code == 1
    assert mock_device_cleanup.call_count == 2


def test_telemetry_tags_sent(app):
    (
        result,
        _parsed_cfg,
        _mock_from_yaml,
        _mock_quantize,
        _mock_device_cleanup,
        tm,
        *_,
    ) = _invoke_quantize_with_mocks(app, ["--config", "config.yaml"])
    assert result.exit_code == 0
    tm.tags.assert_called_once_with(
        model_name="Qwen/Qwen3-0.6B",
        quantization_scheme=QuantizationScheme.FP8_DYNAMIC,
        quantization_algorithm=QuantizationAlgorithm.AUTO,
    )
