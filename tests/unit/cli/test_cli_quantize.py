# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from oumi.cli.quantize import quantize
from oumi.core.configs import QuantizationConfig
from oumi.quantize.base import QuantizationResult

runner = CliRunner()


@pytest.fixture
def app():
    fake_app = typer.Typer()
    fake_app.command()(quantize)
    yield fake_app


@pytest.fixture
def mock_quantize():
    """Stubs the quantizer; the CLI formats these fields, so return real types."""
    with patch("oumi.quantize") as mock:
        mock.return_value = QuantizationResult(
            quantized_size_bytes=1024,
            output_path="/tmp/quantized",
            quantization_method="bnb_4bit",
            format_type="safetensors",
        )
        yield mock


def _captured_config(mock_quantize) -> QuantizationConfig:
    assert mock_quantize.call_count == 1
    return mock_quantize.call_args.args[0]


@pytest.mark.parametrize("method", ["awq_q4_0", "bnb_4bit"])
def test_cli_path_uses_safetensors(app, mock_quantize, method):
    """Without --config the CLI must emit a supported output_format.

    Both branches previously set "pytorch", which is absent from
    SUPPORTED_OUTPUT_FORMATS, so finalize_and_validate() rejected every
    invocation of this path.
    """
    result = runner.invoke(
        app, ["--model", "some/model", "--method", method, "--output", "out"]
    )
    assert result.exit_code == 0, result.output
    assert _captured_config(mock_quantize).output_format == "safetensors"


def test_config_output_path_survives_when_output_not_passed(app, mock_quantize):
    """A config's output_path must not be clobbered by the --output default.

    The guard compared against "quantized_model.gguf" while the option default
    is "quantized_model", so it always fired and overwrote output_path.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "quant.yaml"
        config_path.write_text(
            "model:\n"
            "  model_name: 'some/model'\n"
            "method: 'bnb_4bit'\n"
            "output_path: '/tmp/from_config'\n"
            "output_format: 'safetensors'\n"
        )
        result = runner.invoke(app, ["--config", str(config_path)])
        assert result.exit_code == 0, result.output
        assert _captured_config(mock_quantize).output_path == "/tmp/from_config"


def test_explicit_output_still_overrides_config(app, mock_quantize):
    """An explicit --output must still win over the config's output_path."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "quant.yaml"
        config_path.write_text(
            "model:\n"
            "  model_name: 'some/model'\n"
            "method: 'bnb_4bit'\n"
            "output_path: '/tmp/from_config'\n"
            "output_format: 'safetensors'\n"
        )
        result = runner.invoke(
            app, ["--config", str(config_path), "--output", "/tmp/from_cli"]
        )
        assert result.exit_code == 0, result.output
        assert _captured_config(mock_quantize).output_path == "/tmp/from_cli"
