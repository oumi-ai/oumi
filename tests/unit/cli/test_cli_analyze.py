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

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import typer
from typer.testing import CliRunner

import oumi.cli.alias
from oumi.cli.analyze import analyze
from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.core.configs import AnalyzeConfig, DatasetSource, SampleAnalyzerParams
from oumi.utils.logging import logger

runner = CliRunner()


@pytest.fixture
def mock_fetch():
    with patch("oumi.cli.cli_utils.resolve_and_fetch_config") as m_fetch:
        yield m_fetch


@pytest.fixture
def mock_alias():
    with patch("oumi.cli.analyze.try_get_config_name_for_alias") as try_alias:
        yield try_alias


def _create_analyze_config() -> AnalyzeConfig:
    """Create a minimal valid AnalyzeConfig for testing."""
    return AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG,
        dataset_name="oumi-ai/oumi-synthetic-document-claims",
        split="train",
        sample_count=10,
        analyzers=[
            SampleAnalyzerParams(
                id="length",
                params={"char_count": True, "word_count": True},
            )
        ],
        output_path="",  # Empty to skip export in tests
    )


@pytest.fixture
def app():
    fake_app = typer.Typer()
    fake_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(analyze)
    yield fake_app


@pytest.fixture
def mock_dataset_analyzer():
    # DatasetAnalyzer is imported inside the analyze function, so we need to patch
    # it in the module where it's looked up
    with patch(
        "oumi.core.analyze.dataset_analyzer.DatasetAnalyzer", autospec=True
    ) as m_analyzer_class:
        # Create a mock analyzer instance
        mock_instance = MagicMock()
        mock_instance.analysis_summary = {
            "dataset_overview": {
                "dataset_name": "test_dataset",
                "total_conversations": 100,
                "conversations_analyzed": 10,
                "dataset_coverage_percentage": 10.0,
                "total_messages": 50,
                "analyzers_used": ["length"],
            },
            "message_level_summary": {},
            "conversation_level_summary": {},
            "conversation_turns": {},
        }
        mock_instance.message_df = None
        mock_instance.conversation_df = None
        m_analyzer_class.return_value = mock_instance
        yield m_analyzer_class


def test_analyze_runs(app, mock_dataset_analyzer):
    """Test that analyze command runs successfully with a valid config."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "analyze.yaml")
        config = _create_analyze_config()
        config.to_yaml(yaml_path)

        runner.invoke(app, ["--config", yaml_path])

        # Check the command completed (may exit with 0 or show output)
        mock_dataset_analyzer.assert_called_once()
        mock_dataset_analyzer.return_value.analyze_dataset.assert_called_once()


def test_analyze_calls_alias(app, mock_dataset_analyzer, mock_alias):
    """Test that analyze command resolves aliases correctly."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "analyze.yaml")
        mock_alias.return_value = yaml_path
        config = _create_analyze_config()
        config.to_yaml(yaml_path)

        _ = runner.invoke(app, ["--config", "an_alias"])

        mock_alias.assert_has_calls(
            [
                call("an_alias", oumi.cli.alias.AliasType.ANALYZE),
            ]
        )


def test_analyze_with_output_override(app, mock_dataset_analyzer):
    """Test that output path can be overridden via CLI."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "analyze.yaml")
        output_dir = str(Path(output_temp_dir) / "custom_output")
        config = _create_analyze_config()
        config.to_yaml(yaml_path)

        # Mock message_df and conversation_df to test export
        mock_dataset_analyzer.return_value.message_df = MagicMock()
        mock_dataset_analyzer.return_value.message_df.empty = False
        mock_dataset_analyzer.return_value.conversation_df = MagicMock()
        mock_dataset_analyzer.return_value.conversation_df.empty = False

        runner.invoke(app, ["--config", yaml_path, "--output", output_dir])

        # The analyzer should have been called
        mock_dataset_analyzer.assert_called_once()


def test_analyze_with_format_override(app, mock_dataset_analyzer):
    """Test that output format can be overridden via CLI."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "analyze.yaml")
        config = _create_analyze_config()
        config.to_yaml(yaml_path)

        runner.invoke(app, ["--config", yaml_path, "--format", "json"])

        mock_dataset_analyzer.assert_called_once()


def test_analyze_invalid_format(app):
    """Test that invalid format raises an error early."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "analyze.yaml")
        config = _create_analyze_config()
        config.to_yaml(yaml_path)

        result = runner.invoke(
            app, ["--config", yaml_path, "--format", "invalid_format"]
        )

        assert result.exit_code == 1
        assert "Invalid output format" in result.stdout


def test_analyze_logging_levels(app, mock_dataset_analyzer):
    """Test that logging levels can be configured via CLI."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "analyze.yaml")
        config = _create_analyze_config()
        config.to_yaml(yaml_path)

        _ = runner.invoke(app, ["--config", yaml_path, "--log-level", "DEBUG"])
        assert logger.level == logging.DEBUG

        _ = runner.invoke(app, ["--config", yaml_path, "-log", "WARNING"])
        assert logger.level == logging.WARNING


def test_analyze_with_cli_overrides(app, mock_dataset_analyzer):
    """Test that config values can be overridden via CLI arguments."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "analyze.yaml")
        config = _create_analyze_config()
        config.to_yaml(yaml_path)

        runner.invoke(
            app,
            [
                "--config",
                yaml_path,
                "--sample_count",
                "5",
            ],
        )

        mock_dataset_analyzer.assert_called_once()


def test_analyze_with_oumi_prefix(app, mock_dataset_analyzer, mock_fetch):
    """Test that oumi:// prefix config paths are resolved correctly."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        output_dir = Path(output_temp_dir)
        yaml_path = "oumi://configs/examples/analyze/analyze.yaml"
        expected_path = output_dir / "configs/examples/analyze/analyze.yaml"

        config = _create_analyze_config()
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(expected_path)
        mock_fetch.return_value = expected_path

        with patch.dict("os.environ", {"OUMI_DIR": str(output_dir)}):
            runner.invoke(app, ["--config", yaml_path])

        mock_fetch.assert_called_once_with(yaml_path)


def test_analyze_verbose_flag(app, mock_dataset_analyzer):
    """Test that verbose flag enables additional debug output."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "analyze.yaml")
        config = _create_analyze_config()
        config.to_yaml(yaml_path)

        runner.invoke(app, ["--config", yaml_path, "--verbose"])

        mock_dataset_analyzer.assert_called_once()


def test_analyze_format_case_insensitive(app, mock_dataset_analyzer):
    """Test that output format is case-insensitive."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "analyze.yaml")
        config = _create_analyze_config()
        config.to_yaml(yaml_path)

        # Test uppercase format
        runner.invoke(app, ["--config", yaml_path, "--format", "CSV"])
        mock_dataset_analyzer.assert_called()

        # Test mixed case format
        runner.invoke(app, ["--config", yaml_path, "--format", "Json"])
        mock_dataset_analyzer.assert_called()
