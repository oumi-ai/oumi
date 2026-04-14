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

import json
import tempfile
from pathlib import Path

import typer
import yaml
from typer.testing import CliRunner

from oumi.cli.analyze import analyze
from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS

runner = CliRunner()


def _create_typed_config_yaml(
    dataset_path: str,
    output_path: str = "",
) -> dict:
    """Create a minimal typed analyzer config dict."""
    return {
        "dataset_path": dataset_path,
        "output_path": output_path,
        "analyzers": [
            {"type": "length", "display_name": "Length"},
        ],
    }


def _create_test_dataset(path: str) -> None:
    """Create a minimal JSONL dataset file."""
    conversations = [
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thanks!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {
                    "role": "assistant",
                    "content": "Python is a programming language.",
                },
            ]
        },
    ]
    with open(path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")


def _get_app() -> typer.Typer:
    """Create a test Typer app with the analyze command."""
    app = typer.Typer()
    app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(analyze)
    return app


def test_analyze_runs_with_typed_config():
    """Test that analyze command runs successfully with a typed config."""
    app = _get_app()
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = str(Path(tmpdir) / "data.jsonl")
        output_path = str(Path(tmpdir) / "output")
        config_path = str(Path(tmpdir) / "config.yaml")

        _create_test_dataset(dataset_path)
        config = _create_typed_config_yaml(dataset_path, output_path)
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["--config", config_path])

        assert result.exit_code == 0
        # Output should contain analysis summary
        assert "Analysis Summary" in result.stdout
        assert "Conversations analyzed: 2" in result.stdout
        # Output files should be created
        assert Path(output_path).exists()


def test_analyze_invalid_format():
    """Test that invalid format raises an error early."""
    app = _get_app()
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = str(Path(tmpdir) / "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump({"analyzers": [{"id": "length"}]}, f)

        result = runner.invoke(
            app, ["--config", config_path, "--format", "invalid_format"]
        )

        assert result.exit_code == 1
        assert "Invalid output format" in result.stdout


def test_analyze_missing_config():
    """Test that missing config shows an error."""
    app = _get_app()
    result = runner.invoke(app, [])

    assert result.exit_code == 1
    assert "Missing option" in result.stdout


def test_analyze_list_metrics():
    """Test that --list-metrics flag works without a config."""
    app = _get_app()
    result = runner.invoke(app, ["--list-metrics"])

    assert result.exit_code == 0
    assert "Available Metrics" in result.stdout
    assert "LengthAnalyzer" in result.stdout


def test_analyze_output_format_csv():
    """Test CSV output format."""
    app = _get_app()
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = str(Path(tmpdir) / "data.jsonl")
        output_path = str(Path(tmpdir) / "output")
        config_path = str(Path(tmpdir) / "config.yaml")

        _create_test_dataset(dataset_path)
        config = _create_typed_config_yaml(dataset_path, output_path)
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["--config", config_path, "--format", "csv"])

        assert result.exit_code == 0
        assert (Path(output_path) / "analysis.csv").exists()


def test_analyze_output_format_json():
    """Test JSON output format."""
    app = _get_app()
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = str(Path(tmpdir) / "data.jsonl")
        output_path = str(Path(tmpdir) / "output")
        config_path = str(Path(tmpdir) / "config.yaml")

        _create_test_dataset(dataset_path)
        config = _create_typed_config_yaml(dataset_path, output_path)
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["--config", config_path, "--format", "json"])

        assert result.exit_code == 0
        assert (Path(output_path) / "analysis.json").exists()


def test_analyze_format_case_insensitive():
    """Test that output format is case-insensitive."""
    app = _get_app()
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = str(Path(tmpdir) / "data.jsonl")
        output_path = str(Path(tmpdir) / "output")
        config_path = str(Path(tmpdir) / "config.yaml")

        _create_test_dataset(dataset_path)
        config = _create_typed_config_yaml(dataset_path, output_path)
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["--config", config_path, "--format", "CSV"])

        assert result.exit_code == 0
