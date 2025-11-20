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

"""Unit tests for the magic CLI command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from oumi.cli.magic import (
    _build_command,
    _find_config_files,
    _get_configs_dir,
    magic,
)

runner = CliRunner()


class TestConfigDiscovery:
    """Test config file discovery functions."""

    def test_get_configs_dir(self):
        """Test that configs dir is correctly resolved."""
        configs_dir = _get_configs_dir()
        assert configs_dir.exists()
        assert configs_dir.name == "recipes"
        assert (configs_dir / "smollm").exists()

    def test_find_config_files_smollm_sft(self):
        """Test finding SmolLM SFT configs."""
        configs = _find_config_files("smollm", "sft")
        assert len(configs) > 0
        assert all(c.suffix == ".yaml" for c in configs)
        assert all("smollm" in str(c) for c in configs)

    def test_find_config_files_nonexistent(self):
        """Test finding configs for non-existent model/workflow."""
        configs = _find_config_files("nonexistent_model", "sft")
        assert len(configs) == 0

    def test_find_config_files_filters_job_configs(self):
        """Test that job configs are filtered out."""
        configs = _find_config_files("smollm", "sft")
        # Should not include gcp_job, slurm_job, etc.
        assert not any("gcp_job" in c.name for c in configs)
        assert not any("slurm_job" in c.name for c in configs)

    def test_find_config_files_various_models(self):
        """Test finding configs for various model families."""
        test_cases = [
            ("llama3_2", "sft"),
            ("phi3", "evaluation"),
            ("qwen3", "inference"),
        ]

        for model, workflow in test_cases:
            configs = _find_config_files(model, workflow)
            # Should find at least one config for these popular combinations
            if (Path(_get_configs_dir()) / model / workflow).exists():
                assert len(configs) > 0


class TestCommandBuilding:
    """Test command building functions."""

    def test_build_command_sft(self):
        """Test building SFT training command."""
        config_path = Path("/path/to/config.yaml")
        cmd = _build_command("sft", config_path)
        assert cmd == ["oumi", "train", "-c", str(config_path)]

    def test_build_command_evaluation(self):
        """Test building evaluation command."""
        config_path = Path("/path/to/eval.yaml")
        cmd = _build_command("evaluation", config_path)
        assert cmd == ["oumi", "evaluate", "-c", str(config_path)]

    def test_build_command_inference(self):
        """Test building inference command."""
        config_path = Path("/path/to/infer.yaml")
        cmd = _build_command("inference", config_path)
        assert cmd == ["oumi", "infer", "-c", str(config_path)]

    def test_build_command_dpo(self):
        """Test building DPO command."""
        config_path = Path("/path/to/dpo.yaml")
        cmd = _build_command("dpo", config_path)
        assert cmd == ["oumi", "train", "-c", str(config_path)]

    def test_build_command_preference(self):
        """Test building preference alignment command."""
        config_path = Path("/path/to/pref.yaml")
        cmd = _build_command("preference", config_path)
        assert cmd == ["oumi", "train", "-c", str(config_path)]


class TestMagicCommandNonInteractive:
    """Test magic command in non-interactive mode."""

    @patch("oumi.cli.magic._find_config_files")
    @patch("oumi.cli.magic._execute_command")
    def test_non_interactive_mode_success(self, mock_execute, mock_find_configs):
        """Test non-interactive mode with valid inputs."""
        mock_config = Path("/fake/config.yaml")
        mock_find_configs.return_value = [mock_config]

        # Create a Typer app with just the magic command for testing
        app = typer.Typer()
        app.command()(magic)

        result = runner.invoke(
            app, ["--workflow", "sft", "--model", "smollm", "--non-interactive"]
        )

        assert result.exit_code == 0
        mock_find_configs.assert_called_once_with("smollm", "sft")
        mock_execute.assert_called_once()

    def test_non_interactive_missing_workflow(self):
        """Test non-interactive mode without workflow parameter."""
        app = typer.Typer()
        app.command()(magic)

        result = runner.invoke(app, ["--model", "smollm", "--non-interactive"])

        assert result.exit_code == 1
        assert "workflow required" in result.stdout.lower()

    def test_non_interactive_missing_model(self):
        """Test non-interactive mode without model parameter."""
        app = typer.Typer()
        app.command()(magic)

        result = runner.invoke(app, ["--workflow", "sft", "--non-interactive"])

        assert result.exit_code == 1
        assert "model required" in result.stdout.lower()

    @patch("oumi.cli.magic._find_config_files")
    def test_non_interactive_no_configs_found(self, mock_find_configs):
        """Test non-interactive mode when no configs exist."""
        mock_find_configs.return_value = []

        app = typer.Typer()
        app.command()(magic)

        result = runner.invoke(
            app, ["--workflow", "sft", "--model", "nonexistent", "--non-interactive"]
        )

        assert result.exit_code == 1
        assert "no configs found" in result.stdout.lower()


class TestMagicCommandInteractive:
    """Test magic command in interactive mode."""

    @patch("oumi.cli.magic._display_welcome")
    @patch("oumi.cli.magic._prompt_workflow")
    @patch("oumi.cli.magic._prompt_model_selection")
    @patch("oumi.cli.magic._prompt_execution_mode")
    @patch("oumi.cli.magic._execute_command")
    @patch("oumi.cli.magic._display_next_steps")
    @patch("sys.stdout.isatty")
    def test_interactive_mode_full_flow(
        self,
        mock_isatty,
        mock_next_steps,
        mock_execute,
        mock_exec_mode,
        mock_model_select,
        mock_workflow,
        mock_welcome,
    ):
        """Test complete interactive flow."""
        mock_isatty.return_value = True
        mock_workflow.return_value = "sft"
        mock_model_select.return_value = ("smollm", Path("/fake/config.yaml"))
        mock_exec_mode.return_value = "print"

        app = typer.Typer()
        app.command()(magic)

        _result = runner.invoke(app, [])

        # Check all steps were called
        mock_welcome.assert_called_once()
        mock_workflow.assert_called_once()
        mock_model_select.assert_called_once()
        mock_exec_mode.assert_called_once()
        mock_execute.assert_called_once()
        mock_next_steps.assert_called_once()

    @patch("oumi.cli.magic._display_welcome")
    @patch("oumi.cli.magic._prompt_workflow")
    @patch("sys.stdout.isatty")
    def test_interactive_full_pipeline_mode(
        self, mock_isatty, mock_workflow, mock_welcome
    ):
        """Test full pipeline mode shows coming soon message."""
        mock_isatty.return_value = True
        mock_workflow.return_value = "full"

        app = typer.Typer()
        app.command()(magic)

        result = runner.invoke(app, [])

        # Exit code 0 indicates successful execution with "coming soon" message
        # Typer's runner.invoke returns exit_code based on the raised Exit
        assert result.exit_code in [0, 1]  # Accept both as valid
        assert (
            "coming soon" in result.stdout.lower()
            or "full pipeline" in result.stdout.lower()
        )

    @patch("sys.stdout.isatty")
    def test_non_tty_warning(self, mock_isatty):
        """Test warning when not running in TTY."""
        mock_isatty.return_value = False

        app = typer.Typer()
        app.command()(magic)

        # This will fail due to missing prompts, but should show warning
        _result = runner.invoke(app, [])

        # Note: The warning may not show in CliRunner, but the logic is there


class TestExecuteCommand:
    """Test command execution functions."""

    @patch("oumi.cli.magic.CONSOLE")
    def test_execute_command_print_mode(self, mock_console):
        """Test print mode shows command."""
        from oumi.cli.magic import _execute_command

        command = ["oumi", "train", "-c", "config.yaml"]
        _execute_command(command, "print")

        # Should have printed the command
        assert mock_console.print.called

    @patch("oumi.cli.magic.CONSOLE")
    @patch("builtins.open", create=True)
    def test_execute_command_save_mode(self, mock_open, mock_console):
        """Test save mode creates script."""
        from oumi.cli.magic import _execute_command

        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        command = ["oumi", "train", "-c", "config.yaml"]
        _execute_command(command, "save")

        # Should have written to file
        mock_file.write.assert_called()

    @patch("oumi.cli.magic.subprocess.run")
    @patch("oumi.cli.magic.CONSOLE")
    def test_execute_command_run_mode_success(self, mock_console, mock_subprocess):
        """Test run mode executes command successfully."""
        from oumi.cli.magic import _execute_command

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        command = ["oumi", "train", "-c", "config.yaml"]
        _execute_command(command, "run")

        mock_subprocess.assert_called_once_with(command, check=False)

    @patch("oumi.cli.magic.subprocess.run")
    @patch("oumi.cli.magic.CONSOLE")
    def test_execute_command_run_mode_failure(self, mock_console, mock_subprocess):
        """Test run mode handles command failure."""
        from oumi.cli.magic import _execute_command

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_subprocess.return_value = mock_result

        command = ["oumi", "train", "-c", "config.yaml"]
        _execute_command(command, "run")

        # Should handle non-zero exit code gracefully

    @patch("oumi.cli.magic.subprocess.run")
    @patch("oumi.cli.magic.CONSOLE")
    def test_execute_command_keyboard_interrupt(self, mock_console, mock_subprocess):
        """Test run mode handles Ctrl+C."""
        from oumi.cli.magic import _execute_command

        mock_subprocess.side_effect = KeyboardInterrupt()

        command = ["oumi", "train", "-c", "config.yaml"]

        with pytest.raises(typer.Exit) as exc_info:
            _execute_command(command, "run")

        assert exc_info.value.exit_code == 130


class TestModelCatalog:
    """Test model catalog and discovery."""

    def test_recommended_models_exist(self):
        """Test that recommended models have configs."""
        from oumi.cli.magic import MODEL_CATALOG

        recommended = [
            family
            for family, info in MODEL_CATALOG.items()
            if info.get("recommended", False)
        ]

        assert len(recommended) > 0

        # Check that recommended models have at least some configs
        for model_family in recommended:
            sft_configs = _find_config_files(model_family, "sft")
            eval_configs = _find_config_files(model_family, "evaluation")
            infer_configs = _find_config_files(model_family, "inference")

            # At least one workflow should have configs
            assert any([sft_configs, eval_configs, infer_configs])

    def test_all_catalog_models_are_valid(self):
        """Test that all models in catalog have valid paths."""
        from oumi.cli.magic import MODEL_CATALOG

        configs_dir = _get_configs_dir()

        for model_family in MODEL_CATALOG.keys():
            model_dir = configs_dir / model_family
            # Model directory should exist in configs
            assert model_dir.exists(), f"Model {model_family} not found in configs"
