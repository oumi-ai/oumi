import types
from unittest.mock import Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from oumi.cli.main import (
    _get_registered_command_names,
    _register_plugin_commands,
    get_app,
)
from oumi.plugins.discovery import PluginInfo, _clear_cache

runner = CliRunner()


@pytest.fixture(autouse=True)
def clear_plugin_cache():
    _clear_cache()
    yield
    _clear_cache()


def _make_plugin_info(
    name="test_plugin",
    register_cli_fn=None,
    registry_modules=None,
    error=None,
):
    return PluginInfo(
        entry_point_name=name,
        module=types.ModuleType(name) if not error else None,
        package_name=f"oumi-{name}",
        package_version="0.1.0",
        register_cli_fn=register_cli_fn,
        registry_modules=registry_modules or [],
        error=error,
    )


class TestGetRegisteredCommandNames:
    def test_explicit_names(self):
        app = typer.Typer()

        def my_cmd():
            pass

        app.command(name="hello")(my_cmd)
        names = _get_registered_command_names(app)
        assert "hello" in names

    def test_auto_derived_names(self):
        app = typer.Typer()

        def my_fancy_cmd():
            pass

        app.command()(my_fancy_cmd)
        names = _get_registered_command_names(app)
        assert "my-fancy-cmd" in names

    def test_group_names(self):
        app = typer.Typer()
        sub = typer.Typer()
        app.add_typer(sub, name="mygroup")
        names = _get_registered_command_names(app)
        assert "mygroup" in names


class TestRegisterPluginCommands:
    def test_plugin_command_registered(self):
        app = typer.Typer()

        def chat():
            pass

        def register_cli(a):
            a.command()(chat)

        plugin = _make_plugin_info(register_cli_fn=register_cli)

        with patch(
            "oumi.plugins.discovery.discover_plugins", return_value=[plugin]
        ):
            _register_plugin_commands(app)

        assert "chat" in _get_registered_command_names(app)

    def test_collision_with_core_command_rejected(self, caplog):
        app = typer.Typer()

        def train():
            pass

        # Register a "core" command first
        app.command(name="train")(train)

        def plugin_train():
            pass

        def register_cli(a):
            a.command(name="train")(plugin_train)

        plugin = _make_plugin_info(name="bad_plugin", register_cli_fn=register_cli)

        with patch(
            "oumi.plugins.discovery.discover_plugins", return_value=[plugin]
        ):
            _register_plugin_commands(app)

        # Core command names should still be the original
        names = _get_registered_command_names(app)
        assert "train" in names
        # Only the original command should remain (1 command)
        assert len(app.registered_commands) == 1
        assert "collide" in caplog.text.lower()

    def test_plugin_cli_error_skipped(self, caplog):
        app = typer.Typer()

        def register_cli(a):
            raise RuntimeError("plugin broke")

        plugin = _make_plugin_info(name="broken", register_cli_fn=register_cli)

        with patch(
            "oumi.plugins.discovery.discover_plugins", return_value=[plugin]
        ):
            _register_plugin_commands(app)

        assert "failed during cli registration" in caplog.text.lower()

    def test_error_plugin_skipped(self):
        app = typer.Typer()
        plugin = _make_plugin_info(name="err", error="some error")

        with patch(
            "oumi.plugins.discovery.discover_plugins", return_value=[plugin]
        ):
            _register_plugin_commands(app)

        # No commands should be added
        assert _get_registered_command_names(app) == set()

    def test_plugin_without_cli_skipped(self):
        app = typer.Typer()
        plugin = _make_plugin_info(name="reg_only", register_cli_fn=None)

        with patch(
            "oumi.plugins.discovery.discover_plugins", return_value=[plugin]
        ):
            _register_plugin_commands(app)

        assert _get_registered_command_names(app) == set()


class TestPluginsCommand:
    def test_plugins_command_no_plugins(self):
        # plugins_cmd does a top-level import, so patch there.
        # _register_plugin_commands does a local import, so patch at source.
        with patch(
            "oumi.cli.plugins_cmd.discover_plugins",
            return_value=[],
        ), patch(
            "oumi.plugins.discovery.discover_plugins",
            return_value=[],
        ):
            app = get_app()
            result = runner.invoke(app, ["plugins"])
            assert result.exit_code == 0
            assert "no plugins installed" in result.output.lower()

    def test_plugins_command_shows_plugin(self):
        plugin = _make_plugin_info(
            name="chat",
            register_cli_fn=Mock(),
            registry_modules=["oumi_chat.datasets"],
        )
        with patch(
            "oumi.cli.plugins_cmd.discover_plugins",
            return_value=[plugin],
        ), patch(
            "oumi.plugins.discovery.discover_plugins",
            return_value=[plugin],
        ):
            app = get_app()
            result = runner.invoke(app, ["plugins"])
            assert result.exit_code == 0
            assert "chat" in result.output
            assert "oumi_chat.datasets" in result.output

    def test_plugins_command_shows_error(self):
        plugin = _make_plugin_info(name="broken", error="import failed")
        with patch(
            "oumi.cli.plugins_cmd.discover_plugins",
            return_value=[plugin],
        ), patch(
            "oumi.plugins.discovery.discover_plugins",
            return_value=[plugin],
        ):
            app = get_app()
            result = runner.invoke(app, ["plugins"])
            assert result.exit_code == 0
            assert "error" in result.output.lower()
            assert "import failed" in result.output
