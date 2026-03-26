import types
from unittest.mock import Mock

import pytest

from oumi.plugins.discovery import (
    ENTRY_POINT_GROUP,
    PluginInfo,
    _clear_cache,
    _discover_plugins_impl,
    _load_plugin,
    discover_plugins,
)


@pytest.fixture(autouse=True)
def clear_plugin_cache():
    """Clear discovery cache before and after each test."""
    _clear_cache()
    yield
    _clear_cache()


def _make_entry_point(name="test_plugin", module=None, raises=None, dist_name=None):
    """Create a mock entry point."""
    ep = Mock()
    ep.name = name
    if raises:
        ep.load.side_effect = raises
    else:
        ep.load.return_value = module
    ep.dist = Mock()
    ep.dist.name = dist_name or f"oumi-{name}"
    ep.dist.version = "0.1.0"
    return ep


def _make_plugin_module(
    name="fake_plugin",
    register_cli=None,
    registry_modules=None,
):
    """Create a mock module that looks like a plugin."""
    mod = types.ModuleType(name)
    if register_cli is not None:
        mod.register_cli = register_cli
    if registry_modules is not None:
        mod.REGISTRY_MODULES = registry_modules
    return mod


class TestPluginInfo:
    def test_defaults(self):
        info = PluginInfo(entry_point_name="test")
        assert info.entry_point_name == "test"
        assert info.module is None
        assert info.package_name is None
        assert info.package_version is None
        assert info.register_cli_fn is None
        assert info.registry_modules == []
        assert info.error is None


class TestLoadPlugin:
    def test_valid_plugin_with_cli_and_registry(self):
        cli_fn = Mock()
        mod = _make_plugin_module(
            register_cli=cli_fn,
            registry_modules=["fake_plugin.datasets"],
        )
        ep = _make_entry_point(module=mod)

        info = _load_plugin(ep)

        assert info.error is None
        assert info.module is mod
        assert info.register_cli_fn is cli_fn
        assert info.registry_modules == ["fake_plugin.datasets"]
        assert info.package_name == "oumi-test_plugin"
        assert info.package_version == "0.1.0"

    def test_plugin_cli_only(self):
        cli_fn = Mock()
        mod = _make_plugin_module(register_cli=cli_fn)
        ep = _make_entry_point(module=mod)

        info = _load_plugin(ep)

        assert info.error is None
        assert info.register_cli_fn is cli_fn
        assert info.registry_modules == []

    def test_plugin_registry_only(self):
        mod = _make_plugin_module(registry_modules=["mod.datasets"])
        ep = _make_entry_point(module=mod)

        info = _load_plugin(ep)

        assert info.error is None
        assert info.register_cli_fn is None
        assert info.registry_modules == ["mod.datasets"]

    def test_bare_plugin_no_cli_no_registry(self):
        mod = _make_plugin_module()
        ep = _make_entry_point(module=mod)

        info = _load_plugin(ep)

        assert info.error is None
        assert info.register_cli_fn is None
        assert info.registry_modules == []

    def test_load_failure_returns_error(self, caplog):
        ep = _make_entry_point(raises=ImportError("no such module"))

        info = _load_plugin(ep)

        assert info.error is not None
        assert "no such module" in info.error
        assert info.module is None
        assert "failed to load" in caplog.text.lower()

    def test_non_module_returns_error(self, caplog):
        ep = _make_entry_point(module="not a module")

        info = _load_plugin(ep)

        assert info.error is not None
        assert "must resolve to a module" in info.error

    def test_register_cli_not_callable_ignored(self, caplog):
        mod = _make_plugin_module()
        mod.register_cli = "not callable"
        ep = _make_entry_point(module=mod)

        info = _load_plugin(ep)

        assert info.error is None
        assert info.register_cli_fn is None
        assert "not callable" in caplog.text.lower()

    def test_registry_modules_not_list_ignored(self, caplog):
        mod = _make_plugin_module()
        mod.REGISTRY_MODULES = "not a list"
        ep = _make_entry_point(module=mod)

        info = _load_plugin(ep)

        assert info.error is None
        assert info.registry_modules == []
        assert "must be a list" in caplog.text.lower()

    def test_registry_modules_non_string_items_ignored(self, caplog):
        mod = _make_plugin_module()
        mod.REGISTRY_MODULES = [123, 456]
        ep = _make_entry_point(module=mod)

        info = _load_plugin(ep)

        assert info.registry_modules == []

    def test_dist_missing_handled(self):
        mod = _make_plugin_module()
        ep = _make_entry_point(module=mod)
        ep.dist = None

        info = _load_plugin(ep)

        assert info.error is None
        assert info.package_name is None
        assert info.package_version is None


class TestDiscoverPlugins:
    def test_no_plugins(self, monkeypatch):
        monkeypatch.setattr(
            "oumi.plugins.discovery.importlib.metadata.entry_points",
            lambda group: [],
        )
        assert discover_plugins() == []

    def test_discovers_valid_plugin(self, monkeypatch):
        cli_fn = Mock()
        mod = _make_plugin_module(register_cli=cli_fn)
        ep = _make_entry_point(name="chat", module=mod)

        monkeypatch.setattr(
            "oumi.plugins.discovery.importlib.metadata.entry_points",
            lambda group: [ep],
        )

        result = discover_plugins()
        assert len(result) == 1
        assert result[0].entry_point_name == "chat"
        assert result[0].register_cli_fn is cli_fn

    def test_broken_plugin_included_with_error(self, monkeypatch, caplog):
        ep = _make_entry_point(name="broken", raises=ImportError("boom"))

        monkeypatch.setattr(
            "oumi.plugins.discovery.importlib.metadata.entry_points",
            lambda group: [ep],
        )

        result = discover_plugins()
        assert len(result) == 1
        assert result[0].error is not None

    def test_caching(self, monkeypatch):
        call_count = 0

        def mock_eps(group):
            nonlocal call_count
            call_count += 1
            return []

        monkeypatch.setattr(
            "oumi.plugins.discovery.importlib.metadata.entry_points",
            mock_eps,
        )

        discover_plugins()
        discover_plugins()
        assert call_count == 1

    def test_cache_cleared(self, monkeypatch):
        call_count = 0

        def mock_eps(group):
            nonlocal call_count
            call_count += 1
            return []

        monkeypatch.setattr(
            "oumi.plugins.discovery.importlib.metadata.entry_points",
            mock_eps,
        )

        discover_plugins()
        _clear_cache()
        discover_plugins()
        assert call_count == 2

    def test_multiple_plugins(self, monkeypatch):
        mod1 = _make_plugin_module(name="p1", register_cli=Mock())
        mod2 = _make_plugin_module(name="p2", registry_modules=["p2.data"])
        ep1 = _make_entry_point(name="p1", module=mod1)
        ep2 = _make_entry_point(name="p2", module=mod2)

        monkeypatch.setattr(
            "oumi.plugins.discovery.importlib.metadata.entry_points",
            lambda group: [ep1, ep2],
        )

        result = discover_plugins()
        assert len(result) == 2
        assert result[0].entry_point_name == "p1"
        assert result[1].entry_point_name == "p2"


class TestDiscoverPluginsImpl:
    def test_returns_fresh_list(self, monkeypatch):
        monkeypatch.setattr(
            "oumi.plugins.discovery.importlib.metadata.entry_points",
            lambda group: [],
        )
        result = _discover_plugins_impl()
        assert result == []

    def test_entry_point_group(self, monkeypatch):
        """Verify we use the correct entry point group."""
        assert ENTRY_POINT_GROUP == "oumi.plugins"

        captured_group = None

        def mock_eps(group):
            nonlocal captured_group
            captured_group = group
            return []

        monkeypatch.setattr(
            "oumi.plugins.discovery.importlib.metadata.entry_points",
            mock_eps,
        )
        _discover_plugins_impl()
        assert captured_group == "oumi.plugins"
