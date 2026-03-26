import importlib
import types
from unittest.mock import patch

import pytest

from oumi.core.registry import REGISTRY, Registry, RegistryType
from oumi.core.registry.registry import _load_plugin_registry_modules
from oumi.plugins.discovery import PluginInfo, _clear_cache


@pytest.fixture(autouse=True)
def clear_plugin_cache():
    _clear_cache()
    yield
    _clear_cache()


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("OUMI_EXTRA_DEPS_FILE", "")


@pytest.fixture(autouse=True)
def cleanup():
    """Snapshot and restore registry around each test."""
    snapshot = Registry()
    for reg_type in RegistryType:
        for key, value in REGISTRY.get_all(reg_type).items():
            snapshot.register(key, reg_type, value)
    REGISTRY.clear()
    REGISTRY._initialized = False
    yield
    REGISTRY.clear()
    REGISTRY._initialized = False
    for reg_type in RegistryType:
        for key, value in snapshot.get_all(reg_type).items():
            REGISTRY.register(key, reg_type, value)


def _make_plugin_info(
    name="test_plugin",
    registry_modules=None,
    error=None,
):
    return PluginInfo(
        entry_point_name=name,
        module=types.ModuleType(name) if not error else None,
        package_name=f"oumi-{name}",
        package_version="0.1.0",
        registry_modules=registry_modules or [],
        error=error,
    )


class TestLoadPluginRegistryModules:
    def test_imports_plugin_modules(self, monkeypatch):
        plugin = _make_plugin_info(registry_modules=["fake_plugin.datasets"])
        imported = []
        original_import = importlib.import_module

        def mock_import(name, *args, **kwargs):
            if name == "fake_plugin.datasets":
                imported.append(name)
                return types.ModuleType(name)
            return original_import(name, *args, **kwargs)

        with patch(
            "oumi.plugins.discovery.discover_plugins",
            return_value=[plugin],
        ):
            monkeypatch.setattr(importlib, "import_module", mock_import)
            _load_plugin_registry_modules()

        assert "fake_plugin.datasets" in imported

    def test_multiple_registry_modules(self, monkeypatch):
        plugin = _make_plugin_info(
            registry_modules=["mod.data", "mod.models"],
        )
        imported = []
        original_import = importlib.import_module

        def mock_import(name, *args, **kwargs):
            if name.startswith("mod."):
                imported.append(name)
                return types.ModuleType(name)
            return original_import(name, *args, **kwargs)

        with patch(
            "oumi.plugins.discovery.discover_plugins",
            return_value=[plugin],
        ):
            monkeypatch.setattr(importlib, "import_module", mock_import)
            _load_plugin_registry_modules()

        assert "mod.data" in imported
        assert "mod.models" in imported

    def test_skips_error_plugins(self, monkeypatch):
        plugin = _make_plugin_info(
            name="broken",
            registry_modules=["broken.data"],
            error="load failed",
        )
        imported = []
        original_import = importlib.import_module

        def mock_import(name, *args, **kwargs):
            if name == "broken.data":
                imported.append(name)
                return types.ModuleType(name)
            return original_import(name, *args, **kwargs)

        with patch(
            "oumi.plugins.discovery.discover_plugins",
            return_value=[plugin],
        ):
            monkeypatch.setattr(importlib, "import_module", mock_import)
            _load_plugin_registry_modules()

        assert imported == []

    def test_failed_import_warns_and_continues(self, caplog, monkeypatch):
        plugin1 = _make_plugin_info(name="p1", registry_modules=["p1.bad"])
        plugin2 = _make_plugin_info(name="p2", registry_modules=["p2.good"])
        imported = []
        original_import = importlib.import_module

        def mock_import(name, *args, **kwargs):
            if name == "p1.bad":
                raise ImportError("no such module: p1.bad")
            if name == "p2.good":
                imported.append(name)
                return types.ModuleType(name)
            return original_import(name, *args, **kwargs)

        with patch(
            "oumi.plugins.discovery.discover_plugins",
            return_value=[plugin1, plugin2],
        ):
            monkeypatch.setattr(importlib, "import_module", mock_import)
            _load_plugin_registry_modules()

        assert "p2.good" in imported
        assert "failed to import registry module" in caplog.text.lower()

    def test_no_plugins_is_noop(self, monkeypatch):
        imported = []
        original_import = importlib.import_module

        def mock_import(name, *args, **kwargs):
            imported.append(name)
            return original_import(name, *args, **kwargs)

        with patch(
            "oumi.plugins.discovery.discover_plugins",
            return_value=[],
        ):
            monkeypatch.setattr(importlib, "import_module", mock_import)
            _load_plugin_registry_modules()

        assert imported == []

    def test_plugin_without_registry_modules_skipped(self, monkeypatch):
        plugin = _make_plugin_info(registry_modules=[])
        imported = []
        original_import = importlib.import_module

        def mock_import(name, *args, **kwargs):
            imported.append(name)
            return original_import(name, *args, **kwargs)

        with patch(
            "oumi.plugins.discovery.discover_plugins",
            return_value=[plugin],
        ):
            monkeypatch.setattr(importlib, "import_module", mock_import)
            _load_plugin_registry_modules()

        assert imported == []


class TestRegistryInitWithPlugins:
    def test_plugins_loaded_during_registry_init(self, monkeypatch):
        """Verify plugins are loaded when the registry initializes."""
        plugin = _make_plugin_info(registry_modules=["fake_plugin.datasets"])
        imported = []
        original_import = importlib.import_module

        def mock_import(name, *args, **kwargs):
            if name == "fake_plugin.datasets":
                imported.append(name)
                return types.ModuleType(name)
            return original_import(name, *args, **kwargs)

        with patch(
            "oumi.plugins.discovery.discover_plugins",
            return_value=[plugin],
        ):
            monkeypatch.setattr(importlib, "import_module", mock_import)
            # Trigger registry initialization by accessing it
            REGISTRY.contains("anything", RegistryType.DATASET)

        assert "fake_plugin.datasets" in imported
