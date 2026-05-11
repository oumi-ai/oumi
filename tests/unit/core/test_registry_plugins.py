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

"""Tests for entry-point plugin discovery in :mod:`oumi.core.registry`."""

import types
from unittest.mock import MagicMock, patch

from oumi.core.registry import registry as registry_mod


class _FakeEntryPoint:
    """Stub for :class:`importlib.metadata.EntryPoint`."""

    def __init__(self, name: str, value: str, load_result):
        self.name = name
        self.value = value
        self._load_result = load_result
        self.loaded = False

    def load(self):
        self.loaded = True
        if isinstance(self._load_result, BaseException):
            raise self._load_result
        return self._load_result


class _FakeEntryPoints:
    """Stub for the new-style entry-points API."""

    def __init__(self, by_group):
        self._by_group = by_group

    def select(self, *, group):
        return self._by_group.get(group, [])


def test_load_entry_point_plugins_imports_each_entry():
    fake_module = types.ModuleType("fake_plugin")
    ep = _FakeEntryPoint("fake", "fake_plugin:none", fake_module)
    fake_eps = _FakeEntryPoints({"oumi.plugins": [ep]})

    with patch.object(
        registry_mod.importlib.metadata, "entry_points", return_value=fake_eps
    ):
        registry_mod._load_entry_point_plugins()

    assert ep.loaded


def test_load_entry_point_plugins_logs_and_skips_on_failure():
    boom = ImportError("dependency missing")
    ep = _FakeEntryPoint("broken", "broken_plugin:none", boom)
    fake_eps = _FakeEntryPoints({"oumi.plugins": [ep]})

    with patch.object(
        registry_mod.importlib.metadata, "entry_points", return_value=fake_eps
    ):
        # Must not raise; one broken plugin can't break the rest of oumi.
        registry_mod._load_entry_point_plugins()

    assert ep.loaded  # we tried


def test_load_entry_point_plugins_handles_old_api():
    """Pre-3.10 importlib.metadata returns a dict-like object without .select."""
    fake_module = types.ModuleType("legacy_plugin")
    ep = _FakeEntryPoint("legacy", "legacy_plugin:none", fake_module)
    legacy_eps = MagicMock(spec=[])  # no .select attribute
    legacy_eps.get = MagicMock(return_value=[ep])

    with patch.object(
        registry_mod.importlib.metadata, "entry_points", return_value=legacy_eps
    ):
        registry_mod._load_entry_point_plugins()

    assert ep.loaded
    legacy_eps.get.assert_called_with("oumi.plugins", [])


def test_load_entry_point_plugins_swallows_entry_points_error():
    """If importlib.metadata itself blows up, we log and move on."""
    with patch.object(
        registry_mod.importlib.metadata,
        "entry_points",
        side_effect=RuntimeError("boom"),
    ):
        # Must not raise.
        registry_mod._load_entry_point_plugins()


def test_registry_loads_plugins_on_first_use():
    """First Registry access should trigger _load_entry_point_plugins exactly once."""
    fake_module = types.ModuleType("plugin_first_use")
    ep = _FakeEntryPoint("p", "plugin_first_use:none", fake_module)
    fake_eps = _FakeEntryPoints({"oumi.plugins": [ep]})

    fresh_registry = registry_mod.Registry()

    with patch.object(
        registry_mod.importlib.metadata, "entry_points", return_value=fake_eps
    ):
        # Any read access triggers the lazy bootstrap.
        fresh_registry.get_all(registry_mod.RegistryType.DATASET)

    assert ep.loaded
