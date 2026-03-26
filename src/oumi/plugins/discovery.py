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

"""Plugin discovery via Python entry points.

Third-party packages register as oumi plugins by declaring an entry point
in the ``oumi.plugins`` group. The entry point value must be a Python module.

Convention for plugin modules:
    - ``register_cli(app: typer.Typer)``: called during CLI construction (optional).
    - ``REGISTRY_MODULES: list[str]``: modules imported lazily during registry
      initialization to trigger ``@register_*`` decorator side effects (optional).
"""

import importlib
import importlib.metadata
import types
from collections.abc import Callable
from dataclasses import dataclass, field

from oumi.utils.logging import logger

ENTRY_POINT_GROUP = "oumi.plugins"


@dataclass
class PluginInfo:
    """Metadata for a discovered plugin."""

    entry_point_name: str
    """Name from the entry point declaration (e.g., ``chat``)."""

    module: types.ModuleType | None = None
    """The loaded plugin module, or ``None`` if loading failed."""

    package_name: str | None = None
    """Distribution package name (e.g., ``oumi-chat``)."""

    package_version: str | None = None
    """Distribution package version."""

    register_cli_fn: Callable | None = None
    """The ``register_cli`` function from the plugin module, if present."""

    registry_modules: list[str] = field(default_factory=list)
    """Module paths to import for ``@register_*`` side effects."""

    error: str | None = None
    """Error message if loading failed."""


_cached_plugins: list[PluginInfo] | None = None


def discover_plugins() -> list[PluginInfo]:
    """Discover and load all installed oumi plugins.

    Plugins are discovered via the ``oumi.plugins`` entry point group.
    Results are cached after the first call.

    Returns:
        List of :class:`PluginInfo` for every discovered entry point.
    """
    global _cached_plugins
    if _cached_plugins is not None:
        return _cached_plugins
    _cached_plugins = _discover_plugins_impl()
    return _cached_plugins


def _clear_cache() -> None:
    """Clear the discovery cache. Intended for tests only."""
    global _cached_plugins
    _cached_plugins = None


def _discover_plugins_impl() -> list[PluginInfo]:
    """Scan entry points and load each plugin."""
    eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    plugins: list[PluginInfo] = []
    for ep in eps:
        plugins.append(_load_plugin(ep))
    return plugins


def _load_plugin(ep: importlib.metadata.EntryPoint) -> PluginInfo:
    """Load a single plugin from an entry point.

    Never raises; returns a ``PluginInfo`` with ``error`` set on failure.
    """
    try:
        obj = ep.load()
    except Exception as exc:
        logger.warning(
            "Plugin '%s' failed to load: %s. Skipping.",
            ep.name,
            exc,
        )
        return PluginInfo(
            entry_point_name=ep.name,
            package_name=_ep_dist_name(ep),
            package_version=_ep_dist_version(ep),
            error=str(exc),
        )

    if not isinstance(obj, types.ModuleType):
        msg = (
            f"Plugin '{ep.name}' entry point must resolve to a module, "
            f"got {type(obj).__name__}."
        )
        logger.warning(msg)
        return PluginInfo(
            entry_point_name=ep.name,
            package_name=_ep_dist_name(ep),
            package_version=_ep_dist_version(ep),
            error=msg,
        )

    register_cli_fn = _extract_register_cli(ep.name, obj)
    registry_modules = _extract_registry_modules(ep.name, obj)

    return PluginInfo(
        entry_point_name=ep.name,
        module=obj,
        package_name=_ep_dist_name(ep),
        package_version=_ep_dist_version(ep),
        register_cli_fn=register_cli_fn,
        registry_modules=registry_modules,
    )


def _extract_register_cli(
    plugin_name: str, module: types.ModuleType
) -> Callable | None:
    """Extract ``register_cli`` from a plugin module if present and callable."""
    fn = getattr(module, "register_cli", None)
    if fn is None:
        return None
    if not callable(fn):
        logger.warning(
            "Plugin '%s': 'register_cli' is not callable. Ignoring.",
            plugin_name,
        )
        return None
    return fn


def _extract_registry_modules(
    plugin_name: str, module: types.ModuleType
) -> list[str]:
    """Extract ``REGISTRY_MODULES`` list from a plugin module if present."""
    raw = getattr(module, "REGISTRY_MODULES", None)
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(m, str) for m in raw):
        logger.warning(
            "Plugin '%s': 'REGISTRY_MODULES' must be a list of strings. Ignoring.",
            plugin_name,
        )
        return []
    return raw


def _ep_dist_name(ep: importlib.metadata.EntryPoint) -> str | None:
    """Safely extract distribution name from an entry point."""
    try:
        return ep.dist.name if ep.dist else None
    except Exception:
        return None


def _ep_dist_version(ep: importlib.metadata.EntryPoint) -> str | None:
    """Safely extract distribution version from an entry point."""
    try:
        return ep.dist.version if ep.dist else None
    except Exception:
        return None
