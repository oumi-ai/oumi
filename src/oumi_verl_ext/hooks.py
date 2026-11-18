# Copyright 2026 - Oumi
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

"""Post-import hooks: run a callback right after a module finishes importing."""

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Callable, Sequence
from types import ModuleType

_Hook = Callable[[ModuleType], None]


class _HookedLoader(importlib.abc.Loader):
    """Wraps a loader so the module's registered hooks run after it executes."""

    def __init__(
        self, loader: importlib.abc.Loader, registry: dict[str, list[_Hook]], name: str
    ):
        self._loader = loader
        self._registry = registry
        self._name = name

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        for hook in self._registry.pop(self._name, []):
            hook(module)

    def __getattr__(self, name: str):
        return getattr(self._loader, name)


class _PostImportHooks(importlib.abc.MetaPathFinder):
    """Meta path finder that runs registered hooks once a module has loaded.

    Hooks stay registered until the module executes successfully, so a spec lookup
    that never imports the module, or an import that fails, does not lose them.
    """

    def __init__(self):
        self._hooks: dict[str, list[_Hook]] = {}

    def register(self, name: str, hook: _Hook) -> None:
        if name in sys.modules:
            hook(sys.modules[name])
            return
        self._hooks.setdefault(name, []).append(hook)
        if self not in sys.meta_path:
            sys.meta_path.insert(0, self)

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in self._hooks:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _HookedLoader(spec.loader, self._hooks, fullname)
                return spec
        return None


_HOOKS = _PostImportHooks()


def register_post_import_hook(
    module_name: str, hook: Callable[[ModuleType], None]
) -> None:
    """Runs `hook(module)` once, right after `module_name` finishes importing.

    Runs immediately if the module is already imported. Hooks for one module run in
    registration order. Registering never imports the module.
    """
    _HOOKS.register(module_name, hook)
