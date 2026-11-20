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

"""Runs a callback right after a module finishes importing."""

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Callable, Sequence
from types import ModuleType

_Hook = Callable[[ModuleType], None]


class _HookedLoader(importlib.abc.Loader):
    """Wraps a loader to call `on_exec(module)` after the module executes."""

    def __init__(self, loader: importlib.abc.Loader, on_exec: _Hook):
        self._loader = loader
        self._on_exec = on_exec

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        self._on_exec(module)

    def __getattr__(self, name: str):
        return getattr(self._loader, name)


class _WaitForModule(importlib.abc.MetaPathFinder):
    """Meta path finder that runs one hook once one module has executed.

    The hook stays armed until the module executes successfully, so a spec lookup
    that never imports the module, or an import that fails, does not lose it.
    """

    def __init__(self, module_name: str, hook: _Hook):
        self._module_name = module_name
        self._hook: _Hook | None = hook

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != self._module_name or self._hook is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _HookedLoader(spec.loader, self._run_hook)
                return spec
        return None

    def _run_hook(self, module: ModuleType) -> None:
        hook, self._hook = self._hook, None
        if hook is not None:
            hook(module)


def register_post_import_hook(module_name: str, hook: _Hook) -> None:
    """Runs `hook(module)` once, right after `module_name` finishes importing.

    Runs immediately if the module is already imported. Registering never imports
    the module.
    """
    if module_name in sys.modules:
        hook(sys.modules[module_name])
    else:
        sys.meta_path.insert(0, _WaitForModule(module_name, hook))
