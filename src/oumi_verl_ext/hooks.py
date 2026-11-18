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


class _HookedLoader(importlib.abc.Loader):
    """Wraps a loader so that `hook` runs after the module has executed."""

    def __init__(
        self, loader: importlib.abc.Loader, hook: Callable[[ModuleType], None]
    ):
        self._loader = loader
        self._hook = hook

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        self._hook(module)

    def __getattr__(self, name: str):
        return getattr(self._loader, name)


class _PostImportHooks(importlib.abc.MetaPathFinder):
    """Meta path finder that attaches a one-shot hook to registered modules."""

    def __init__(self):
        self._hooks: dict[str, Callable[[ModuleType], None]] = {}

    def register(self, name: str, hook: Callable[[ModuleType], None]) -> None:
        if name in sys.modules:
            hook(sys.modules[name])
            return
        self._hooks[name] = hook
        if self not in sys.meta_path:
            sys.meta_path.insert(0, self)

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        hook = self._hooks.get(fullname)
        if hook is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                del self._hooks[fullname]
                spec.loader = _HookedLoader(spec.loader, hook)
                return spec
        return None


_HOOKS = _PostImportHooks()


def register_post_import_hook(
    module_name: str, hook: Callable[[ModuleType], None]
) -> None:
    """Runs `hook(module)` once, right after `module_name` finishes importing.

    Runs immediately if the module is already imported. Registering never imports
    the module.
    """
    _HOOKS.register(module_name, hook)
