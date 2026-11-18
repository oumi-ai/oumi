import importlib
import sys

import pytest

from oumi_verl_ext import hooks
from oumi_verl_ext.hooks import register_post_import_hook


@pytest.fixture(autouse=True)
def _restore_import_state(monkeypatch):
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(hooks, "_HOOKS", hooks._PostImportHooks())
    yield
    for name in ("fake_mod_a", "fake_mod_b"):
        sys.modules.pop(name, None)


def test_hook_runs_once_after_the_module_loads(tmp_path, monkeypatch):
    (tmp_path / "fake_mod_a.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    calls = []
    register_post_import_hook("fake_mod_a", calls.append)
    assert "fake_mod_a" not in sys.modules

    import fake_mod_a  # type: ignore[import-not-found]

    importlib.reload(fake_mod_a)
    assert calls == [fake_mod_a]


def test_hook_sees_the_fully_executed_module(tmp_path, monkeypatch):
    (tmp_path / "fake_mod_b.py").write_text("VALUE = 7\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = []
    register_post_import_hook("fake_mod_b", lambda module: seen.append(module.VALUE))

    importlib.import_module("fake_mod_b")

    assert seen == [7]


def test_hook_runs_at_once_for_an_already_imported_module():
    calls = []
    register_post_import_hook("sys", calls.append)
    assert calls == [sys]
