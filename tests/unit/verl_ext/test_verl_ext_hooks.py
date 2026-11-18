import importlib
import importlib.util
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
    for name in ("fake_mod_a", "fake_mod_b", "fakepkg", "fakepkg.sub"):
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


def test_finding_the_spec_before_importing_keeps_the_hook(tmp_path, monkeypatch):
    (tmp_path / "fake_mod_a.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    calls = []
    register_post_import_hook("fake_mod_a", calls.append)

    assert importlib.util.find_spec("fake_mod_a") is not None
    module = importlib.import_module("fake_mod_a")

    assert calls == [module]


def test_hooks_for_one_module_run_in_registration_order(tmp_path, monkeypatch):
    (tmp_path / "fake_mod_a.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    order = []
    register_post_import_hook("fake_mod_a", lambda module: order.append("first"))
    register_post_import_hook("fake_mod_a", lambda module: order.append("second"))

    importlib.import_module("fake_mod_a")

    assert order == ["first", "second"]


def test_hook_survives_a_failed_import_and_runs_once_on_retry(tmp_path, monkeypatch):
    flag = tmp_path / "ready"
    (tmp_path / "fake_mod_a.py").write_text(
        f"import pathlib\nassert pathlib.Path({str(flag)!r}).exists()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    calls = []
    register_post_import_hook("fake_mod_a", calls.append)

    with pytest.raises(AssertionError):
        importlib.import_module("fake_mod_a")
    assert calls == []

    flag.write_text("")
    module = importlib.import_module("fake_mod_a")
    assert calls == [module]


def test_hook_works_for_a_dotted_submodule(tmp_path, monkeypatch):
    package = tmp_path / "fakepkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "sub.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    calls = []
    register_post_import_hook("fakepkg.sub", calls.append)

    module = importlib.import_module("fakepkg.sub")

    assert calls == [module]
