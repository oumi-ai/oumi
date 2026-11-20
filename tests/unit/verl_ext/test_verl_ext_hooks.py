import importlib
import importlib.util
import sys

import pytest

from oumi_verl_ext.hooks import register_post_import_hook


@pytest.fixture(autouse=True)
def _isolate_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    for name in ("fake_mod", "fakepkg", "fakepkg.sub"):
        sys.modules.pop(name, None)


def test_hook_runs_once_after_a_dotted_module_finishes_loading(tmp_path):
    (tmp_path / "fakepkg").mkdir()
    (tmp_path / "fakepkg" / "__init__.py").write_text("")
    (tmp_path / "fakepkg" / "sub.py").write_text("VALUE = 7\n")
    seen = []
    register_post_import_hook("fakepkg.sub", lambda module: seen.append(module.VALUE))
    assert "fakepkg.sub" not in sys.modules

    importlib.reload(importlib.import_module("fakepkg.sub"))

    assert seen == [7]


def test_hook_runs_at_once_for_an_already_imported_module():
    calls = []
    register_post_import_hook("sys", calls.append)
    assert calls == [sys]


def test_a_spec_probe_before_the_import_keeps_the_hook(tmp_path):
    (tmp_path / "fake_mod.py").write_text("")
    calls = []
    register_post_import_hook("fake_mod", calls.append)

    assert importlib.util.find_spec("fake_mod") is not None
    module = importlib.import_module("fake_mod")

    assert calls == [module]


def test_a_failed_import_keeps_the_hook_for_the_retry(tmp_path):
    flag = tmp_path / "ready"
    (tmp_path / "fake_mod.py").write_text(
        f"import pathlib\nassert pathlib.Path({str(flag)!r}).exists()\n"
    )
    calls = []
    register_post_import_hook("fake_mod", calls.append)

    with pytest.raises(AssertionError):
        importlib.import_module("fake_mod")
    flag.write_text("")
    module = importlib.import_module("fake_mod")

    assert calls == [module]
