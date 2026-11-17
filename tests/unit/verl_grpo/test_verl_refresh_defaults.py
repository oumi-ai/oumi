import hashlib
import stat
import sys
from pathlib import Path

from oumi.verl_grpo.refresh_defaults import main, refresh_defaults


def _write_wrapper(path: Path, python_args: str, env_prefix: str = "") -> Path:
    path.write_text(
        f'#!/bin/sh\n{env_prefix}exec "{sys.executable}" {python_args} "$@"\n'
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _make_fake_verl(root: Path, generated: str) -> Path:
    config_dir = root / "verl" / "trainer" / "config"
    config_dir.mkdir(parents=True)
    (root / "verl" / "__init__.py").write_text("raise ImportError('imported verl')")
    generated_file = config_dir / "_generated_ppo_trainer.yaml"
    generated_file.write_text(generated)
    dist_info = root / "verl-1.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: verl\nVersion: 1.1\n"
    )
    return generated_file


def test_refresh_replaces_the_copy_and_reports_key_changes(tmp_path):
    defaults_dir = tmp_path / "defaults"
    defaults_dir.mkdir()
    (defaults_dir / "ppo_trainer_defaults_1.0.yaml").write_text(
        "a: 1\nb: {c: 2, d: 3}\n"
    )
    fake_generated = _make_fake_verl(tmp_path / "site", "a: 1\nb: {c: 5, e: 4}\n")
    fake_python = _write_wrapper(
        tmp_path / "fake_python",
        "",
        env_prefix=f'PYTHONPATH="{tmp_path / "site"}" ',
    )

    change = refresh_defaults(str(fake_python), defaults_dir=defaults_dir)

    assert (change.version, change.added, change.removed, change.changed) == (
        "1.1",
        ["b.e"],
        ["b.d"],
        ["b.c"],
    )
    assert [p.name for p in defaults_dir.glob("ppo_trainer_defaults_*.yaml")] == [
        "ppo_trainer_defaults_1.1.yaml"
    ]
    copied = defaults_dir / "ppo_trainer_defaults_1.1.yaml"
    assert copied.read_bytes() == fake_generated.read_bytes()
    assert change.sha256 == hashlib.sha256(fake_generated.read_bytes()).hexdigest()


def test_refresh_main_fails_for_a_python_without_verl(tmp_path, capsys):
    isolated_python = _write_wrapper(tmp_path / "isolated_python", "-I -S")
    assert main(["--verl-python", str(isolated_python)]) == 1
    assert "no verl" in capsys.readouterr().err


def test_refresh_main_fails_cleanly_for_a_missing_interpreter(tmp_path, capsys):
    assert main(["--verl-python", str(tmp_path / "no_such_python")]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_refresh_ignores_stray_output_before_the_verl_location(tmp_path):
    defaults_dir = tmp_path / "defaults"
    defaults_dir.mkdir()
    _make_fake_verl(tmp_path / "site", "a: 1\n")
    noisy_python = _write_wrapper(
        tmp_path / "noisy_python",
        "",
        env_prefix=f'echo noise\nPYTHONPATH="{tmp_path / "site"}" ',
    )

    change = refresh_defaults(str(noisy_python), defaults_dir=defaults_dir)

    assert change.version == "1.1"
    assert change.added == ["a"]


def test_refresh_keeps_the_copy_when_the_version_is_unchanged(tmp_path):
    defaults_dir = tmp_path / "defaults"
    defaults_dir.mkdir()
    (defaults_dir / "ppo_trainer_defaults_1.1.yaml").write_text("a: 1\n")
    fake_generated = _make_fake_verl(tmp_path / "site", "a: 2\n")
    fake_python = _write_wrapper(
        tmp_path / "fake_python",
        "",
        env_prefix=f'PYTHONPATH="{tmp_path / "site"}" ',
    )

    change = refresh_defaults(str(fake_python), defaults_dir=defaults_dir)

    assert change.changed == ["a"]
    copied = defaults_dir / "ppo_trainer_defaults_1.1.yaml"
    assert copied.read_bytes() == fake_generated.read_bytes()
