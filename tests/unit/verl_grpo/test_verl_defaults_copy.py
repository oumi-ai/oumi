import hashlib
import os
import subprocess
import sys
from pathlib import Path

import oumi
from oumi.verl_grpo.config import VERL_DEFAULTS_FILE, VERL_DEFAULTS_SHA256, VERL_VERSION


def test_defaults_copy_is_unedited():
    digest = hashlib.sha256(VERL_DEFAULTS_FILE.read_bytes()).hexdigest()
    assert digest == VERL_DEFAULTS_SHA256


def test_only_the_pinned_copy_ships():
    copies = sorted(
        p.name for p in VERL_DEFAULTS_FILE.parent.glob("ppo_trainer_defaults_*.yaml")
    )
    assert copies == [f"ppo_trainer_defaults_{VERL_VERSION}.yaml"]


def test_config_module_imports_without_verl_or_trainers(tmp_path):
    (tmp_path / "verl").mkdir()
    (tmp_path / "verl" / "__init__.py").write_text(
        "raise ImportError('verl was imported')"
    )
    src = Path(oumi.__file__).resolve().parents[1]
    code = (
        "import sys; from oumi.verl_grpo.config import merge_verl_config; "
        "merge_verl_config({'trainer': {'total_epochs': 1}}); "
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "{'verl','ray','vllm','torch','transformers','datasets'} "
        "or m.startswith(('oumi.core.trainers','oumi.builders','oumi.train'))]; "
        "assert not bad, bad"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{src}"},
    )
