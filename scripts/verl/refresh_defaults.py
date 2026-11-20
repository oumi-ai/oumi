"""Refreshes oumi's pinned copy of verl's default trainer config.

Run it when bumping verl, with the Python of an env that has the new verl:

    python scripts/verl/refresh_defaults.py --verl-python /path/to/verl-env/bin/python

It replaces src/oumi/verl_grpo/ppo_trainer_defaults_<version>.yaml, then prints the new
sha256 and the config keys verl added, removed or changed. Afterwards, update
VERL_VERSION and VERL_DEFAULTS_SHA256 in src/oumi/verl_grpo/config.py, and the verl
version in src/oumi_verl_ext/install.py.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

DEFAULTS_DIR = Path(__file__).resolve().parents[2] / "src" / "oumi" / "verl_grpo"
# verl takes about a minute to import, so its install is found without importing it.
LOCATE_VERL = (
    "import importlib.metadata, importlib.util, json; print(json.dumps({"
    "'version': importlib.metadata.version('verl'), "
    "'root': importlib.util.find_spec('verl').submodule_search_locations[0]}))"
)


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Maps each dotted key to its value; empty maps and lists count as values."""
    if not isinstance(node, dict) or not node:
        return {prefix: node}
    flat = {}
    for key, value in node.items():
        flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def _load_flat(path: Path) -> dict[str, Any]:
    return _flatten(OmegaConf.to_container(OmegaConf.load(path), resolve=False))


def main() -> None:
    """Replaces the defaults copy with the installed verl's and prints the diff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-python", required=True)
    args = parser.parse_args()

    output = subprocess.run(
        [args.verl_python, "-c", LOCATE_VERL],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    # Interpreters can print before the snippet runs, such as from sitecustomize.
    verl = json.loads(output.strip().splitlines()[-1])
    source = Path(verl["root"]) / "trainer" / "config" / "_generated_ppo_trainer.yaml"

    old_copies = sorted(DEFAULTS_DIR.glob("ppo_trainer_defaults_*.yaml"))
    old_keys = _load_flat(old_copies[-1]) if old_copies else {}
    new_keys = _load_flat(source)
    destination = DEFAULTS_DIR / f"ppo_trainer_defaults_{verl['version']}.yaml"
    shutil.copyfile(source, destination)
    for old_copy in old_copies:
        if old_copy != destination:
            old_copy.unlink()

    print(f"verl {verl['version']}")
    print(f"sha256 {hashlib.sha256(destination.read_bytes()).hexdigest()}")
    shared = old_keys.keys() & new_keys.keys()
    for label, keys in (
        ("added", new_keys.keys() - old_keys.keys()),
        ("removed", old_keys.keys() - new_keys.keys()),
        ("changed", {key for key in shared if old_keys[key] != new_keys[key]}),
    ):
        print(f"{label} ({len(keys)}):")
        for key in sorted(keys):
            print(f"  {key}")


if __name__ == "__main__":
    main()
