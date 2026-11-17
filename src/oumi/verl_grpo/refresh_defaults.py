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

"""Refreshes the pinned copy of verl's default trainer config.

Run with `python -m oumi.verl_grpo.refresh_defaults --verl-python PATH`, where PATH
is the Python interpreter of the environment that has the new verl installed.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from oumi.verl_grpo.config import VERL_DEFAULTS_FILE

_DEFAULTS_GLOB = "ppo_trainer_defaults_*.yaml"
_GENERATED_CONFIG = Path("trainer") / "config" / "_generated_ppo_trainer.yaml"

# verl takes about a minute to import, so its install is located without importing it.
_VERL_NOT_FOUND_EXIT_CODE = 1
_LOCATE_VERL = """
import importlib.metadata, importlib.util, json, sys
try:
    version = importlib.metadata.version("verl")
except importlib.metadata.PackageNotFoundError:
    sys.exit(1)
spec = importlib.util.find_spec("verl")
if spec is None or not spec.submodule_search_locations:
    sys.exit(1)
print(json.dumps({"version": version, "root": spec.submodule_search_locations[0]}))
"""

_UPDATE_REMINDER = (
    "update VERL_VERSION and VERL_DEFAULTS_SHA256 in oumi/verl_grpo/config.py "
    "and EXPECTED_VERL_VERSION in oumi_verl_ext/version.py"
)


@dataclass(frozen=True)
class DefaultsChange:
    """What a refresh changed in the pinned defaults."""

    version: str
    sha256: str
    added: list[str]
    removed: list[str]
    changed: list[str]


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Maps each dotted key to its value; empty maps and lists count as values."""
    if not isinstance(node, dict) or not node:
        return {prefix: node}
    flat: dict[str, Any] = {}
    for key, value in node.items():
        flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def _load_flat(path: Path) -> dict[str, Any]:
    return _flatten(OmegaConf.to_container(OmegaConf.load(path), resolve=False))


def refresh_defaults(verl_python: str, *, defaults_dir: Path) -> DefaultsChange:
    """Replaces the defaults copy in `defaults_dir` with the installed verl's.

    Args:
        verl_python: Python interpreter of an environment with verl installed.
        defaults_dir: Directory that holds the `ppo_trainer_defaults_*.yaml` copy.

    Returns:
        The new version, its sha256, and the dotted keys that were added, removed
        or given a different default.

    Raises:
        RuntimeError: If verl or its generated config can't be found.
    """
    try:
        located = subprocess.run(
            [verl_python, "-c", _LOCATE_VERL], capture_output=True, text=True
        )
    except OSError as exc:
        raise RuntimeError(f"can't run '{verl_python}': {exc}") from exc
    if located.returncode == _VERL_NOT_FOUND_EXIT_CODE:
        raise RuntimeError(f"no verl is installed for '{verl_python}'")
    if located.returncode != 0:
        raise RuntimeError(
            f"'{verl_python}' failed to locate verl: {located.stderr.strip()}"
        )
    # Interpreters can print before the snippet runs, such as from sitecustomize.
    info = json.loads(located.stdout.strip().splitlines()[-1])
    source = Path(info["root"]) / _GENERATED_CONFIG
    if not source.is_file():
        raise RuntimeError(f"verl {info['version']} has no '{_GENERATED_CONFIG}'")

    old_copies = sorted(defaults_dir.glob(_DEFAULTS_GLOB))
    old_keys = _load_flat(old_copies[-1]) if old_copies else {}
    new_keys = _load_flat(source)

    destination = defaults_dir / f"ppo_trainer_defaults_{info['version']}.yaml"
    shutil.copyfile(source, destination)
    for old_copy in old_copies:
        if old_copy != destination:
            old_copy.unlink()

    shared = old_keys.keys() & new_keys.keys()
    return DefaultsChange(
        version=info["version"],
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        added=sorted(new_keys.keys() - old_keys.keys()),
        removed=sorted(old_keys.keys() - new_keys.keys()),
        changed=sorted(key for key in shared if old_keys[key] != new_keys[key]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Refresh the pinned copy of verl's default trainer config."
    )
    parser.add_argument(
        "--verl-python",
        required=True,
        help="Python interpreter of the environment that has verl installed.",
    )
    args = parser.parse_args(argv)

    try:
        change = refresh_defaults(
            args.verl_python, defaults_dir=VERL_DEFAULTS_FILE.parent
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"verl {change.version}")
    print(f"sha256 {change.sha256}")
    for label, keys in (
        ("added", change.added),
        ("removed", change.removed),
        ("changed", change.changed),
    ):
        print(f"{label} ({len(keys)}):")
        for key in keys:
            print(f"  {key}")
    print(_UPDATE_REMINDER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
