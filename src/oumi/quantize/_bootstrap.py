# Copyright 2025 - Oumi
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

"""Auto-bootstrap for the llama.cpp Python tools (gguf-py + convert script).

The ``llama-quantize`` and ``llama-imatrix`` binaries are user-installed
(``brew install llama.cpp`` on Mac, distro packages or a source build on
Linux). The Python-side ``convert_hf_to_gguf.py`` script ships with those
binaries but depends on llama.cpp's in-tree ``gguf-py`` package, which
PyPI's ``gguf`` does **not** track 1:1 (the same version number, e.g.
0.18.0, ships different content over time).

Rather than vendor ~10k LoC of third-party code, we bootstrap a tiny
sparse blobless clone of llama.cpp into a user cache directory the first
time GGUF quantization is requested. Subsequent runs reuse the clone.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from oumi.utils.logging import logger

# Pinned tag of the llama.cpp checkout we bootstrap. The convert script
# and matching gguf-py at this tag must work together.
PINNED_LLAMACPP_TAG = "b9020"

LLAMACPP_REPO_URL = "https://github.com/ggml-org/llama.cpp"

# Override the default cache location.
ENV_LLAMACPP_HOME = "OUMI_LLAMACPP_HOME"

# Skip the interactive prompt (set to "1" / "true" / "yes" in CI).
ENV_AUTO_INSTALL = "OUMI_LLAMACPP_AUTO_INSTALL"

_SPARSE_PATHS = ("gguf-py", "convert_hf_to_gguf.py")


def cache_root() -> Path:
    """Where the bootstrapped llama.cpp clone lives."""
    override = os.environ.get(ENV_LLAMACPP_HOME)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "oumi" / "llamacpp"


def is_valid_clone(path: Path) -> bool:
    """Whether ``path`` looks like a usable llama.cpp checkout."""
    return (
        (path / "gguf-py" / "gguf" / "__init__.py").is_file()
        and (path / "convert_hf_to_gguf.py").is_file()
    )


def gguf_py_path(root: Path) -> Path:
    """The ``gguf-py`` directory to put on ``PYTHONPATH``."""
    return root / "gguf-py"


def convert_script_path(root: Path) -> Path:
    """The ``convert_hf_to_gguf.py`` to run."""
    return root / "convert_hf_to_gguf.py"


def ensure_llamacpp_python_tools() -> Path:
    """Return a path to a usable llama.cpp checkout, bootstrapping if needed.

    First-run path: prompts the user to confirm the install (unless
    ``OUMI_LLAMACPP_AUTO_INSTALL`` is set), then performs a sparse blobless
    clone (~2MB) into the cache directory. Subsequent runs reuse it.

    Raises:
        RuntimeError: If git is unavailable, the user declines the install,
            stdin is not a tty (and auto-install isn't set), or the clone fails.
    """
    root = cache_root()
    if is_valid_clone(root):
        return root

    if shutil.which("git") is None:
        raise RuntimeError(
            "GGUF quantization needs `git` on PATH to bootstrap llama.cpp tools."
        )

    if not _confirm_install(root):
        raise RuntimeError(
            "GGUF quantization aborted: llama.cpp tools are required but the "
            f"user declined the auto-install. Either set {ENV_AUTO_INSTALL}=1 "
            f"to skip the prompt, or set {ENV_LLAMACPP_HOME} to point at an "
            "existing llama.cpp checkout containing `gguf-py/` and "
            "`convert_hf_to_gguf.py`."
        )

    _do_install(root)
    if not is_valid_clone(root):
        raise RuntimeError(
            f"Bootstrap completed but {root} is not a valid llama.cpp checkout "
            "(missing gguf-py or convert_hf_to_gguf.py)."
        )
    return root


def _confirm_install(root: Path) -> bool:
    if os.environ.get(ENV_AUTO_INSTALL, "").lower() in ("1", "true", "yes"):
        logger.info(
            f"{ENV_AUTO_INSTALL} set; proceeding with llama.cpp bootstrap."
        )
        return True
    if not sys.stdin.isatty():
        logger.error(
            "GGUF quantization needs llama.cpp tools and stdin is not a tty. "
            f"Set {ENV_AUTO_INSTALL}=1 in CI or pre-clone llama.cpp and set "
            f"{ENV_LLAMACPP_HOME}."
        )
        return False
    sys.stderr.write(
        "\noumi: GGUF quantization needs llama.cpp's convert_hf_to_gguf.py and\n"
        "      its matching gguf-py Python package.\n"
        f"  Will sparse-clone {LLAMACPP_REPO_URL} at tag {PINNED_LLAMACPP_TAG}\n"
        f"  into {root}\n"
        f"  (sparse blobless checkout, ~2MB on disk, ~5s).\n"
        f"  Skip prompt next time: export {ENV_AUTO_INSTALL}=1\n"
        f"  Use a different checkout: export {ENV_LLAMACPP_HOME}=/path/to/llama.cpp\n"
        "Proceed? [y/N] "
    )
    sys.stderr.flush()
    answer = sys.stdin.readline().strip().lower()
    return answer in ("y", "yes")


def _do_install(root: Path) -> None:
    """Sparse blobless clone of llama.cpp at the pinned tag.

    Fetches only the paths we need (``gguf-py/`` and the convert script).
    """
    if root.exists():
        # Either partial / wrong SHA — start fresh. Safe because we own the
        # cache directory exclusively.
        logger.warning(f"Removing partial llama.cpp checkout at {root}")
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Cloning {LLAMACPP_REPO_URL} (tag {PINNED_LLAMACPP_TAG}) into {root}..."
    )
    _git(
        [
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--depth",
            "1",
            "--branch",
            PINNED_LLAMACPP_TAG,
            LLAMACPP_REPO_URL,
            str(root),
        ],
        cwd=None,
    )
    _git(["sparse-checkout", "init", "--no-cone"], cwd=root)
    sparse_file = root / ".git" / "info" / "sparse-checkout"
    sparse_file.write_text("\n".join(_SPARSE_PATHS) + "\n")
    _git(["checkout", PINNED_LLAMACPP_TAG], cwd=root)
    logger.info(f"llama.cpp tools installed at {root}.")


def _git(args: list[str], *, cwd: Path | None) -> None:
    """Run a git command, surfacing stderr on failure."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"`git {' '.join(args)}` failed (exit {result.returncode}):\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
