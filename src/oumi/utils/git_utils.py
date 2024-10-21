import subprocess
from pathlib import Path
from typing import Optional


def get_git_revision_hash() -> Optional[str]:
    """Get the current git revision hash.

    Returns:
        Optional[str]: The current git revision hash, or None if it cannot be retrieved.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None


def get_git_tag() -> Optional[str]:
    """Get the current git tag.

    Returns:
        Optional[str]: The current git tag, or None if it cannot be retrieved.
    """
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0", "--exact-match"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None
