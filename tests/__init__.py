import functools
from pathlib import Path


@functools.cache
def get_configs_dir() -> Path:
    """Retrieve the absolute path to the 'configs' directory.

    This function returns the absolute path to the 'configs' directory,
    which is located two levels up from the current file's directory.

    Returns:
        Path: The absolute path to the 'configs' directory."""
    return (Path(__file__).parent.parent / "configs").resolve()


@functools.cache
def get_testdata_dir() -> Path:
    """Returns the absolute path to the 'testdata' directory.

    This function provides a cached reference to the 'testdata' directory 
    located in the same directory as the current file, ensuring efficient 
    retrieval on subsequent calls.

    Returns:
        Path: The absolute path to the 'testdata' directory."""
    return (Path(__file__).parent / "testdata").resolve()


@functools.cache
def get_notebooks_dir() -> Path:
    """Get the absolute path to the 'notebooks' directory.

    Returns:
        Path: An absolute Path object pointing to the 'notebooks' directory,
        which is located two levels up from the current file's directory."""
    return (Path(__file__).parent.parent / "notebooks").resolve()
