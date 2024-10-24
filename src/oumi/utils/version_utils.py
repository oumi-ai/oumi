from importlib.metadata import version


def is_dev_build() -> bool:
    """Checks if the current version of Oumi is a development build."""
    return ".dev" in version("oumi")
