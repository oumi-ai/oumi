from importlib import metadata
from importlib.metadata import version


def is_dev_build() -> bool:
    """Checks if the current version of Oumi is a development build."""
    return ".dev" in version("oumi")


def get_python_package_versions() -> dict[str, str]:
    """Returns a dictionary of the installed package names and their versions."""
    packages = {}
    for distribution in metadata.distributions():
        package_name = distribution.metadata["Name"]
        package_version = distribution.version
        packages[package_name] = package_version
    return packages
