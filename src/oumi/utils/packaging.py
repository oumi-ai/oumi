import importlib.metadata
from collections import namedtuple
from enum import Enum
from typing import Optional, Union

from packaging import version

PackagePrerequisites = namedtuple(
    "PackagePrerequisites",
    ["package_name", "package_version", "version_comparator"],
    defaults=["", None, None],
)

# Default error messages, if the package prerequisites are not met.
RUNTIME_ERROR_PREFIX = (
    "The current run cannot be launched because the platform prerequisites are not "
    "satisfied. In order to proceed, the following package(s) must be installed and "
    "have the correct version:\n"
)
RUNTIME_ERROR_SUFFIX = ""
MISSING_PACKAGE_NO_REQUIRED_VERSION_STR = "Package `{package_name}` is not installed."
MISSING_PACKAGE_WITH_REQUIRED_VERSION_STR = (
    "Package `{package_name}` is not installed. Please install version "
    "{comparator}{required_version}."
)
INCOMPATIBLE_VERSION_STR = (
    "Package `{package_name}` version is {actual_version} but "
    "{comparator}{required_version} is required. Please install version "
    "{comparator}{required_version}."
)


class Comparator(Enum):
    GREATER = ">"
    GREATER_EQUAL = ">="
    EQUAL = "=="
    LESS_EQUAL = "<="
    LESS = "<"
    ANY = None

    def version_compatible(self, actual_version, required_version):
        """Returns True if actual version meets the requirements, False otherwise."""
        if (required_version is None) or (self == Comparator.ANY):
            return True
        elif actual_version is None:
            return False

        actual_version = version.parse(actual_version)
        required_version = version.parse(required_version)
        if self == Comparator.GREATER:
            return actual_version > required_version
        elif self == Comparator.GREATER_EQUAL:
            return actual_version >= required_version
        elif self == Comparator.EQUAL:
            return actual_version == required_version
        elif self == Comparator.LESS_EQUAL:
            return actual_version <= required_version
        elif self == Comparator.LESS:
            return actual_version < required_version
        else:
            raise ValueError(f"Unknown comparator: {self.value}")


def _package_error_message(
    package_name: str,
    actual_version: Union[str, None],
    comparator: Optional[str] = None,
    required_version: Optional[str] = None,
) -> Union[str, None]:
    """Checks if a package is installed and if its version is compatible.

    This function checks if the package with name `package_name` is installed and if the
    installed version (`actual_version`) is compatible with the required version
    (`required_version`). The compatibility is determined by the `comparator` argument,
    which can be [`>`, `>=`, `==`, `<=`, `<`, `None`], as follows:

        actual_version (comparator) required_version

    If either the package is not installed or the version is incompatible, the function
    returns a user-friendly error message, otherwise it returns `None`.

    Args:
        package_name: Name of the package to check.
        actual_version: Actual version of the package in our Oumi environment.
        comparator: The comparator to use for the version check.
        required_version: The required version of the package.

    Returns:
        Error message (str) if the package is not installed or the version is
            incompatible, otherwise returns `None` (indicating that the check passed).
    """
    if (actual_version is None) and (required_version is None):
        # Required package NOT present, no required version.
        return MISSING_PACKAGE_NO_REQUIRED_VERSION_STR.format(package_name=package_name)
    elif (actual_version is None) and (required_version is not None):
        # Required package NOT present, specific version required.
        return MISSING_PACKAGE_WITH_REQUIRED_VERSION_STR.format(
            package_name=package_name,
            comparator=comparator,
            required_version=required_version,
        )
    elif (actual_version is not None) and (required_version is None):
        # Required package present, no specific version is needed (check passed).
        return None

    # Required package present, specific version required.
    comparator_enum = Comparator(comparator)
    if comparator_enum.version_compatible(actual_version, required_version):
        return None  # Compatible version (check passed).
    else:
        return INCOMPATIBLE_VERSION_STR.format(
            package_name=package_name,
            actual_version=actual_version,
            comparator=comparator,
            required_version=required_version,
        )


def _package_prerequisites_error_messages(
    package_prerequisites: list[PackagePrerequisites],
) -> list[str]:
    """Checks if a list of package prerequisites are satisfied.

    This function checks if a list of package prerequisites are satisfied and returns an
    error message for each package that is not installed or has an incompatible version.
    If the function returns an empty list, all prerequisites are satisfied.
    """
    error_messages = []

    for package_prerequisite in package_prerequisites:
        package_name = package_prerequisite.package_name
        try:
            actual_package_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            actual_package_version = None

        error_message = _package_error_message(
            package_name=package_name,
            actual_version=actual_package_version,
            comparator=package_prerequisite.version_comparator,
            required_version=package_prerequisite.package_version,
        )

        if error_message is not None:
            error_messages.append(error_message)

    return error_messages


def check_package_prerequisites(
    package_prerequisites: list[PackagePrerequisites],
    runtime_error_prefix: str = RUNTIME_ERROR_PREFIX,
    runtime_error_suffix: str = RUNTIME_ERROR_SUFFIX,
) -> None:
    """Checks if the package prerequisites are satisfied and raises an error if not."""
    if error_messages := _package_prerequisites_error_messages(package_prerequisites):
        raise RuntimeError(
            runtime_error_prefix + "\n".join(error_messages) + runtime_error_suffix
        )
    else:
        return
