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

import importlib.metadata
from collections import namedtuple
from typing import Optional, Union

from packaging import version

PackagePrerequisites = namedtuple(
    "PackagePrerequisites",
    ["package_name", "min_package_version", "max_package_version"],
    defaults=["", None, None],
)

# Default error messages, if the package prerequisites are not met.
RUNTIME_ERROR_PREFIX = (
    "The current run cannot be launched because the platform prerequisites are not "
    "satisfied. In order to proceed, the following package(s) must be installed and "
    "have the correct version:\n"
)
RUNTIME_ERROR_SUFFIX = ""


def _version_bounds_str(min_package_version, max_package_version):
    """Returns a string representation of the version bounds."""
    if min_package_version is not None and max_package_version is not None:
        return f"{min_package_version} <= version <= {max_package_version}"
    elif min_package_version is not None:
        return f"version >= {min_package_version}"
    elif max_package_version is not None:
        return f"version <= {max_package_version}"
    else:
        return "any version"


def _package_error_message(
    package_name: str,
    actual_package_version: Union[version.Version, None],
    min_package_version: Optional[version.Version] = None,
    max_package_version: Optional[version.Version] = None,
) -> Union[str, None]:
    """Checks if a package is installed and if its version is compatible.

    This function checks if the package with name `package_name` is installed and if the
    installed version (`actual_package_version`) is compatible with the required
    version range. The installed version is considered compatible if it is
    both greater/equal to the `min_package_version` and less/equal to the
    `max_package_version`. If either the package is not installed or the version is
    incompatible, the function returns a user-friendly error message, otherwise it
    returns `None`.

    Args:
        package_name: Name of the package to check.
        actual_package_version: Actual version of the package in our Oumi environment.
        min_package_version: The minimum acceptable version of the package.
        max_package_version: The maximum acceptable version of the package.

    Returns:
        Error message (str) if the package is not installed or the version is
            incompatible, otherwise returns `None` (indicating that the check passed).
    """
    no_required_version = min_package_version is None and max_package_version is None

    if actual_package_version is None:
        if no_required_version:
            # Required package NOT present, no required version.
            return f"Package `{package_name}` is not installed."
        else:
            # Required package NOT present, specific version required.
            return (
                f"Package `{package_name}` is not installed. Please install: "
                f"{_version_bounds_str(min_package_version, max_package_version)}."
            )

    # Required package is present, but no specific version is required.
    if no_required_version:
        return None

    # Required package is present and a specific version is required.
    if (min_package_version and actual_package_version < min_package_version) or (
        max_package_version and actual_package_version > max_package_version
    ):
        return (
            f"Package `{package_name}` version is {actual_package_version}, which is "
            "not compatible. Please install: "
            f"{_version_bounds_str(min_package_version, max_package_version)}."
        )
    else:
        return None  # Compatible version (check passed).


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
            actual_package_version = version.parse(actual_package_version)
        except importlib.metadata.PackageNotFoundError:
            actual_package_version = None

        error_message = _package_error_message(
            package_name=package_name,
            actual_package_version=actual_package_version,
            min_package_version=version.parse(package_prerequisite.min_package_version)
            if package_prerequisite.min_package_version
            else None,
            max_package_version=version.parse(package_prerequisite.max_package_version)
            if package_prerequisite.max_package_version
            else None,
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
