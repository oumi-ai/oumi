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
