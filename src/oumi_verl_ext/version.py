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

"""Checks that the installed verl is the version this extension targets."""

import importlib.metadata
from typing import Final

EXPECTED_VERL_VERSION: Final = "0.9.1"


class VerlVersionError(RuntimeError):
    """Raised when the installed verl is missing or is not the expected version."""


def check_verl_version() -> None:
    """Raises if verl is not installed at `EXPECTED_VERL_VERSION`.

    Reads the package metadata, so verl itself is never imported.
    """
    fix = f"Install verl=={EXPECTED_VERL_VERSION} in the Python env that runs verl."
    try:
        installed = importlib.metadata.version("verl")
    except importlib.metadata.PackageNotFoundError:
        raise VerlVersionError(
            f"verl is not installed; expected {EXPECTED_VERL_VERSION}. {fix}"
        ) from None
    if installed != EXPECTED_VERL_VERSION:
        raise VerlVersionError(
            f"Found verl {installed}, but oumi_verl_ext needs "
            f"{EXPECTED_VERL_VERSION}. {fix}"
        )
