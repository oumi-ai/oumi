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

"""Provider-agnostic exceptions for the Oumi launcher.

Callers should depend on these exceptions rather than provider-specific ones
(e.g. ``sky.exceptions.ClusterDoesNotExist``) so that swapping providers does
not require touching call sites.
"""


class LauncherError(Exception):
    """Base class for all launcher-level errors."""


class ClusterNotFoundError(LauncherError):
    """Raised when the requested cluster does not exist on the cloud."""
