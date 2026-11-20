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

"""Errors raised by oumi's verl GRPO integration."""


class UnknownVerlKeyError(ValueError):
    """A setting that verl's defaults don't define."""

    def __init__(self, key: str, verl_version: str):
        """Initializes the error with the dotted path of the unknown key."""
        super().__init__(f"verl {verl_version} has no setting '{key}'")
        self.key = key
