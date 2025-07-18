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

"""A framework used for registering and accessing objects across Oumi."""

from oumi.core.registry.registry import (
    REGISTRY,
    Registry,
    RegistryType,
    register,
    register_cloud_builder,
    register_dataset,
    register_evaluation_function,
    register_judge,
)

__all__ = [
    "REGISTRY",
    "Registry",
    "RegistryType",
    "register",
    "register_cloud_builder",
    "register_dataset",
    "register_evaluation_function",
    "register_judge",
]
