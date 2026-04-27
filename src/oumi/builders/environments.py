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

"""Build runtime environments from `EnvironmentParams`."""

from __future__ import annotations

from typing import cast

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.registry import REGISTRY, RegistryType
from oumi.environments.base_environment import BaseEnvironment


def build_environment(params: EnvironmentParams) -> BaseEnvironment:
    """Construct a concrete environment instance from its params.

    Dispatches based on `params.env_type` via the registry. Concrete classes
    are expected to expose a `from_params(EnvironmentParams)` classmethod;
    the builder calls into it.

    Raises:
        ValueError: If `params.env_type` is not registered.
    """
    cls = REGISTRY.get(params.env_type, RegistryType.ENVIRONMENT)
    if cls is None:
        known = sorted(REGISTRY.get_all(RegistryType.ENVIRONMENT))
        raise ValueError(f"Unknown env_type '{params.env_type}'. Known types: {known}")
    env_cls = cast(type[BaseEnvironment], cls)
    return env_cls.from_params(params)  # type: ignore[attr-defined]
