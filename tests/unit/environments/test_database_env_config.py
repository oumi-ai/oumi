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

"""End-to-end: build the EHR database env from its YAML config."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from oumi.builders.environments import build_environment
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.environments.database_executable_environment import (
    DatabaseExecutableEnvironment,
)

_CONFIG = Path("configs/examples/database_env/ehr_database_env.yaml")


def test_build_environment_from_yaml():
    raw = OmegaConf.to_container(OmegaConf.load(_CONFIG), resolve=True)
    params = EnvironmentParams(**raw)
    env = build_environment(params)
    assert isinstance(env, DatabaseExecutableEnvironment)
    try:
        [result] = env.step([("lookup_patient", {"pat_id": 1})])
        assert result.output == {"name": "Bob", "meds": "aspirin"}
    finally:
        env.close()
