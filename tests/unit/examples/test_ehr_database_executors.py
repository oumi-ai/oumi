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

"""Tests for the EHR database synthesis example's tool executors."""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from oumi.builders.environments import build_environment
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.environments.database_executable_environment import (
    DatabaseExecutableEnvironment,
)
from oumi.environments.utils import resolve_executor

_EXAMPLE_DIR = (
    Path(__file__).parents[3]
    / "configs"
    / "examples"
    / "synthesis"
    / "ehr_database_agent"
)
_CONFIG = _EXAMPLE_DIR / "ehr_database_synth.yaml"


@pytest.fixture(scope="module")
def ehr_db():
    """Load the example's executors the way OUMI_EXTRA_DEPS_FILE does."""
    spec = importlib.util.spec_from_file_location(
        "ehr_database_example_executors", _EXAMPLE_DIR / "executors.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env(ehr_db) -> Iterator[DatabaseExecutableEnvironment]:
    """Build the example's environment from its config; ehr_db registers its tools."""
    raw = cast(
        "dict[str, Any]", OmegaConf.to_container(OmegaConf.load(_CONFIG), resolve=True)
    )
    params = EnvironmentParams(**raw["environment_config"]["environments"][0])
    environment = build_environment(params)
    assert isinstance(environment, DatabaseExecutableEnvironment)
    try:
        yield environment
    finally:
        environment.close()


def test_config_executor_names_resolve_to_the_example_functions(ehr_db):
    """Every ``executor:`` in the config must be a registered ``ehr_db.*`` name."""
    names = re.findall(r"^\s*executor:\s*(\S+)", _CONFIG.read_text(), re.MULTILINE)
    assert len(names) == 3
    for name in names:
        namespace, _, attr = name.partition(".")
        assert namespace == "ehr_db"
        assert resolve_executor(name, "t") is getattr(ehr_db, attr)


def test_list_patients_returns_every_seeded_row(env):
    [result] = env.step([("list_patients", {})])
    assert isinstance(result.output, dict)
    patients = result.output["patients"]
    assert len(patients) == 12
    assert patients[0] == {"id": 1, "name": "Bob Martin"}


def test_lookup_patient_returns_name_and_meds(env):
    [result] = env.step([("lookup_patient", {"pat_id": 2})])
    assert result.output == {"name": "Alice Chen", "meds": "ibuprofen"}


def test_lookup_patient_unknown_id_returns_error(env):
    [result] = env.step([("lookup_patient", {"pat_id": 999})])
    assert result.output == {"error": "not found"}


def test_update_meds_is_visible_to_a_later_lookup(env):
    [update] = env.step([("update_meds", {"pat_id": 1, "medication": "warfarin"})])
    assert update.output == {"updated_rows": 1}
    [lookup] = env.step([("lookup_patient", {"pat_id": 1})])
    assert lookup.output == {"name": "Bob Martin", "meds": "warfarin"}
