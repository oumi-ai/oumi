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

"""Tests for the sql_execution_match reward."""

from __future__ import annotations

from oumi.datasets.grpo.rewards.sql_execution_match import sql_execution_match

_SCHEMA = "CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);"
_SEED = "INSERT INTO patients VALUES (1,'Bob',50),(2,'Alice',40),(3,'Carol',65);"


def _extra():
    return {"schema_sql": _SCHEMA, "seed_sql": _SEED}


def test_exact_match_scores_one():
    gold = "SELECT name FROM patients WHERE age > 45 ORDER BY name"
    candidate = "SELECT name FROM patients WHERE age >= 50 ORDER BY name"
    score = sql_execution_match(
        data_source="ehr",
        solution_str=candidate,
        ground_truth=gold,
        extra_info=_extra(),
    )
    assert score == 1.0  # Bob, Carol in both


def test_mismatch_scores_zero():
    gold = "SELECT name FROM patients WHERE age > 45 ORDER BY name"
    candidate = "SELECT name FROM patients ORDER BY name"
    score = sql_execution_match(
        data_source="ehr",
        solution_str=candidate,
        ground_truth=gold,
        extra_info=_extra(),
    )
    assert score == 0.0


def test_invalid_sql_scores_zero():
    score = sql_execution_match(
        data_source="ehr",
        solution_str="SELECT FROM nonsense(",
        ground_truth="SELECT 1",
        extra_info=_extra(),
    )
    assert score == 0.0
