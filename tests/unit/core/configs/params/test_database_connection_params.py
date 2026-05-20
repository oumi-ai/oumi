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

"""Skeleton-shape tests for ``DatabaseConnectionConfig``."""

from __future__ import annotations

import pytest

from oumi.core.configs.params.database_connection_params import (
    DatabaseConnectionConfig,
)


def test_rejects_neither_mode():
    """Empty config rejects: structured XOR DSN must be set."""
    config = DatabaseConnectionConfig()
    with pytest.raises(ValueError, match="either"):
        config.finalize_and_validate()


def test_rejects_both_modes():
    """Setting both structured fields and DSN env var is invalid."""
    config = DatabaseConnectionConfig(
        driver="sqlite", database="x.db", dsn_env_var="DB_URL"
    )
    with pytest.raises(ValueError, match="not both"):
        config.finalize_and_validate()


def test_accepts_structured_mode():
    """Structured fields (driver + database) is a valid mode."""
    config = DatabaseConnectionConfig(driver="sqlite", database="x.db")
    config.finalize_and_validate()


def test_accepts_dsn_mode():
    """DSN env var alone is a valid mode."""
    config = DatabaseConnectionConfig(dsn_env_var="DATABASE_URL")
    config.finalize_and_validate()


def test_rejects_zero_pool_size():
    """pool_size must be >= 1."""
    config = DatabaseConnectionConfig(driver="sqlite", database="x.db", pool_size=0)
    with pytest.raises(ValueError, match="pool_size"):
        config.finalize_and_validate()


def test_rejects_non_positive_connect_timeout():
    """connect_timeout_s must be > 0."""
    config = DatabaseConnectionConfig(
        driver="sqlite", database="x.db", connect_timeout_s=0
    )
    with pytest.raises(ValueError, match="connect_timeout_s"):
        config.finalize_and_validate()
