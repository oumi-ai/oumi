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

"""Skeleton-shape tests for ``DatabaseExecutableEnvironment``."""

from __future__ import annotations

import pytest

from oumi.core.configs.params.database_connection_params import (
    DatabaseConnectionConfig,
)
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.environments.database_executable_environment import (
    DatabaseExecutableEnvironment,
    DatabaseExecutableEnvironmentKwargs,
)
from oumi.environments.database_executable_tool import DatabaseExecutableTool


def test_executor_context_kwarg_is_db():
    """DatabaseExecutableEnvironment passes the connection as ``db=`` to executors."""
    assert DatabaseExecutableEnvironment._executor_context_kwarg == "db"


def test_tool_params_cls_is_database_executable_tool():
    """The env binds to ``DatabaseExecutableTool``."""
    assert DatabaseExecutableEnvironment.tool_params_cls is DatabaseExecutableTool


def test_from_params_is_skeleton():
    """``from_params`` raises NotImplementedError until the implementation phase."""
    params = EnvironmentParams(id="test", env_type="database")
    with pytest.raises(NotImplementedError, match="skeleton"):
        DatabaseExecutableEnvironment.from_params(params)


def test_kwargs_requires_connection():
    """Kwargs validation rejects a missing ``connection``."""
    kwargs = DatabaseExecutableEnvironmentKwargs()
    with pytest.raises(ValueError, match="connection"):
        kwargs.finalize_and_validate()


def test_kwargs_rejects_non_positive_statement_timeout():
    """Env-level ``statement_timeout_ms`` must be > 0 when set."""
    kwargs = DatabaseExecutableEnvironmentKwargs(
        connection=DatabaseConnectionConfig(driver="sqlite", database="x.db"),
        statement_timeout_ms=0,
    )
    with pytest.raises(ValueError, match="statement_timeout_ms"):
        kwargs.finalize_and_validate()


def test_kwargs_coerces_connection_dict():
    """A raw connection dict is coerced into ``DatabaseConnectionConfig``."""
    kwargs = DatabaseExecutableEnvironmentKwargs(
        connection={"driver": "sqlite", "database": "x.db"},  # type: ignore[arg-type]
    )
    assert isinstance(kwargs.connection, DatabaseConnectionConfig)
    assert kwargs.connection.driver == "sqlite"
