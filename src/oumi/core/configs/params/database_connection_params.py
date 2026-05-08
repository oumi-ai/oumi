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
"""Connection parameters for DatabaseExecutableEnvironment."""

from __future__ import annotations

import os
from dataclasses import dataclass

import sqlalchemy
import sqlalchemy.engine

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class DatabaseConnectionConfig(BaseParams):
    """SQLAlchemy connection config for DatabaseExecutableEnvironment.

    Two modes (mutually exclusive):

    - **Structured fields:** populate ``driver`` and ``database`` (and host/port/
      username as needed). The password is resolved at ``resolve_url()`` time
      from ``password_env_var`` so it never appears in YAML.
    - **DSN env var:** populate ``dsn_env_var`` with the name of an env var
      holding a complete SQLAlchemy URL.
    """

    # Mode 1: structured fields
    driver: str = ""  # e.g. "postgresql+psycopg", "mysql+pymysql", "sqlite"
    host: str = ""
    port: int | None = None
    database: str = ""
    username: str = ""
    password_env_var: str = ""  # name of env var holding the password

    # Mode 2: full DSN from env (escape hatch)
    dsn_env_var: str = ""  # mutually exclusive with structured fields

    # Pool / timeouts
    pool_size: int = 5
    pool_max_overflow: int = 10
    pool_pre_ping: bool = True
    connect_timeout_s: float = 10.0

    def __finalize_and_validate__(self) -> None:
        """Validate mode XOR (structured vs DSN) and pool/timeout numeric bounds."""
        structured = bool(self.driver and self.database)
        dsn = bool(self.dsn_env_var)
        if structured and dsn:
            raise ValueError(
                "DatabaseConnectionConfig: set EITHER (driver, database, ...) "
                "OR dsn_env_var, not both."
            )
        if not structured and not dsn:
            raise ValueError(
                "DatabaseConnectionConfig: must set either (driver + database) "
                "OR dsn_env_var. Got neither."
            )
        if self.pool_size < 1:
            raise ValueError(
                f"DatabaseConnectionConfig.pool_size must be >= 1, "
                f"got {self.pool_size}."
            )
        if self.pool_max_overflow < 0:
            raise ValueError(
                f"DatabaseConnectionConfig.pool_max_overflow must be >= 0, "
                f"got {self.pool_max_overflow}."
            )
        if self.connect_timeout_s <= 0:
            raise ValueError(
                f"DatabaseConnectionConfig.connect_timeout_s must be > 0, "
                f"got {self.connect_timeout_s}."
            )

    def resolve_url(self) -> sqlalchemy.URL:
        """Build the SQLAlchemy URL.

        Reads ``password_env_var`` / ``dsn_env_var`` from the environment at
        call time (not at YAML-parse time), so configs are safe to log/dump.
        """
        if self.dsn_env_var:
            dsn = os.environ.get(self.dsn_env_var)
            if not dsn:
                raise ValueError(
                    f"DatabaseConnectionConfig: env var '{self.dsn_env_var}' "
                    f"is not set; cannot build connection URL."
                )
            return sqlalchemy.engine.make_url(dsn)
        password = (
            os.environ.get(self.password_env_var) if self.password_env_var else None
        )
        return sqlalchemy.URL.create(
            drivername=self.driver,
            host=self.host or None,
            port=self.port,
            database=self.database,
            username=self.username or None,
            password=password,
        )
