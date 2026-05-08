# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Unit tests for DatabaseConnectionConfig."""

from __future__ import annotations

import pytest

from oumi.core.configs.params.database_connection_params import (
    DatabaseConnectionConfig,
)


def test_structured_fields_resolve_url(monkeypatch):
    monkeypatch.setenv("PWD_VAR", "secret123")
    cfg = DatabaseConnectionConfig(
        driver="postgresql+psycopg",
        host="db.local",
        port=5432,
        database="ehr",
        username="oumi",
        password_env_var="PWD_VAR",
    )
    cfg.finalize_and_validate()
    url = cfg.resolve_url()
    assert url.drivername == "postgresql+psycopg"
    assert url.host == "db.local"
    assert url.port == 5432
    assert url.database == "ehr"
    assert url.username == "oumi"
    assert url.password == "secret123"


def test_dsn_env_var_resolve_url(monkeypatch):
    monkeypatch.setenv("DSN_VAR", "sqlite:///:memory:")
    cfg = DatabaseConnectionConfig(dsn_env_var="DSN_VAR")
    cfg.finalize_and_validate()
    url = cfg.resolve_url()
    assert url.drivername == "sqlite"
    assert url.database == ":memory:"


def test_structured_and_dsn_mutually_exclusive():
    cfg = DatabaseConnectionConfig(
        driver="postgresql+psycopg",
        database="ehr",
        dsn_env_var="DSN_VAR",
    )
    with pytest.raises(ValueError, match="not both"):
        cfg.finalize_and_validate()


def test_neither_structured_nor_dsn_raises():
    cfg = DatabaseConnectionConfig()
    with pytest.raises(ValueError, match="Got neither"):
        cfg.finalize_and_validate()


def test_dsn_env_var_unset_raises(monkeypatch):
    monkeypatch.delenv("DSN_VAR", raising=False)
    cfg = DatabaseConnectionConfig(dsn_env_var="DSN_VAR")
    cfg.finalize_and_validate()
    with pytest.raises(ValueError, match="not set"):
        cfg.resolve_url()


def test_password_env_var_unset_yields_no_password(monkeypatch):
    monkeypatch.delenv("MISSING_PWD", raising=False)
    cfg = DatabaseConnectionConfig(
        driver="postgresql+psycopg",
        database="ehr",
        password_env_var="MISSING_PWD",
    )
    cfg.finalize_and_validate()
    url = cfg.resolve_url()
    assert url.password is None


def test_pool_size_must_be_positive():
    cfg = DatabaseConnectionConfig(driver="sqlite", database=":memory:", pool_size=0)
    with pytest.raises(ValueError, match="pool_size"):
        cfg.finalize_and_validate()


def test_connect_timeout_must_be_positive():
    cfg = DatabaseConnectionConfig(
        driver="sqlite", database=":memory:", connect_timeout_s=0
    )
    with pytest.raises(ValueError, match="connect_timeout_s"):
        cfg.finalize_and_validate()


def test_pool_max_overflow_must_be_nonnegative():
    cfg = DatabaseConnectionConfig(
        driver="sqlite", database=":memory:", pool_max_overflow=-1
    )
    with pytest.raises(ValueError, match="pool_max_overflow"):
        cfg.finalize_and_validate()
