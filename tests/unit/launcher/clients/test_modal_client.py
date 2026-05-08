# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for :mod:`oumi.launcher.clients.modal_client`.

The actual ``modal`` SDK is heavy and requires real credentials, so these
tests stub it out with ``unittest.mock``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import JobConfig, JobResources
from oumi.core.launcher import ClusterNotFoundError, JobState
from oumi.launcher.clients.modal_client import ModalClient


def _job(cloud: str = "modal", **overrides) -> JobConfig:
    resources = JobResources(
        cloud=cloud,
        accelerators=overrides.pop("accelerators", "H100:8"),
        image_id=overrides.pop("image_id", None),
    )
    return JobConfig(
        name=overrides.pop("name", "myjob"),
        user="user",
        working_dir="./",
        num_nodes=1,
        resources=resources,
        envs=overrides.pop("envs", {"HF_TOKEN": "redacted"}),
        file_mounts={},
        storage_mounts={},
        setup=overrides.pop("setup", "pip install -e ."),
        run=overrides.pop("run", "./run.sh"),
    )


@pytest.fixture
def fake_modal():
    """Constructs a fake ``modal`` SDK with the surface area we touch."""
    fake = MagicMock(name="modal")

    # Image chain: Image.debian_slim().pip_install("uv").run_commands([...])
    image_obj = MagicMock(name="Image")
    image_obj.run_commands.return_value = image_obj
    image_obj.pip_install.return_value = image_obj
    fake.Image.debian_slim.return_value = image_obj
    fake.Image.from_registry.return_value = image_obj

    # Secret.from_dict
    secret_obj = MagicMock(name="Secret")
    fake.Secret.from_dict.return_value = secret_obj

    # App.run() context manager.
    app_obj = MagicMock(name="App")
    app_run_ctx = MagicMock()
    app_run_ctx.__enter__ = MagicMock(return_value=app_run_ctx)
    app_run_ctx.__exit__ = MagicMock(return_value=False)
    app_obj.run.return_value = app_run_ctx

    # @app.function(...) returns a decorator that returns a function-like
    # object exposing .spawn(...) → FunctionCall(object_id=...).
    fn_obj = MagicMock(name="Function")
    function_call = SimpleNamespace(object_id="fc-deadbeef")
    fn_obj.spawn.return_value = function_call
    decorator = MagicMock(return_value=fn_obj)
    app_obj.function.return_value = decorator
    fake.App.return_value = app_obj

    # Exception namespace used by _function_call_state.
    fake.exception = SimpleNamespace(
        OutputExpiredError=type("OutputExpiredError", (Exception,), {}),
        FunctionTimeoutError=type("FunctionTimeoutError", (Exception,), {}),
    )
    return fake


def test_launch_returns_pending_status_with_call_id(fake_modal):
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        client = ModalClient()
        status = client.launch(_job(), cluster_name="my-cluster")
    assert status.id == "fc-deadbeef"
    assert status.cluster == "fc-deadbeef"
    assert status.state == JobState.PENDING
    assert not status.done
    # H100 list price * 8 ≈ 31.6.
    assert status.cost_per_hour == pytest.approx(31.6)
    fake_modal.App.assert_called_once()


def test_launch_uses_image_from_registry_when_image_id_set(fake_modal):
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        ModalClient().launch(_job(image_id="docker:my/repo:tag"))
    fake_modal.Image.from_registry.assert_called()


def test_launch_omits_secrets_when_envs_empty(fake_modal):
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        ModalClient().launch(_job(envs={}))
    fake_modal.Secret.from_dict.assert_not_called()


def test_get_call_raises_cluster_not_found_on_lookup_failure(fake_modal):
    fake_modal.FunctionCall.from_id.side_effect = RuntimeError("no such call")
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        client = ModalClient()
        with pytest.raises(ClusterNotFoundError):
            client.get_call("fc-missing")


def test_cancel_swallows_underlying_errors(fake_modal):
    call = MagicMock()
    call.cancel.side_effect = RuntimeError("nope")
    fake_modal.FunctionCall.from_id.return_value = call
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        # Should not raise.
        ModalClient().cancel("fc-id")


def test_get_status_running_when_call_pending(fake_modal):
    call = MagicMock()
    call.get.side_effect = TimeoutError("still running")
    fake_modal.FunctionCall.from_id.return_value = call
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        status = ModalClient().get_status("fc-id")
    assert status.state == JobState.RUNNING
    assert status.done is False


def test_get_status_succeeded_when_call_returns(fake_modal):
    call = MagicMock()
    call.get.return_value = 0
    fake_modal.FunctionCall.from_id.return_value = call
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        status = ModalClient().get_status("fc-id")
    assert status.state == JobState.SUCCEEDED
    assert status.done is True


def test_get_status_failed_when_call_raises_user_exception(fake_modal):
    call = MagicMock()
    call.get.side_effect = ValueError("user code blew up")
    fake_modal.FunctionCall.from_id.return_value = call
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        status = ModalClient().get_status("fc-id")
    assert status.state == JobState.FAILED
    assert status.done is True


def test_estimate_cost_per_hour_known_and_unknown():
    assert ModalClient.estimate_cost_per_hour(None) is None
    assert ModalClient.estimate_cost_per_hour("XYZ:4") is None
    assert ModalClient.estimate_cost_per_hour("A100-80GB") == pytest.approx(2.5)
    assert ModalClient.estimate_cost_per_hour("H100:8") == pytest.approx(31.6)


def test_get_logs_stream_returns_no_op_when_logs_unsupported(fake_modal):
    call = MagicMock(spec=[])  # no `logs` attribute
    fake_modal.FunctionCall.from_id.return_value = call
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        stream = ModalClient().get_logs_stream("fc-id")
    # readline on an empty iterator returns "".
    assert stream.readline() == ""
