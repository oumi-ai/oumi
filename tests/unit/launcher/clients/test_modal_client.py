# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for :mod:`oumi.launcher.clients.modal_client`.

The actual ``modal`` SDK is heavy and requires real credentials, so these
tests stub it out with ``unittest.mock``.
"""

from __future__ import annotations

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

    # Image chain: Image.debian_slim().apt_install(...).uv_pip_install(...).
    image_obj = MagicMock(name="Image")
    image_obj.apt_install.return_value = image_obj
    image_obj.pip_install.return_value = image_obj
    image_obj.uv_pip_install.return_value = image_obj
    image_obj.run_commands.return_value = image_obj
    fake.Image.debian_slim.return_value = image_obj
    fake.Image.from_registry.return_value = image_obj

    # Secret.from_dict
    secret_obj = MagicMock(name="Secret")
    fake.Secret.from_dict.return_value = secret_obj

    # Volume.from_name returns a workspace-scoped volume handle.
    volume_obj = MagicMock(name="Volume")
    fake.Volume.from_name.return_value = volume_obj

    # App.lookup returns a persistent app reference (no context manager).
    app_obj = MagicMock(name="App")
    fake.App.lookup.return_value = app_obj

    # Sandbox.create returns a sandbox handle with a stable object_id.
    sandbox_obj = MagicMock(name="Sandbox")
    sandbox_obj.object_id = "sb-deadbeef"
    fake.Sandbox.create.return_value = sandbox_obj
    return fake


def test_launch_returns_pending_status_with_sandbox_id(fake_modal):
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        client = ModalClient()
        status = client.launch(_job(), cluster_name="my-cluster")
    # ``id`` is the sandbox object_id; ``cluster`` honors the caller's name.
    assert status.id == "sb-deadbeef"
    assert status.cluster == "my-cluster"
    assert client.sandboxes_for_cluster("my-cluster") == ["sb-deadbeef"]
    assert status.state == JobState.PENDING
    assert not status.done
    # Modal pricing isn't exposed via SDK; the OSS launcher leaves
    # ``cost_per_hour`` unset and lets callers fill it from their own
    # pricing source.
    assert status.cost_per_hour is None
    fake_modal.Sandbox.create.assert_called_once()
    fake_modal.App.lookup.assert_called_once()


def test_launch_mounts_hf_cache_volume(fake_modal):
    """A workspace-scoped Volume is mounted at the HF cache path."""
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        ModalClient().launch(_job(), cluster_name="my-cluster")
    fake_modal.Volume.from_name.assert_called_once_with(
        "oumi-hf-cache", create_if_missing=True
    )
    _, kwargs = fake_modal.Sandbox.create.call_args
    assert "/root/.cache/huggingface" in kwargs["volumes"]


def test_launch_default_image_uses_uv_pip_install(fake_modal):
    """Default image installs awscli via uv_pip_install (Modal-recommended)."""
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        ModalClient().launch(_job())
    image_obj = fake_modal.Image.debian_slim.return_value
    image_obj.uv_pip_install.assert_called_once_with("awscli")
    image_obj.apt_install.assert_called_once_with("zip", "curl", "git")


def test_launch_uses_sandbox_id_as_cluster_when_name_omitted(fake_modal):
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        client = ModalClient()
        status = client.launch(_job(), cluster_name=None)
    assert status.cluster == "sb-deadbeef"
    assert client.sandboxes_for_cluster("sb-deadbeef") == ["sb-deadbeef"]


def test_launch_tags_sandbox_with_cluster_name(fake_modal):
    """Sandbox is tagged so cleanup can find it across worker restarts."""
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        ModalClient().launch(_job(), cluster_name="my-cluster")
    fake_modal.Sandbox.create.return_value.set_tags.assert_called_once_with(
        {"oumi_cluster": "my-cluster"}
    )


def test_launch_swallows_set_tags_failure(fake_modal):
    """A tagging failure shouldn't abort the launch — fallback to in-process map."""
    fake_modal.Sandbox.create.return_value.set_tags.side_effect = RuntimeError("boom")
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        client = ModalClient()
        status = client.launch(_job(), cluster_name="my-cluster")
    assert status.id == "sb-deadbeef"
    assert client.sandboxes_for_cluster("my-cluster") == ["sb-deadbeef"]


def test_find_sandboxes_for_cluster_uses_modal_list(fake_modal):
    """Stateless lookup via Modal tag filter survives worker restarts."""
    sb1, sb2 = MagicMock(), MagicMock()
    sb1.object_id, sb2.object_id = "sb-a", "sb-b"
    fake_modal.Sandbox.list.return_value = iter([sb1, sb2])
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        client = ModalClient()
        ids = client.find_sandboxes_for_cluster("my-cluster")
    assert ids == ["sb-a", "sb-b"]
    fake_modal.Sandbox.list.assert_called_once_with(tags={"oumi_cluster": "my-cluster"})


def test_find_sandboxes_falls_back_to_in_process_tracker_on_list_failure(fake_modal):
    fake_modal.Sandbox.list.side_effect = RuntimeError("api down")
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        client = ModalClient()
        # Populate the in-process tracker by launching.
        client.launch(_job(), cluster_name="my-cluster")
        ids = client.find_sandboxes_for_cluster("my-cluster")
    assert ids == ["sb-deadbeef"]


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


def test_launch_concatenates_setup_and_run_with_sudo_stripped(fake_modal):
    """Setup script (with sudo) and run script are joined and sudo-stripped."""
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        ModalClient().launch(
            _job(
                setup="sudo apt-get update && sudo apt-get install -y zip",
                run="echo done",
            )
        )
    # Sandbox.create is called with /bin/bash -lc <script>.
    args, _ = fake_modal.Sandbox.create.call_args
    script = args[2]  # ("/bin/bash", "-lc", script)
    assert "sudo" not in script
    assert "apt-get update && apt-get install -y zip" in script
    assert "echo done" in script


def test_get_call_raises_cluster_not_found_on_lookup_failure(fake_modal):
    fake_modal.Sandbox.from_id.side_effect = RuntimeError("no such sandbox")
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        client = ModalClient()
        with pytest.raises(ClusterNotFoundError):
            client.get_call("sb-missing")


def test_cancel_swallows_underlying_errors(fake_modal):
    sandbox = MagicMock()
    sandbox.terminate.side_effect = RuntimeError("nope")
    fake_modal.Sandbox.from_id.return_value = sandbox
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        # Should not raise.
        ModalClient().cancel("sb-id")


def test_get_status_running_when_sandbox_poll_returns_none(fake_modal):
    sandbox = MagicMock()
    sandbox.poll.return_value = None
    fake_modal.Sandbox.from_id.return_value = sandbox
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        status = ModalClient().get_status("sb-id")
    assert status.state == JobState.RUNNING
    assert status.done is False


def test_get_status_succeeded_when_sandbox_exits_zero(fake_modal):
    sandbox = MagicMock()
    sandbox.poll.return_value = 0
    fake_modal.Sandbox.from_id.return_value = sandbox
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        status = ModalClient().get_status("sb-id")
    assert status.state == JobState.SUCCEEDED
    assert status.done is True


def test_get_status_failed_when_sandbox_exits_nonzero(fake_modal):
    sandbox = MagicMock()
    sandbox.poll.return_value = 1
    fake_modal.Sandbox.from_id.return_value = sandbox
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        status = ModalClient().get_status("sb-id")
    assert status.state == JobState.FAILED
    assert status.done is True


def test_get_logs_stream_returns_concatenated_stdout_and_stderr(fake_modal):
    sandbox = MagicMock()
    sandbox.stdout = iter(["hello\n", "world\n"])
    sandbox.stderr = iter([])
    fake_modal.Sandbox.from_id.return_value = sandbox
    with patch(
        "oumi.launcher.clients.modal_client._import_modal", return_value=fake_modal
    ):
        stream = ModalClient().get_logs_stream("sb-id")
    assert stream.readline() == "hello\n"
    assert stream.readline() == "world\n"
    assert stream.readline() == ""


def test_strip_sudo_removes_inline_and_chained_invocations():
    from oumi.launcher.clients.modal_client import _strip_sudo

    src = "sudo apt-get update && sudo apt-get install -y zip\nsudo dpkg -i bar.deb"
    assert _strip_sudo(src) == (
        "apt-get update && apt-get install -y zip\ndpkg -i bar.deb"
    )
