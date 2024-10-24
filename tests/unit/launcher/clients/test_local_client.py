import tempfile
import time
from datetime import datetime, timezone
from io import TextIOWrapper
from subprocess import PIPE
from unittest.mock import Mock, call, patch

import pytest

from oumi.core.configs import JobConfig, JobResources, StorageMount
from oumi.core.launcher import JobStatus
from oumi.launcher.clients.local_client import LocalClient


#
# Fixtures
#
@pytest.fixture
def mock_time():
    with patch("oumi.launcher.clients.local_client.time") as time_mock:
        yield time_mock


@pytest.fixture
def mock_thread():
    with patch("oumi.launcher.clients.local_client.Thread") as thread_mock:
        yield thread_mock


@pytest.fixture
def mock_popen():
    with patch("oumi.launcher.clients.local_client.Popen") as popen_mock:
        yield popen_mock


@pytest.fixture
def mock_os():
    with patch("oumi.launcher.clients.local_client.os") as os_mock:
        yield os_mock


@pytest.fixture
def mock_datetime():
    with patch("oumi.launcher.clients.local_client.datetime") as datetime_mock:
        yield datetime_mock


class OpenEquivalent(TextIOWrapper):
    def __init__(self, file):
        self.file = file

    def __eq__(self, other):
        return (
            self.file.mode == other.mode
            and self.file.name == other.name
            and self.file.encoding == other.encoding
        )


def _get_default_job() -> JobConfig:
    resources = JobResources(
        cloud="local",
        region="us-central1",
        zone=None,
        accelerators="A100-80",
        cpus="4",
        memory="64",
        instance_type=None,
        use_spot=True,
        disk_size=512,
        disk_tier="low",
    )
    return JobConfig(
        name="myjob",
        user="user",
        working_dir="./",
        num_nodes=2,
        resources=resources,
        envs={"var1": "val1"},
        file_mounts={},
        storage_mounts={
            "~/home/remote/path/gcs/": StorageMount(
                source="gs://mybucket/", store="gcs"
            )
        },
        setup="pip install -r requirements.txt",
        run="./hello_world.sh",
    )


#
# Tests
#
def test_local_client_init(mock_thread):
    client = LocalClient()
    mock_thread.assert_called_once_with(target=client._worker_loop, daemon=True)


def test_local_client_submit_job(mock_thread):
    client = LocalClient()
    job = _get_default_job()
    job_status = client.submit_job(job)
    expected_status = JobStatus(
        id="0", name=str(job.name), cluster="", status="QUEUED", metadata="", done=False
    )
    assert job_status == expected_status
    assert client.list_jobs() == [expected_status]


def test_local_client_submit_job_execution(
    mock_time, mock_popen, mock_os, mock_datetime
):
    client = LocalClient()
    mock_os.environ.copy.return_value = {"preset": "value"}
    mock_datetime.fromtimestamp.return_value = datetime.fromtimestamp(10)
    mock_time.time.return_value = 10
    mock_process = Mock()
    mock_popen.return_value = mock_process
    mock_process.wait.return_value = 0
    job = _get_default_job()
    job_status = client.submit_job(job)
    time.sleep(1)
    expected_cmds = """cd ./
pip install -r requirements.txt
./hello_world.sh"""
    mock_popen.assert_called_once_with(
        expected_cmds,
        shell=True,
        env={"preset": "value", "var1": "val1"},
        stdout=PIPE,
        stderr=PIPE,
    )
    expected_status = JobStatus(
        id="0",
        name=str(job.name),
        cluster="",
        status="COMPLETED",
        metadata=f"Job finished at {datetime.fromtimestamp(10).isoformat()} .",
        done=True,
    )
    assert job_status == expected_status
    assert client.list_jobs() == [expected_status]


def test_local_client_submit_job_execution_with_logging(
    mock_time, mock_popen, mock_os, mock_datetime
):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        client = LocalClient()
        mock_os.environ.copy.return_value = {
            "preset": "value",
            "OUMI_LOGGING_DIR": output_temp_dir,
        }
        mock_datetime.fromtimestamp.return_value = datetime.fromtimestamp(10)
        mock_datetime.now.return_value = datetime.fromtimestamp(10.1234, timezone.utc)
        mock_time.time.return_value = 10
        mock_process = Mock()
        mock_popen.return_value = mock_process
        mock_process.wait.return_value = 0
        job = _get_default_job()
        job_status = client.submit_job(job)
        time.sleep(1)
        expected_cmds = """cd ./
pip install -r requirements.txt
./hello_world.sh"""
        stdout_handler = open(
            f"{output_temp_dir}/1970_01_01_00_00_10_123_0.stdout", "w"
        )
        std_err_handler = open(
            f"{output_temp_dir}/1970_01_01_00_00_10_123_0.stderr", "w"
        )
        mock_popen.assert_called_once_with(
            expected_cmds,
            shell=True,
            env={
                "preset": "value",
                "OUMI_LOGGING_DIR": output_temp_dir,
                "var1": "val1",
            },
            stdout=OpenEquivalent(stdout_handler),
            stderr=OpenEquivalent(std_err_handler),
        )
        expected_status = JobStatus(
            id="0",
            name=str(job.name),
            cluster="",
            status="COMPLETED",
            metadata=f"Job finished at {datetime.fromtimestamp(10).isoformat()} ."
            f" Logs available at: {output_temp_dir}/1970_01_01_00_00_10_123_0.stdout",
            done=True,
        )
        assert job_status == expected_status
        assert client.list_jobs() == [expected_status]


def test_local_client_submit_job_execution_multiple(
    mock_time, mock_popen, mock_os, mock_datetime
):
    client = LocalClient()
    mock_datetime.fromtimestamp.return_value = datetime.fromtimestamp(10)
    mock_os.environ.copy.side_effect = [{"preset": "value"}, {"preset": "value"}]
    mock_time.time.return_value = 10
    mock_process = Mock()
    mock_process.wait.return_value = 0
    mock_process2 = Mock()
    mock_process2.wait.return_value = 1
    mock_process2.stderr = "hidden message" + "a" * 1024
    mock_popen.side_effect = [mock_process, mock_process2]
    job = _get_default_job()
    second_job = _get_default_job()
    second_job.envs = {}
    second_job.setup = None
    second_job.run = "echo 'hello'"
    second_job.working_dir = "~/second"
    job_status = client.submit_job(job)
    second_job_status = client.submit_job(second_job)
    time.sleep(1)
    expected_cmds = """cd ./
pip install -r requirements.txt
./hello_world.sh"""
    second_expected_cmds = """cd ~/second

echo 'hello'"""
    mock_popen.assert_has_calls(
        [
            call(
                expected_cmds,
                shell=True,
                env={"preset": "value", "var1": "val1"},
                stdout=PIPE,
                stderr=PIPE,
            ),
            call(
                second_expected_cmds,
                shell=True,
                env={"preset": "value"},
                stdout=PIPE,
                stderr=PIPE,
            ),
        ]
    )
    expected_status = JobStatus(
        id="0",
        name=str(job.name),
        cluster="",
        status="COMPLETED",
        metadata=f"Job finished at {datetime.fromtimestamp(10).isoformat()} .",
        done=True,
    )
    second_expected_status = JobStatus(
        id="1",
        name=str(second_job.name),
        cluster="",
        status="FAILED",
        metadata="a" * 1024,
        done=True,
    )
    assert job_status == expected_status
    assert second_job_status == second_expected_status
    assert client._running_process is None
    assert client.list_jobs() == [expected_status, second_expected_status]


def test_local_client_list_jobs(mock_thread):
    client = LocalClient()
    job = _get_default_job()
    job_status = client.submit_job(job)
    assert client.list_jobs() == [job_status]


def test_local_client_list_jobs_multiple(mock_thread):
    client = LocalClient()
    job = _get_default_job()
    job2 = _get_default_job()
    job2.name = "secondjob"
    job3 = _get_default_job()
    job3.name = "thirdjob"
    job_status = client.submit_job(job)
    job_status2 = client.submit_job(job2)
    job_status3 = client.submit_job(job3)
    assert client.list_jobs() == [job_status, job_status2, job_status3]
    assert job_status.id == "0"
    assert job_status2.id == "1"
    assert job_status3.id == "2"
    assert job_status.name == "myjob"
    assert job_status2.name == "secondjob"
    assert job_status3.name == "thirdjob"


def test_local_client_get_job(mock_thread):
    client = LocalClient()
    job = _get_default_job()
    job2 = _get_default_job()
    job2.name = "secondjob"
    job3 = _get_default_job()
    job3.name = "thirdjob"
    job_status = client.submit_job(job)
    job_status2 = client.submit_job(job2)
    job_status3 = client.submit_job(job3)
    assert client.get_job("0") == job_status
    assert client.get_job("1") == job_status2
    assert client.get_job("2") == job_status3
    client.cancel("1")
    updated_status = client.get_job("1")
    assert updated_status is not None
    assert updated_status.status == "CANCELED"
    assert client.get_job("3") is None


def test_local_client_cancel(mock_thread):
    client = LocalClient()
    job = _get_default_job()
    job2 = _get_default_job()
    job2.name = "secondjob"
    job_status = client.submit_job(job)
    job_status2 = client.submit_job(job2)
    assert job_status.status == "QUEUED"
    assert job_status2.status == "QUEUED"
    job_status2 = client.cancel("1")
    assert job_status.status == "QUEUED"
    assert job_status2 is not None
    assert job_status2.status == "CANCELED"
    job_status = client.cancel("0")
    assert job_status is not None
    assert job_status.status == "CANCELED"
    assert job_status2 is not None
    assert job_status2.status == "CANCELED"
    assert client.cancel("2") is None


def test_local_client_cancel_completed_jobs(mock_time, mock_popen, mock_os):
    client = LocalClient()
    mock_os.environ.copy.side_effect = [{"preset": "value"}, {"preset": "value"}]
    mock_time.time.return_value = 10
    mock_process = Mock()
    mock_process.wait.return_value = 0
    mock_process2 = Mock()
    mock_process2.wait.return_value = 1
    mock_process2.stderr = "hidden message" + "a" * 1024
    mock_popen.side_effect = [mock_process, mock_process2]
    job = _get_default_job()
    second_job = _get_default_job()
    second_job.envs = {}
    second_job.setup = None
    second_job.run = "echo 'hello'"
    second_job.working_dir = "~/second"
    job_status = client.submit_job(job)
    second_job_status = client.submit_job(second_job)
    time.sleep(1)
    job_status = client.get_job("0")
    second_job_status = client.get_job("1")
    assert job_status is not None
    assert second_job_status is not None
    assert job_status.status == "COMPLETED"
    assert second_job_status.status == "FAILED"

    job_status_after_cancel = client.cancel("0")
    job_status2_after_cancel = client.cancel("1")
    assert job_status_after_cancel is not None
    assert job_status2_after_cancel is not None
    assert job_status_after_cancel.status == "COMPLETED"
    assert job_status2_after_cancel.status == "FAILED"
