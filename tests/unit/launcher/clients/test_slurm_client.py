import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest

from oumi.core.launcher import JobStatus
from oumi.launcher.clients.slurm_client import SlurmClient

_CTRL_PATH: str = "-S ~/.ssh/control-%h-%p-%r"
_SACCT_CMD = (
    "sacct --user=user --format='JobId%-30,JobName%30,User%30,State%30,Reason%30' "
    "-X --starttime 2025-01-01"
)


#
# Fixtures
#
@pytest.fixture
def mock_subprocess_no_init():
    with patch("oumi.launcher.clients.slurm_client.subprocess") as sp:
        yield sp


@pytest.fixture
def mock_subprocess():
    with patch("oumi.launcher.clients.slurm_client.subprocess") as sp:
        sp.TimeoutExpired = subprocess.TimeoutExpired
        mock_child = Mock()
        sp.run.return_value = mock_child
        mock_child.returncode = 0
        yield sp


def _get_test_data(file_name: str) -> str:
    data_path = Path(__file__).parent / "data" / file_name
    with open(data_path) as f:
        return f.read()


def _run_commands_template(commands: list[str]) -> str:
    user = "user"
    ctrl_path = "-S ~/.ssh/control-%h-%p-%r"
    ssh_cmd = f"ssh {ctrl_path} {user}@host  << 'EOF'"
    eof_suffix = "EOF"
    return "\n".join([ssh_cmd, *commands, eof_suffix])


#
# Tests
#
def test_slurm_client_init(mock_subprocess):
    _ = SlurmClient("user", "host", "cluster_name")
    mock_subprocess.run.assert_called_once_with(
        "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
        shell=True,
        capture_output=True,
        timeout=10,
    )


def test_slurm_client_submit_job(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = b"2032"
    mock_run.stderr = b"err"
    mock_run.returncode = 0
    client = SlurmClient("user", "host", "cluster_name")
    result = client.submit_job("./job.sh", "work_dir", 2, None)
    mock_subprocess.run.assert_has_calls(
        [
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                _run_commands_template(
                    [
                        "cd work_dir",
                        "sbatch --nodes=2  --parsable ./job.sh",
                    ]
                ),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
        ]
    )
    assert result == "2032"


def test_slurm_client_submit_job_name(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = b"2032"
    mock_run.stderr = b"err"
    mock_run.returncode = 0
    client = SlurmClient("user", "host", "cluster_name")
    result = client.submit_job("./job.sh", "work_dir", 2, "somename")
    mock_subprocess.run.assert_has_calls(
        [
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                _run_commands_template(
                    [
                        "cd work_dir",
                        "sbatch --nodes=2 --job-name=somename --parsable ./job.sh",
                    ]
                ),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
        ]
    )
    assert result == "2032"


def test_slurm_client_submit_job_error(mock_subprocess):
    mock_success_run = Mock()
    mock_success_run.stdout = b"out"
    mock_success_run.stderr = b"err"
    mock_success_run.returncode = 0
    mock_run = Mock()
    mock_subprocess.run.side_effect = [
        mock_success_run,
        mock_success_run,
        mock_run,
    ]
    mock_run.stdout = b"3141592653polaris-pbs-01"
    mock_run.stderr = b"foo"
    mock_run.returncode = 1
    client = SlurmClient("user", "host", "cluster_name")
    with pytest.raises(RuntimeError, match="Failed to submit job. stderr: foo"):
        _ = client.submit_job("./job.sh", "work_dir", 2, None)
    mock_subprocess.run.assert_has_calls(
        [
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                _run_commands_template(
                    [
                        "cd work_dir",
                        "sbatch --nodes=2  --parsable ./job.sh",
                    ]
                ),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
        ]
    )


def test_slurm_client_submit_job_retry_auth(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = b"3141592653polaris-pbs-01"
    mock_run.stderr = b"err"
    mock_run.returncode = 0
    client = SlurmClient("user", "host", "cluster_name")
    result = client.submit_job("./job.sh", "work_dir", 2, None)
    mock_subprocess.run.assert_has_calls(
        [
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                _run_commands_template(
                    [
                        "cd work_dir",
                        "sbatch --nodes=2  --parsable ./job.sh",
                    ]
                ),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
        ]
    )
    assert result == "3141592653polaris-pbs-01"


def test_slurm_client_list_jobs_success(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = _get_test_data("sacct.txt").encode("utf-8")
    mock_run.stderr = b"foo"
    mock_run.returncode = 0

    client = SlurmClient("user", "host", "cluster_name")
    job_list = client.list_jobs()
    mock_subprocess.run.assert_called_with(
        _run_commands_template([_SACCT_CMD]),
        shell=True,
        capture_output=True,
        timeout=180,
    )
    job_ids = [job.id for job in job_list]
    expected_ids = [
        "6",
        "6.batch",
        "7",
        "7.batch",
    ]
    assert job_ids == expected_ids


def test_slurm_client_list_jobs_first_login_success(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = _get_test_data("sacct_full.txt").encode("utf-8")
    mock_run.stderr = b"foo"
    mock_run.returncode = 0

    client = SlurmClient("user", "host", "cluster_name")
    job_list = client.list_jobs()
    mock_subprocess.run.assert_called_with(
        _run_commands_template([_SACCT_CMD]),
        shell=True,
        capture_output=True,
        timeout=180,
    )
    job_ids = [job.id for job in job_list]
    expected_ids = [
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
    assert job_ids == expected_ids


def test_slurm_client_list_jobs_fails_missing_header(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    data = _get_test_data("sacct_full.txt").encode("utf-8")
    data = b"\n".join(data.split(b"\n")[:-7])
    mock_run.stdout = data
    mock_run.stderr = b"foo"
    mock_run.returncode = 0
    client = SlurmClient("user", "host", "cluster_name")

    with pytest.raises(
        RuntimeError, match="Failed to parse job list. Unexpected format:"
    ):
        client = SlurmClient("user", "host", "cluster_name")
        _ = client.list_jobs()
        mock_subprocess.run.assert_called_with(
            _run_commands_template([_SACCT_CMD]),
            shell=True,
            capture_output=True,
            timeout=180,
        )


def test_slurm_client_list_jobs_handles_empty_string(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = b""
    mock_run.stderr = b"foo"
    mock_run.returncode = 0

    client = SlurmClient("user", "host", "cluster_name")
    job_list = client.list_jobs()
    mock_subprocess.run.assert_called_with(
        _run_commands_template([_SACCT_CMD]),
        shell=True,
        capture_output=True,
        timeout=180,
    )
    job_ids = [job.id for job in job_list]
    expected_ids = []
    assert job_ids == expected_ids


def test_slurm_client_list_jobs_failure(mock_subprocess):
    mock_success_run = Mock()
    mock_success_run.stdout = b"out"
    mock_success_run.stderr = b"err"
    mock_success_run.returncode = 0
    mock_run = Mock()
    mock_subprocess.run.side_effect = [
        mock_success_run,
        mock_success_run,
        mock_success_run,
        mock_run,
    ]
    mock_run.stdout = b""
    mock_run.stderr = b"foo"
    mock_run.returncode = 1

    client = SlurmClient("user", "host", "cluster_name")
    with pytest.raises(RuntimeError, match="Failed to list jobs. stderr: foo"):
        client = SlurmClient("user", "host", "cluster_name")
        _ = client.list_jobs()
    mock_subprocess.run.assert_called_with(
        _run_commands_template([_SACCT_CMD]),
        shell=True,
        capture_output=True,
        timeout=180,
    )


def test_slurm_client_get_job_success(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = _get_test_data("sacct.txt").encode("utf-8")
    mock_run.stderr = b"foo"
    mock_run.returncode = 0

    client = SlurmClient("user", "host", "cluster_name")
    job_status = client.get_job("6")
    mock_subprocess.run.assert_called_with(
        _run_commands_template(
            [
                _SACCT_CMD,
            ]
        ),
        shell=True,
        capture_output=True,
        timeout=180,
    )
    expected_status = JobStatus(
        id="6",
        name="test",
        status="COMPLETED",
        cluster="cluster_name",
        metadata=(
            "JobID                                                 JobName                           User                          State                         Reason\n"  # noqa: E501
            "------------------------------ ------------------------------ ------------------------------ ------------------------------ ------------------------------\n"  # noqa: E501
            "                             6                           test                         taenin                      COMPLETED                           None"  # noqa: E501
        ),
        done=True,
    )
    assert job_status == expected_status


def test_slurm_client_get_job_success_one_job(mock_subprocess):
    data = _get_test_data("sacct.txt").encode("utf-8")
    data = b"\n".join(data.split(b"\n")[:3])
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = data
    mock_run.stderr = b"foo"
    mock_run.returncode = 0

    client = SlurmClient("user", "host", "cluster_name")
    job_status = client.get_job("6")
    mock_subprocess.run.assert_called_with(
        _run_commands_template(
            [
                _SACCT_CMD,
            ]
        ),
        shell=True,
        capture_output=True,
        timeout=180,
    )
    expected_status = JobStatus(
        id="6",
        name="test",
        status="COMPLETED",
        cluster="cluster_name",
        metadata=(
            "JobID                                                 JobName                           User                          State                         Reason\n"  # noqa: E501
            "------------------------------ ------------------------------ ------------------------------ ------------------------------ ------------------------------\n"  # noqa: E501
            "                             6                           test                         taenin                      COMPLETED                           None"  # noqa: E501
        ),
        done=True,
    )
    assert job_status == expected_status


def test_slurm_client_get_job_not_found(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = _get_test_data("sacct.txt").encode("utf-8")
    mock_run.stderr = b"foo"
    mock_run.returncode = 0
    client = SlurmClient("user", "host", "cluster_name")
    job_status = client.get_job("2017652")
    mock_subprocess.run.assert_called_with(
        _run_commands_template([_SACCT_CMD]),
        shell=True,
        capture_output=True,
        timeout=180,
    )
    assert job_status is None


def test_slurm_client_get_job_not_found_empty(mock_subprocess):
    data = _get_test_data("sacct.txt").encode("utf-8")
    data = b"\n".join(data.split(b"\n")[:2])
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = data
    mock_run.stderr = b"foo"
    mock_run.returncode = 0

    client = SlurmClient("user", "host", "cluster_name")
    job_status = client.get_job("6")
    mock_subprocess.run.assert_called_with(
        _run_commands_template([_SACCT_CMD]),
        shell=True,
        capture_output=True,
        timeout=180,
    )
    assert job_status is None


def test_slurm_client_get_job_failure(mock_subprocess):
    mock_success_run = Mock()
    mock_success_run.stdout = b"out"
    mock_success_run.stderr = b"err"
    mock_success_run.returncode = 0
    mock_run = Mock()
    mock_subprocess.run.side_effect = [
        mock_success_run,
        mock_success_run,
        mock_run,
    ]
    mock_run.stdout = _get_test_data("sacct.txt").encode("utf-8")
    mock_run.stderr = b"foo"
    mock_run.returncode = 1
    client = SlurmClient("user", "host", "cluster_name")
    with pytest.raises(RuntimeError, match="Failed to list jobs. stderr: foo"):
        _ = client.get_job("2017652")
    mock_subprocess.run.assert_called_with(
        _run_commands_template([_SACCT_CMD]),
        shell=True,
        capture_output=True,
        timeout=180,
    )


def test_slurm_client_cancel_success(mock_subprocess):
    mock_run2 = Mock()
    mock_run2.stdout = _get_test_data("sacct.txt").encode("utf-8")
    mock_run2.stderr = b"foo"
    mock_run2.returncode = 0
    mock_subprocess.run.return_value = mock_run2

    client = SlurmClient("user", "host", "cluster_name")
    job_status = client.cancel("7.batch")
    mock_subprocess.run.assert_has_calls(
        [
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                _run_commands_template(["scancel 7.batch"]),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                _run_commands_template([_SACCT_CMD]),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
        ]
    )
    expected_status = JobStatus(
        id="7.batch",
        name="batch",
        status="RUNNING",
        cluster="cluster_name",
        metadata=(
            "JobID                                                 JobName                           User                          State                         Reason\n"  # noqa: E501
            "------------------------------ ------------------------------ ------------------------------ ------------------------------ ------------------------------\n"  # noqa: E501
            "                       7.batch                          batch                                                       RUNNING"  # noqa: E501
        ),
        done=False,
    )
    assert job_status == expected_status


def test_slurm_client_cancel_scancel_failure(mock_subprocess):
    mock_success_run = Mock()
    mock_success_run.stdout = b"out"
    mock_success_run.stderr = b"err"
    mock_success_run.returncode = 0
    mock_run = Mock()
    mock_subprocess.run.side_effect = [
        mock_success_run,
        mock_success_run,
        mock_run,
    ]
    mock_run.stdout = b""
    mock_run.stderr = b"foo"
    mock_run.returncode = 1
    with pytest.raises(RuntimeError, match="Failed to cancel job. stderr: foo"):
        client = SlurmClient("user", "host", "cluster_name")
        _ = client.cancel("2017652")
    mock_subprocess.run.assert_has_calls(
        [
            call(
                _run_commands_template(["scancel 2017652"]),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
        ]
    )


def test_slurm_client_cancel_sacct_failure(mock_subprocess):
    mock_run1 = Mock()
    mock_run1.stdout = b""
    mock_run1.stderr = b""
    mock_run1.returncode = 0
    mock_run2 = Mock()
    mock_run2.stdout = _get_test_data("sacct.txt").encode("utf-8")
    mock_run2.stderr = b"foo"
    mock_run2.returncode = 1
    mock_subprocess.run.side_effect = [
        mock_run1,
        mock_run1,
        mock_run1,
        mock_run1,
        mock_run2,
    ]
    with pytest.raises(RuntimeError, match="Failed to list jobs. stderr: foo"):
        client = SlurmClient("user", "host", "cluster_name")
        _ = client.cancel("2017652")
    mock_subprocess.run.assert_has_calls(
        [
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                _run_commands_template(["scancel 2017652"]),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                _run_commands_template([_SACCT_CMD]),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
        ]
    )


def test_slurm_client_cancel_job_not_found_success(mock_subprocess):
    mock_run2 = Mock()
    mock_run2.stdout = _get_test_data("sacct.txt").encode("utf-8")
    mock_run2.stderr = b"foo"
    mock_run2.returncode = 0
    mock_subprocess.run.return_value = mock_run2
    client = SlurmClient("user", "host", "cluster_name")
    job_status = client.cancel("2017652")
    mock_subprocess.run.assert_has_calls(
        [
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                _run_commands_template(["scancel 2017652"]),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
            call(
                "ssh -S ~/.ssh/control-%h-%p-%r -O check user@host",
                shell=True,
                capture_output=True,
                timeout=10,
            ),
            call(
                _run_commands_template([_SACCT_CMD]),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
        ]
    )
    assert job_status is None


def test_slurm_client_run_commands_success(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = b"out"
    mock_run.stderr = b"err"
    mock_run.returncode = 0
    commands = [
        "first command",
        "cd second/command",
        "cd third/command",
        "fourth command",
        "cd fifth/command",
        "final command",
    ]
    client = SlurmClient("user", "host", "cluster_name")
    result = client.run_commands(commands)
    mock_subprocess.run.assert_called_with(
        _run_commands_template(commands),
        shell=True,
        capture_output=True,
        timeout=180,
    )
    assert result.exit_code == 0
    assert result.stdout == "out"
    assert result.stderr == "err"


def test_slurm_client_run_commands_success_empty(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = b"out"
    mock_run.stderr = b"err"
    mock_run.returncode = 0
    client = SlurmClient("user", "host", "cluster_name")
    result = client.run_commands([])
    mock_subprocess.run.assert_has_calls(
        [
            call(
                _run_commands_template([]),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
        ]
    )
    assert result.exit_code == 0
    assert result.stdout == "out"
    assert result.stderr == "err"


def test_slurm_client_run_commands_fails(mock_subprocess):
    mock_success_run = Mock()
    mock_success_run.stdout = b"out"
    mock_success_run.stderr = b"err"
    mock_success_run.returncode = 0
    mock_run = Mock()
    mock_subprocess.run.side_effect = [
        mock_success_run,
        mock_success_run,
        mock_run,
    ]
    mock_run.stdout = b"out"
    mock_run.stderr = b"err"
    mock_run.returncode = 1
    client = SlurmClient("user", "host", "cluster_name")
    result = client.run_commands([])
    mock_subprocess.run.assert_called_with(
        _run_commands_template([]),
        shell=True,
        capture_output=True,
        timeout=180,
    )
    assert result.exit_code == 1
    assert result.stdout == "out"
    assert result.stderr == "err"


def test_slurm_client_put_recursive_success(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = b"out"
    mock_run.stderr = b"err"
    mock_run.returncode = 0
    client = SlurmClient("user", "host", "cluster_name")
    client.put_recursive(
        "source",
        "destination",
    )
    mock_subprocess.run.assert_has_calls(
        [
            call(
                'rsync -e "ssh -S ~/.ssh/control-%h-%p-%r" -avz --delete '
                "source user@host:destination",
                shell=True,
                capture_output=True,
                timeout=300,
            ),
        ]
    )


def test_slurm_client_put_recursive_success_gitignore(mock_subprocess):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        with open(Path(output_temp_dir) / ".gitignore", "w") as f:
            f.write("*.txt")
        mock_run = Mock()
        mock_subprocess.run.return_value = mock_run
        mock_run.stdout = b"out"
        mock_run.stderr = b"err"
        mock_run.returncode = 0
        client = SlurmClient("user", "host", "cluster_name")
        client.put_recursive(
            output_temp_dir,
            "destination",
        )
        mock_subprocess.run.assert_has_calls(
            [
                call(
                    'rsync -e "ssh -S ~/.ssh/control-%h-%p-%r" -avz --delete '
                    f"--exclude-from {output_temp_dir}/.gitignore "
                    f"{output_temp_dir} user@host:destination",
                    shell=True,
                    capture_output=True,
                    timeout=300,
                ),
            ]
        )


def test_slurm_client_put_recursive_success_tests(mock_subprocess):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        tests = Path(output_temp_dir) / "tests"
        tests.mkdir()
        with open(tests / "file.txt", "w") as f:
            f.write("*.txt")
        mock_run = Mock()
        mock_subprocess.run.return_value = mock_run
        mock_run.stdout = b"out"
        mock_run.stderr = b"err"
        mock_run.returncode = 0
        client = SlurmClient("user", "host", "cluster_name")
        client.put_recursive(
            output_temp_dir,
            "destination",
        )
        mock_subprocess.run.assert_has_calls(
            [
                call(
                    'rsync -e "ssh -S ~/.ssh/control-%h-%p-%r" -avz --delete '
                    f"--exclude {output_temp_dir}/tests "
                    f"{output_temp_dir} user@host:destination",
                    shell=True,
                    capture_output=True,
                    timeout=300,
                ),
            ]
        )


def test_slurm_client_put_recursive_success_tests_gitignore(mock_subprocess):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        tests = Path(output_temp_dir) / "tests"
        tests.mkdir()
        with open(tests / "file.txt", "w") as f:
            f.write("*.txt")
        with open(Path(output_temp_dir) / ".gitignore", "w") as f:
            f.write("*.txt")
        mock_run = Mock()
        mock_subprocess.run.return_value = mock_run
        mock_run.stdout = b"out"
        mock_run.stderr = b"err"
        mock_run.returncode = 0
        client = SlurmClient("user", "host", "cluster_name")
        client.put_recursive(
            output_temp_dir,
            "destination",
        )
        mock_subprocess.run.assert_has_calls(
            [
                call(
                    'rsync -e "ssh -S ~/.ssh/control-%h-%p-%r" -avz --delete '
                    f"--exclude-from {output_temp_dir}/.gitignore "
                    f"--exclude {output_temp_dir}/tests "
                    f"{output_temp_dir} user@host:destination",
                    shell=True,
                    capture_output=True,
                    timeout=300,
                ),
            ]
        )


def test_slurm_client_put_recursive_failure(mock_subprocess_no_init):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        mock_subprocess_no_init.TimeoutExpired = subprocess.TimeoutExpired
        mock_success_run = Mock()
        mock_success_run.stdout = b"out"
        mock_success_run.stderr = b"err"
        mock_success_run.returncode = 0
        mock_run = Mock()
        mock_subprocess_no_init.run.side_effect = [
            mock_success_run,
            mock_success_run,
            mock_success_run,
            mock_success_run,
            mock_run,
        ]
        mock_run.stdout = b"out"
        mock_run.stderr = b"err"
        mock_run.returncode = 1
        with pytest.raises(RuntimeError, match="Rsync failed. stderr: err"):
            client = SlurmClient("user", "host", "cluster_name")
            client.put_recursive(
                output_temp_dir,
                "destination",
            )
        mock_subprocess_no_init.run.assert_has_calls(
            [
                call(
                    'rsync -e "ssh -S ~/.ssh/control-%h-%p-%r" -avz --delete '
                    f"{output_temp_dir} user@host:destination",
                    shell=True,
                    capture_output=True,
                    timeout=300,
                ),
            ]
        )


def test_slurm_client_put_recursive_timeout(mock_subprocess_no_init):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        mock_subprocess_no_init.TimeoutExpired = subprocess.TimeoutExpired
        mock_success_run = Mock()
        mock_success_run.stdout = b"out"
        mock_success_run.stderr = b"err"
        mock_success_run.returncode = 0
        mock_run = Mock()
        mock_subprocess_no_init.run.side_effect = [
            mock_success_run,
            mock_success_run,
            mock_success_run,
            mock_success_run,
            subprocess.TimeoutExpired("Timeout!", 1),
        ]
        mock_run.stdout = b"out"
        mock_run.stderr = b"err"
        mock_run.returncode = 1
        with pytest.raises(RuntimeError, match="Timeout while running rsync command."):
            client = SlurmClient("user", "host", "cluster_name")
            client.put_recursive(
                output_temp_dir,
                "destination",
            )
        mock_subprocess_no_init.run.assert_has_calls(
            [
                call(
                    'rsync -e "ssh -S ~/.ssh/control-%h-%p-%r" -avz --delete '
                    f"{output_temp_dir} user@host:destination",
                    shell=True,
                    capture_output=True,
                    timeout=300,
                ),
            ]
        )


def test_slurm_client_put_recursive_memory_error(mock_subprocess_no_init):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        mock_subprocess_no_init.TimeoutExpired = subprocess.TimeoutExpired
        mock_success_run = Mock()
        mock_success_run.stdout = b"out"
        mock_success_run.stderr = b"err"
        mock_success_run.returncode = 0
        mock_run = Mock()
        mock_subprocess_no_init.run.side_effect = [
            mock_success_run,
            mock_success_run,
            mock_success_run,
            mock_success_run,
            MemoryError("OOM!"),
        ]
        mock_run.stdout = b"out"
        mock_run.stderr = b"err"
        mock_run.returncode = 1
        with pytest.raises(MemoryError, match="OOM!"):
            client = SlurmClient("user", "host", "cluster_name")
            client.put_recursive(
                output_temp_dir,
                "destination",
            )
        mock_subprocess_no_init.run.assert_has_calls(
            [
                call(
                    'rsync -e "ssh -S ~/.ssh/control-%h-%p-%r" -avz --delete '
                    f"{output_temp_dir} user@host:destination",
                    shell=True,
                    capture_output=True,
                    timeout=300,
                ),
            ]
        )


def test_slurm_client_put_success(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = b"out"
    mock_run.stderr = b"err"
    mock_run.returncode = 0
    client = SlurmClient("user", "host", "cluster_name")
    client.put(
        file_contents="file contents",
        destination="destination/file.txt",
    )
    mock_subprocess.run.assert_has_calls(
        [
            call(
                _run_commands_template(
                    [
                        "mkdir -p destination",
                        "touch destination/file.txt",
                        'cat <<"SCRIPTFILETAG" > destination/file.txt',
                        "file contents",
                        "SCRIPTFILETAG",
                    ]
                ),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
        ]
    )


def test_slurm_client_put_failure(mock_subprocess):
    mock_success_run = Mock()
    mock_success_run.stdout = b"out"
    mock_success_run.stderr = b"err"
    mock_success_run.returncode = 0
    mock_run = Mock()
    mock_subprocess.run.side_effect = [
        mock_success_run,
        mock_success_run,
        mock_run,
    ]
    mock_run.stdout = b"out"
    mock_run.stderr = b"err"
    mock_run.returncode = 1
    with pytest.raises(RuntimeError, match="Failed to write file. stderr: err"):
        client = SlurmClient("user", "host", "cluster_name")
        client.put(
            file_contents="file contents",
            destination="destination/file.txt",
        )
    mock_subprocess.run.assert_has_calls(
        [
            call(
                _run_commands_template(
                    [
                        "mkdir -p destination",
                        "touch destination/file.txt",
                        'cat <<"SCRIPTFILETAG" > destination/file.txt',
                        "file contents",
                        "SCRIPTFILETAG",
                    ]
                ),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
        ]
    )


def test_slurm_client_put_other_exception(mock_subprocess):
    mock_success_run = Mock()
    mock_success_run.stdout = b"out"
    mock_success_run.stderr = b"err"
    mock_success_run.returncode = 0
    mock_subprocess.run.side_effect = [
        mock_success_run,
        mock_success_run,
        ValueError("Dummy test exception!"),
    ]
    with pytest.raises(ValueError, match="Dummy test exception!"):
        client = SlurmClient("user", "host", "cluster_name")
        client.put(
            file_contents="file contents",
            destination="destination/file.txt",
        )
    mock_subprocess.run.assert_has_calls(
        [
            call(
                _run_commands_template(
                    [
                        "mkdir -p destination",
                        "touch destination/file.txt",
                        'cat <<"SCRIPTFILETAG" > destination/file.txt',
                        "file contents",
                        "SCRIPTFILETAG",
                    ]
                ),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
        ]
    )


def test_slurm_client_get_active_users(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = (
        b"control-host-22-matthew\n"
        b"control-host-22-user1\n"
        b"control-host-22-user2\n"
        b"control-host-22-user-with-dash-in-name\n"
    )
    mock_run.stderr = b"foo"
    mock_run.returncode = 0

    active_users = SlurmClient.get_active_users("host")
    mock_subprocess.run.assert_called_with(
        "ls ~/.ssh/ | egrep 'control-host-.*-.*'",
        shell=True,
        capture_output=True,
    )
    assert set(active_users) == {"matthew", "user1", "user2", "user-with-dash-in-name"}


def test_slurm_client_get_active_users_empty(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = b""
    mock_run.stderr = b"foo"
    mock_run.returncode = 0

    active_users = SlurmClient.get_active_users("host")
    mock_subprocess.run.assert_called_with(
        "ls ~/.ssh/ | egrep 'control-host-.*-.*'",
        shell=True,
        capture_output=True,
    )
    assert active_users == []


def test_slurm_client_get_active_users_failure(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = (
        b"control-host-22-matthew\ncontrol-host-22-user1\ncontrol-host-22-user2\n"
    )
    mock_run.stderr = b"foo"
    mock_run.returncode = 1

    active_users = SlurmClient.get_active_users("host")
    mock_subprocess.run.assert_called_with(
        "ls ~/.ssh/ | egrep 'control-host-.*-.*'",
        shell=True,
        capture_output=True,
    )
    assert active_users == []
