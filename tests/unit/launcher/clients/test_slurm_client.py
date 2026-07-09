import signal
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest

from oumi.core.launcher import JobState
from oumi.launcher.clients.slurm_client import SlurmClient

_CTRL_PATH: str = "-S ~/.ssh/control-%h-%p-%r"
# Frozen via the mock_datetime fixture: "now" is 2025-01-31, 30 days back.
_SACCT_CMD = (
    "sacct --user=user --format='JobId%-30,JobName%30,User%30,State%30,Reason%30' "
    "-X --starttime 2025-01-01"
)


#
# Fixtures
#
@pytest.fixture
def mock_datetime():
    with patch("oumi.launcher.clients.slurm_client.datetime") as dt:
        dt.now.return_value = datetime(2025, 1, 31)
        yield dt


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
                        "sbatch --nodes=2 --output=$HOME/oumi_slurm_logs/%j.out "
                        "--parsable ./job.sh",
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
                        "sbatch --nodes=2 --job-name=somename "
                        "--output=$HOME/oumi_slurm_logs/%j.out --parsable "
                        "./job.sh",
                    ]
                ),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
        ]
    )
    assert result == "2032"


def test_slurm_client_submit_job_with_all_args(mock_subprocess):
    mock_run = Mock()
    mock_subprocess.run.return_value = mock_run
    mock_run.stdout = b"2032"
    mock_run.stderr = b"err"
    mock_run.returncode = 0
    client = SlurmClient("user", "host", "cluster_name")
    result = client.submit_job(
        "./job.sh",
        "work_dir",
        2,
        name="somename",
        export="NONE",
        account="oumi",
        ntasks=2,
        threads_per_core=1,
        distribution="block:cyclic",
        partition="extended",
        qos="debug",
        stdout_file="~/stdout.txt",
        stderr_file="$HOME/stderr.txt",
        # kwargs
        foo="bar",
    )
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
                        " ".join(
                            [
                                "sbatch",
                                "--nodes=2",
                                "--job-name=somename",
                                "--export=NONE",
                                "--account=oumi",
                                "--ntasks=2",
                                "--threads-per-core=1",
                                "--distribution=block:cyclic",
                                "--partition=extended",
                                "--qos=debug",
                                "--output=~/stdout.txt",
                                "--error=$HOME/stderr.txt",
                                "--foo=bar",
                                "--parsable",
                                "./job.sh",
                            ]
                        ),
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
                        "sbatch --nodes=2 --output=$HOME/oumi_slurm_logs/%j.out "
                        "--parsable ./job.sh",
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
                        "sbatch --nodes=2 --output=$HOME/oumi_slurm_logs/%j.out "
                        "--parsable ./job.sh",
                    ]
                ),
                shell=True,
                capture_output=True,
                timeout=180,
            ),
        ]
    )
    assert result == "3141592653polaris-pbs-01"


def test_slurm_client_list_jobs_success(mock_subprocess, mock_datetime):
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


def test_slurm_client_list_jobs_first_login_success(mock_subprocess, mock_datetime):
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


def test_slurm_client_list_jobs_handles_empty_string(mock_subprocess, mock_datetime):
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


def test_slurm_client_list_jobs_failure(mock_subprocess, mock_datetime):
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


def _mock_refresh_creds_run() -> Mock:
    """Mock for the ``ssh -O check`` that ``@retry_auth`` runs before each command."""
    r = Mock()
    r.stdout = b""
    r.stderr = b""
    r.returncode = 0
    return r


def test_slurm_client_get_job_returns_active_job_from_squeue(mock_subprocess):
    squeue_ok = Mock()
    squeue_ok.stdout = b"100 myjob user RUNNING 1700000000 node-1\n"
    squeue_ok.stderr = b""
    squeue_ok.returncode = 0

    mock_subprocess.run.side_effect = [
        _mock_refresh_creds_run(),
        _mock_refresh_creds_run(),
        squeue_ok,
    ]

    client = SlurmClient("user", "host", "cluster_name")
    job_status = client.get_job("100")

    assert job_status is not None
    assert job_status.id == "100"
    assert job_status.state == JobState.RUNNING
    assert job_status.submit_time == 1700000000.0


def test_slurm_client_get_job_falls_back_to_scontrol_for_terminal_state(
    mock_subprocess,
):
    squeue_empty = Mock()
    squeue_empty.stdout = b""
    squeue_empty.stderr = b""
    squeue_empty.returncode = 0

    scontrol_ok = Mock()
    scontrol_ok.stdout = (
        b"JobId=100 JobName=myjob\n"
        b"   UserId=user(1000) GroupId=user(1000) MCS_label=N/A\n"
        b"   JobState=COMPLETED Reason=None Dependency=(null)\n"
        b"   SubmitTime=1700000000 StartTime=1700000050\n"
    )
    scontrol_ok.stderr = b""
    scontrol_ok.returncode = 0

    mock_subprocess.run.side_effect = [
        _mock_refresh_creds_run(),
        _mock_refresh_creds_run(),
        squeue_empty,
        _mock_refresh_creds_run(),
        scontrol_ok,
    ]

    client = SlurmClient("user", "host", "cluster_name")
    job_status = client.get_job("100")

    assert job_status is not None
    assert job_status.id == "100"
    assert job_status.state == JobState.SUCCEEDED
    assert job_status.done is True
    assert job_status.submit_time == 1700000000.0


def test_slurm_client_get_job_returns_none_when_purged(mock_subprocess):
    squeue_empty = Mock()
    squeue_empty.stdout = b""
    squeue_empty.stderr = b""
    squeue_empty.returncode = 0

    scontrol_fail = Mock()
    scontrol_fail.stdout = b""
    scontrol_fail.stderr = b"slurm_load_jobs error: Invalid job id specified\n"
    scontrol_fail.returncode = 1

    mock_subprocess.run.side_effect = [
        _mock_refresh_creds_run(),
        _mock_refresh_creds_run(),
        squeue_empty,
        _mock_refresh_creds_run(),
        scontrol_fail,
    ]

    client = SlurmClient("user", "host", "cluster_name")
    assert client.get_job("999") is None


def test_slurm_client_get_job_squeue_failure_raises(mock_subprocess):
    squeue_fail = Mock()
    squeue_fail.stdout = b""
    squeue_fail.stderr = b"squeue: error: connection refused\n"
    squeue_fail.returncode = 1

    mock_subprocess.run.side_effect = [
        _mock_refresh_creds_run(),
        _mock_refresh_creds_run(),
        squeue_fail,
    ]

    client = SlurmClient("user", "host", "cluster_name")
    with pytest.raises(RuntimeError, match="Failed to list jobs via squeue"):
        _ = client.get_job("100")


def test_slurm_client_cancel_success(mock_subprocess):
    scancel_ok = Mock()
    scancel_ok.stdout = b""
    scancel_ok.stderr = b""
    scancel_ok.returncode = 0

    squeue_ok = Mock()
    squeue_ok.stdout = b"7.batch batch user RUNNING 1700000000 node-1\n"
    squeue_ok.stderr = b""
    squeue_ok.returncode = 0

    mock_subprocess.run.side_effect = [
        _mock_refresh_creds_run(),
        _mock_refresh_creds_run(),
        scancel_ok,
        _mock_refresh_creds_run(),
        squeue_ok,
    ]

    client = SlurmClient("user", "host", "cluster_name")
    job_status = client.cancel("7.batch")

    assert job_status is not None
    assert job_status.id == "7.batch"
    assert job_status.state == JobState.RUNNING


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


def test_slurm_client_cancel_squeue_failure(mock_subprocess):
    scancel_ok = Mock()
    scancel_ok.stdout = b""
    scancel_ok.stderr = b""
    scancel_ok.returncode = 0

    squeue_fail = Mock()
    squeue_fail.stdout = b""
    squeue_fail.stderr = b"foo"
    squeue_fail.returncode = 1

    mock_subprocess.run.side_effect = [
        _mock_refresh_creds_run(),
        _mock_refresh_creds_run(),
        scancel_ok,
        _mock_refresh_creds_run(),
        squeue_fail,
    ]

    with pytest.raises(RuntimeError, match="Failed to list jobs via squeue"):
        client = SlurmClient("user", "host", "cluster_name")
        _ = client.cancel("2017652")


def test_slurm_client_cancel_job_not_found_success(mock_subprocess):
    scancel_ok = Mock()
    scancel_ok.stdout = b""
    scancel_ok.stderr = b""
    scancel_ok.returncode = 0

    squeue_empty = Mock()
    squeue_empty.stdout = b""
    squeue_empty.stderr = b""
    squeue_empty.returncode = 0

    scontrol_fail = Mock()
    scontrol_fail.stdout = b""
    scontrol_fail.stderr = b"slurm_load_jobs error: Invalid job id specified\n"
    scontrol_fail.returncode = 1

    mock_subprocess.run.side_effect = [
        _mock_refresh_creds_run(),
        _mock_refresh_creds_run(),
        scancel_ok,
        _mock_refresh_creds_run(),
        squeue_empty,
        _mock_refresh_creds_run(),
        scontrol_fail,
    ]

    client = SlurmClient("user", "host", "cluster_name")
    assert client.cancel("2017652") is None


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


#
# SlurmLogStream — tail-subprocess termination
#
def _build_log_stream(mock_proc, mock_client):
    """Construct a SlurmLogStream that bypasses the live SSH preflight.

    ``__init__`` normally calls ``_start_job_checking`` inside
    ``_start_tail_process`` — patching the latter skips it, so we invoke
    the watcher explicitly here.
    """
    from oumi.launcher.clients.slurm_client import SlurmLogStream

    with patch.object(SlurmLogStream, "_start_tail_process", return_value=mock_proc):
        stream = SlurmLogStream("test-cluster", "job-123", mock_client)
    stream._start_job_checking(mock_proc)
    return stream


def test_slurm_log_stream_terminates_tail_when_job_done():
    """Happy path: job completes, watcher signals the ``tail`` process
    group so the SSH child gets the signal too (not just the ``sh`` wrapper).
    """
    mock_proc = Mock()
    mock_proc.pid = 12345
    mock_job = Mock(done=True)
    mock_client = Mock()
    mock_client.get_job.return_value = mock_job

    with patch("oumi.launcher.clients.slurm_client.os.killpg") as mock_killpg:
        stream = _build_log_stream(mock_proc, mock_client)
        assert stream._job_check_thread is not None
        stream._job_check_thread.join(timeout=2)
        assert not stream._job_check_thread.is_alive()
        mock_killpg.assert_any_call(12345, signal.SIGTERM)


def test_slurm_log_stream_terminates_tail_when_get_job_raises():
    """Regression: a transient ``get_job`` exception used to ``break`` out
    of the watcher loop without calling ``proc.terminate()``, leaving the
    ``tail -F`` running against a static log file. The ``finally``-block
    must always signal the proc.
    """
    mock_proc = Mock()
    mock_proc.pid = 12345
    mock_client = Mock()
    mock_client.get_job.side_effect = RuntimeError("ssh wedged")

    with patch("oumi.launcher.clients.slurm_client.os.killpg") as mock_killpg:
        stream = _build_log_stream(mock_proc, mock_client)
        assert stream._job_check_thread is not None
        stream._job_check_thread.join(timeout=2)
        assert not stream._job_check_thread.is_alive()
        mock_killpg.assert_any_call(12345, signal.SIGTERM)


def test_slurm_log_stream_escalates_to_sigkill_when_wait_times_out():
    """If ``proc.wait`` times out after SIGTERM, the watcher must escalate
    to SIGKILL on the process group so the consumer always sees EOF.
    """
    mock_proc = Mock()
    mock_proc.pid = 12345
    mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="tail", timeout=5)
    mock_job = Mock(done=True)
    mock_client = Mock()
    mock_client.get_job.return_value = mock_job

    with patch("oumi.launcher.clients.slurm_client.os.killpg") as mock_killpg:
        stream = _build_log_stream(mock_proc, mock_client)
        assert stream._job_check_thread is not None
        stream._job_check_thread.join(timeout=2)
        sigterm = [
            c for c in mock_killpg.call_args_list if c == call(12345, signal.SIGTERM)
        ]
        sigkill = [
            c for c in mock_killpg.call_args_list if c == call(12345, signal.SIGKILL)
        ]
        assert sigterm, "SIGTERM should be sent first"
        assert sigkill, "SIGKILL should escalate after wait timeout"
