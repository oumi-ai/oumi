from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from fabric import Connection

from lema.core.types.base_cluster import JobStatus
from lema.launcher.clients.polaris_client import PolarisClient


#
# Fixtures
#
@pytest.fixture
def mock_fabric():
    with patch("lema.launcher.clients.polaris_client.Connection") as mock_connection:
        yield mock_connection


@pytest.fixture
def mock_auth():
    with patch("lema.launcher.clients.polaris_client.getpass") as mock_getpass:
        mock_getpass.return_value = "password"
        yield mock_getpass


@pytest.fixture
def mock_patchwork():
    with patch("lema.launcher.clients.polaris_client.rsync") as mock_rsync:
        yield mock_rsync


def _get_test_data(file_name: str) -> str:
    data_path = Path(__file__).parent / "data" / file_name
    with open(data_path) as f:
        return f.read()


#
# Tests
#
def test_polaris_client_init(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    _ = PolarisClient("user")
    mock_fabric.assert_called_with(
        "polaris.alcf.anl.gov", user="user", connect_kwargs={"password": "password"}
    )
    mock_connection.open.assert_called_once()
    mock_connection.close.assert_not_called()


def test_polaris_client_refresh_creds(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_connection2 = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection, mock_connection2]
    client = PolarisClient("user")
    mock_fabric.assert_called_with(
        "polaris.alcf.anl.gov", user="user", connect_kwargs={"password": "password"}
    )
    mock_connection.open.assert_called_once()
    client.refresh_creds(close_connection=True)
    mock_connection.close.assert_called_once()
    mock_fabric.assert_called_with(
        "polaris.alcf.anl.gov", user="user", connect_kwargs={"password": "password"}
    )
    mock_connection2.open.assert_called_once()


def test_polaris_client_submit_job_debug(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_connection.run.return_value = mock_command
    mock_command.stdout = "2032.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"
    client = PolarisClient("user")
    result = client.submit_job("./job.sh", 2, client.SupportedQueues.DEBUG, None)
    mock_connection.run.assert_called_with(
        "qsub -l select=2:system=polaris -q debug  ./job.sh",
        warn=True,
    )
    assert result == "2032"


def test_polaris_client_submit_job_demand(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_connection.run.return_value = mock_command
    mock_command.stdout = "2032.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"
    client = PolarisClient("user")
    result = client.submit_job("./job.sh", 2, client.SupportedQueues.DEMAND, None)
    mock_connection.run.assert_called_with(
        "qsub -l select=2:system=polaris -q demand  ./job.sh",
        warn=True,
    )
    assert result == "2032"


def test_polaris_client_submit_job_preemptable(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_connection.run.return_value = mock_command
    mock_command.stdout = "2032.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"
    client = PolarisClient("user")
    result = client.submit_job("./job.sh", 2, client.SupportedQueues.PREEMPTABLE, None)
    mock_connection.run.assert_called_with(
        "qsub -l select=2:system=polaris -q preemptable  ./job.sh",
        warn=True,
    )
    assert result == "2032"


def test_polaris_client_submit_job_debug_name(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_connection.run.return_value = mock_command
    mock_command.stdout = "2032.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"
    client = PolarisClient("user")
    result = client.submit_job("./job.sh", 2, client.SupportedQueues.DEBUG, "somename")
    mock_connection.run.assert_called_with(
        "qsub -l select=2:system=polaris -q debug -N somename ./job.sh",
        warn=True,
    )
    assert result == "2032"


def test_polaris_client_submit_job_debug_scaling(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_connection.run.return_value = mock_command
    mock_command.stdout = "2032341411.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"
    client = PolarisClient("user")
    result = client.submit_job(
        "./job.sh",
        2,
        client.SupportedQueues.DEBUG_SCALING,
        None,
    )
    mock_connection.run.assert_called_with(
        "qsub -l select=2:system=polaris -q debug-scaling  ./job.sh",
        warn=True,
    )
    assert result == "2032341411"


def test_polaris_client_submit_job_prod(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_connection.run.return_value = mock_command
    mock_command.stdout = "3141592653.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"
    client = PolarisClient("user")
    result = client.submit_job("./job.sh", 2, client.SupportedQueues.PROD, None)
    mock_connection.run.assert_called_with(
        "qsub -l select=2:system=polaris -q prod  ./job.sh",
        warn=True,
    )
    assert result == "3141592653"


def test_polaris_client_submit_job_invalid_job_format(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_connection.run.return_value = mock_command
    mock_command.stdout = "3141592653polaris-pbs-01"
    client = PolarisClient("user")
    result = client.submit_job("./job.sh", 2, client.SupportedQueues.PROD, None)
    mock_connection.run.assert_called_with(
        "qsub -l select=2:system=polaris -q prod  ./job.sh",
        warn=True,
    )
    assert result == "3141592653polaris-pbs-01"


def test_polaris_client_submit_job_error(mock_fabric, mock_auth):
    with pytest.raises(RuntimeError):
        mock_connection = Mock(spec=Connection)
        mock_fabric.side_effect = [mock_connection]
        mock_result = MagicMock()
        mock_result.__bool__.return_value = False
        mock_result.stderr = "error"
        mock_connection.run.return_value = mock_result
        client = PolarisClient("user")
        _ = client.submit_job("./job.sh", 2, client.SupportedQueues.PROD, None)


def test_polaris_client_submit_job_retry_auth(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_connection2 = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection, mock_connection2]
    mock_command = Mock()
    mock_connection.run.side_effect = [EOFError]
    mock_command.stdout = "3141592653polaris-pbs-01"
    mock_command2 = Mock()
    mock_command2.stdout = "-pbs-01"
    mock_connection.run.return_value = mock_command
    mock_connection2.run.return_value = mock_command2
    client = PolarisClient("user")
    result = client.submit_job("./job.sh", 2, client.SupportedQueues.PROD, None)
    mock_connection.run.assert_called_with(
        "qsub -l select=2:system=polaris -q prod  ./job.sh",
        warn=True,
    )
    mock_connection.close.assert_called_once()
    mock_connection2.run.assert_called_with(
        "qsub -l select=2:system=polaris -q prod  ./job.sh",
        warn=True,
    )
    mock_connection2.open.assert_called_once()
    assert result == "-pbs-01"


def test_polaris_client_list_jobs_success_debug(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_command.stdout = _get_test_data("qstat.txt")
    mock_connection.run.return_value = mock_command
    client = PolarisClient("user")
    job_list = client.list_jobs(client.SupportedQueues.DEBUG)
    mock_connection.run.assert_called_with("qstat -s -x -w -u user", warn=True)
    job_ids = [job.id for job in job_list]
    expected_ids = [
        "2017611",
        "2017643",
        "2017652",
        "2017654",
        "2018469",
        "2019593",
        "2019726",
        "2019730",
        "2019731",
        "2019743",
        "2019765",
        "2019769",
        "2021153",
        "2037042",
        "2037048",
    ]
    assert job_ids == expected_ids


def test_polaris_client_list_jobs_success_debug_scaling(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_command.stdout = _get_test_data("qstat.txt")
    mock_connection.run.return_value = mock_command
    client = PolarisClient("user")
    job_list = client.list_jobs(client.SupportedQueues.DEBUG_SCALING)
    mock_connection.run.assert_called_with("qstat -s -x -w -u user", warn=True)
    job_ids = [job.id for job in job_list]
    expected_ids = [
        "2029871",
        "2029885",
    ]
    assert job_ids == expected_ids


def test_polaris_client_list_jobs_success_prod(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_command.stdout = _get_test_data("qstat.txt")
    mock_connection.run.return_value = mock_command
    client = PolarisClient("user")
    job_list = client.list_jobs(client.SupportedQueues.PROD)
    mock_connection.run.assert_called_with("qstat -s -x -w -u user", warn=True)
    job_ids = [job.id for job in job_list]
    expected_ids = [
        "123",
        "234",
        "345",
        "456",
        "567",
        "678",
    ]
    assert job_ids == expected_ids


def test_polaris_client_list_jobs_handles_empty_string(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_command.stdout = ""
    mock_connection.run.return_value = mock_command
    client = PolarisClient("user")
    job_list = client.list_jobs(client.SupportedQueues.DEBUG)
    mock_connection.run.assert_called_with("qstat -s -x -w -u user", warn=True)
    job_ids = [job.id for job in job_list]
    expected_ids = []
    assert job_ids == expected_ids


def test_polaris_client_list_jobs_failure(mock_fabric, mock_auth):
    with pytest.raises(RuntimeError):
        mock_connection = Mock(spec=Connection)
        mock_fabric.side_effect = [mock_connection]
        mock_result = MagicMock()
        mock_result.__bool__.return_value = False
        mock_result.stderr = "error"
        mock_connection.run.return_value = mock_result
        client = PolarisClient("user")
        _ = client.list_jobs(client.SupportedQueues.DEBUG)


def test_polaris_client_get_job_success(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_command.stdout = _get_test_data("qstat.txt")
    mock_connection.run.return_value = mock_command
    client = PolarisClient("user")
    job_status = client.get_job("2017652", client.SupportedQueues.DEBUG)
    mock_connection.run.assert_called_with("qstat -s -x -w -u user", warn=True)
    expected_status = JobStatus(
        id="2017652",
        name="example_job.sh",
        status="F",
        cluster="debug",
        metadata=(
            "                                                                      "
            "                             Req'd  Req'd   Elap\n"
            "Job ID                         Username        Queue           Jobname"
            "         SessID   NDS  TSK   Memory Time  S Time\n"
            "------------------------------ --------------- --------------- "
            "--------------- -------- ---- ----- ------ ----- - -----\n"
            "2017652.polaris-pbs-01.hsn.cm* matthew         debug           "
            "example_job.sh   2354947    1    64    --  00:10 F 00:00:43\n"
            "   Job run at Wed Jul 10 at 23:28 on (x3006c0s19b1n0:ncpus=64) and "
            "failed"
        ),
    )
    assert job_status == expected_status


def test_polaris_client_get_job_not_found(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_command.stdout = _get_test_data("qstat.txt")
    mock_connection.run.return_value = mock_command
    client = PolarisClient("user")
    job_status = client.get_job("2017652", client.SupportedQueues.DEBUG_SCALING)
    mock_connection.run.assert_called_with("qstat -s -x -w -u user", warn=True)
    assert job_status is None


def test_polaris_client_get_job_failure(mock_fabric, mock_auth):
    with pytest.raises(RuntimeError):
        mock_connection = Mock(spec=Connection)
        mock_fabric.side_effect = [mock_connection]
        mock_result = MagicMock()
        mock_result.__bool__.return_value = False
        mock_result.stderr = "error"
        mock_connection.run.return_value = mock_result
        client = PolarisClient("user")
        _ = client.get_job("2017652", client.SupportedQueues.DEBUG_SCALING)


def test_polaris_client_cancel_success(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_qdel_command = Mock()
    mock_qdel_command.stdout = ""
    mock_qstat_command = Mock()
    mock_qstat_command.stdout = _get_test_data("qstat.txt")
    mock_connection.run.side_effect = [mock_qdel_command, mock_qstat_command]
    client = PolarisClient("user")
    job_status = client.cancel("2017652", client.SupportedQueues.DEBUG)
    mock_connection.run.assert_has_calls(
        [
            call("qdel 2017652", warn=True),
            call("qstat -s -x -w -u user", warn=True),
        ]
    )
    expected_status = JobStatus(
        id="2017652",
        name="example_job.sh",
        status="F",
        cluster="debug",
        metadata=(
            "                                                                      "
            "                             Req'd  Req'd   Elap\n"
            "Job ID                         Username        Queue           Jobname"
            "         SessID   NDS  TSK   Memory Time  S Time\n"
            "------------------------------ --------------- --------------- "
            "--------------- -------- ---- ----- ------ ----- - -----\n"
            "2017652.polaris-pbs-01.hsn.cm* matthew         debug           "
            "example_job.sh   2354947    1    64    --  00:10 F 00:00:43\n"
            "   Job run at Wed Jul 10 at 23:28 on (x3006c0s19b1n0:ncpus=64) and "
            "failed"
        ),
    )
    assert job_status == expected_status


def test_polaris_client_cancel_qdel_failure(mock_fabric, mock_auth):
    with pytest.raises(RuntimeError):
        mock_connection = Mock(spec=Connection)
        mock_fabric.side_effect = [mock_connection]
        mock_qdel_command = MagicMock()
        mock_qdel_command.__bool__.return_value = False
        mock_connection.run.side_effect = [mock_qdel_command]
        client = PolarisClient("user")
        _ = client.cancel("2017652", client.SupportedQueues.DEBUG)


def test_polaris_client_cancel_qstat_failure(mock_fabric, mock_auth):
    with pytest.raises(RuntimeError):
        mock_connection = Mock(spec=Connection)
        mock_fabric.side_effect = [mock_connection]
        mock_qdel_command = Mock()
        mock_qdel_command.stdout = ""
        mock_qstat_command = MagicMock()
        mock_qstat_command.__bool__.return_value = False
        mock_connection.run.side_effect = [mock_qdel_command, mock_qstat_command]
        client = PolarisClient("user")
        _ = client.cancel("2017652", client.SupportedQueues.DEBUG)


def test_polaris_client_cancel_job_not_found_success(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_qdel_command = Mock()
    mock_qdel_command.stdout = ""
    mock_qstat_command = Mock()
    mock_qstat_command.stdout = _get_test_data("qstat.txt")
    mock_connection.run.side_effect = [mock_qdel_command, mock_qstat_command]
    client = PolarisClient("user")
    job_status = client.cancel("2017652", client.SupportedQueues.PROD)
    mock_connection.run.assert_has_calls(
        [
            call("qdel 2017652", warn=True),
            call("qstat -s -x -w -u user", warn=True),
        ]
    )
    assert job_status is None


def test_polaris_client_run_commands_success(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_first_command = Mock()
    mock_fourth_command = Mock()
    mock_final_command = Mock()
    mock_connection.run.side_effect = [
        mock_first_command,
        mock_fourth_command,
        mock_final_command,
    ]
    mock_second_command = MagicMock()
    mock_third_command = MagicMock()
    mock_fifth_command = MagicMock()
    mock_connection.cd.side_effect = [
        mock_second_command,
        mock_third_command,
        mock_fifth_command,
    ]
    mock_third_command.run.side_effect = [mock_fourth_command]
    mock_fifth_command.run.side_effect = [mock_final_command]
    commands = [
        "first command",
        "cd second/command",
        "cd third/command",
        "fourth command",
        "cd fifth/command",
        "final command",
    ]
    client = PolarisClient("user")
    client.run_commands(commands)
    mock_connection.cd.assert_has_calls(
        [
            call("second/command"),
            call("third/command"),
            call("fifth/command"),
        ]
    )
    mock_connection.run.assert_has_calls(
        [
            call("first command", warn=True),
            call("fourth command", warn=True),
            call("final command", warn=True),
        ]
    )


def test_polaris_client_run_commands_success_empty(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    client = PolarisClient("user")
    client.run_commands([])
    mock_connection.cd.never_called()
    mock_connection.run.never_called()


def test_polaris_client_run_commands_fails(mock_fabric, mock_auth):
    with pytest.raises(RuntimeError):
        mock_connection = Mock(spec=Connection)
        mock_fabric.side_effect = [mock_connection]
        mock_first_command = Mock()
        mock_fourth_command = MagicMock()
        mock_fourth_command.__bool__.return_value = False
        mock_connection.run.side_effect = [mock_first_command, mock_fourth_command]
        mock_second_command = MagicMock()
        mock_third_command = MagicMock()
        mock_fifth_command = MagicMock()
        mock_connection.cd.side_effect = [
            mock_second_command,
            mock_third_command,
            mock_fifth_command,
        ]
        mock_third_command.run.side_effect = [mock_fourth_command]
        commands = [
            "first command",
            "cd second/command",
            "cd third/command",
            "fourth command",
            "cd fifth/command",
            "final command",
        ]
        client = PolarisClient("user")
        client.run_commands(commands)


def test_polaris_client_rsync_success(mock_fabric, mock_auth, mock_patchwork):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_patchwork.return_value = mock_command
    client = PolarisClient("user")
    client.rsync(
        source="source",
        destination="destination",
        delete=True,
        exclude=["foo", "bar"],
        rsync_opts="-avz",
    )
    mock_patchwork.assert_called_once_with(
        c=mock_connection,
        source="source",
        target="destination",
        exclude=["foo", "bar"],
        delete=True,
        rsync_opts="-avz",
    )


def test_polaris_client_rsync_failure(mock_fabric, mock_auth, mock_patchwork):
    with pytest.raises(RuntimeError):
        mock_connection = Mock(spec=Connection)
        mock_fabric.side_effect = [mock_connection]
        mock_command = MagicMock()
        mock_command.__bool__.return_value = False
        mock_patchwork.return_value = mock_command
        client = PolarisClient("user")
        client.rsync(
            source="source",
            destination="destination",
            delete=True,
            exclude=["foo", "bar"],
            rsync_opts="-avz",
        )
