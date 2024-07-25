from unittest.mock import MagicMock, Mock, patch

import pytest
from fabric import Connection

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
    result = client.submit_job("./job.sh", 2, client.SupportedQueues.DEBUG)
    mock_connection.run.assert_called_with(
        "qsub -l select=2:system=polaris -q debug ./job.sh"
    )
    assert result == "2032"


def test_polaris_client_submit_job_debugscaling(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_connection.run.return_value = mock_command
    mock_command.stdout = "2032341411.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"
    client = PolarisClient("user")
    result = client.submit_job("./job.sh", 2, client.SupportedQueues.DEBUG_SCALING)
    mock_connection.run.assert_called_with(
        "qsub -l select=2:system=polaris -q debug-scaling ./job.sh"
    )
    assert result == "2032341411"


def test_polaris_client_submit_job_PROD(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_connection.run.return_value = mock_command
    mock_command.stdout = "3141592653.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"
    client = PolarisClient("user")
    result = client.submit_job("./job.sh", 2, client.SupportedQueues.PROD)
    mock_connection.run.assert_called_with(
        "qsub -l select=2:system=polaris -q prod ./job.sh"
    )
    assert result == "3141592653"


def test_polaris_client_submit_job_invalid_job_format(mock_fabric, mock_auth):
    mock_connection = Mock(spec=Connection)
    mock_fabric.side_effect = [mock_connection]
    mock_command = Mock()
    mock_connection.run.return_value = mock_command
    mock_command.stdout = "3141592653polaris-pbs-01"
    client = PolarisClient("user")
    result = client.submit_job("./job.sh", 2, client.SupportedQueues.PROD)
    mock_connection.run.assert_called_with(
        "qsub -l select=2:system=polaris -q prod ./job.sh"
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
        _ = client.submit_job("./job.sh", 2, client.SupportedQueues.PROD)


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
    result = client.submit_job("./job.sh", 2, client.SupportedQueues.PROD)
    mock_connection.run.assert_called_with(
        "qsub -l select=2:system=polaris -q prod ./job.sh"
    )
    mock_connection.close.assert_called_once()
    mock_connection2.run.assert_called_with(
        "qsub -l select=2:system=polaris -q prod ./job.sh"
    )
    mock_connection2.open.assert_called_once()
    assert result == "-pbs-01"


# cancel_job
# list_jobs
# get_job
# run_commands
