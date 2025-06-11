import re
from datetime import datetime
from unittest.mock import Mock, call, patch

import pytest

from oumi.core.configs import JobConfig, JobResources, StorageMount
from oumi.core.launcher import JobStatus
from oumi.launcher.clients.slurm_client import SlurmClient
from oumi.launcher.clusters.slurm_cluster import SlurmCluster


#
# Fixtures
#
@pytest.fixture
def mock_slurm_client():
    yield Mock(spec=SlurmClient)


@pytest.fixture
def mock_time():
    with patch("oumi.launcher.clusters.slurm_cluster.time") as mock_t:
        yield mock_t


@pytest.fixture
def mock_datetime():
    with patch("oumi.launcher.clusters.slurm_cluster.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 10, 9, 13, 4, 24, 513094)
        yield mock_dt


@pytest.fixture
def mock_os():
    with patch("oumi.launcher.clusters.slurm_cluster.os") as os_mock:
        os_mock.getenv.return_value = ""
        yield os_mock


def _get_default_job(cloud: str) -> JobConfig:
    resources = JobResources(
        cloud=cloud,
        region="us-central1",
        zone=None,
        accelerators="A100-80GB",
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
        file_mounts={
            "~/home/remote/path.bar": "~/local/path.bar",
            "~/home/remote/path2.txt": "~/local/path2.txt",
        },
        storage_mounts={
            "~/home/remote/path/gcs/": StorageMount(
                source="gs://mybucket/", store="gcs"
            )
        },
        setup=(
            "#SBATCH --gpus-per-task=8 \n#SBATCH --cpus-per-task=4\n"
            "pip install -r requirements.txt"
        ),
        run="./hello_world.sh",
    )


#
# Tests
#


def test_slurm_cluster_parse_cluster_name():
    assert SlurmCluster.parse_cluster_name("user@host") == SlurmCluster.ConnectionInfo(
        user="user", hostname="host"
    )
    assert SlurmCluster.parse_cluster_name(
        "user.-dotdash@192.168.0.1"
    ) == SlurmCluster.ConnectionInfo(user="user.-dotdash", hostname="192.168.0.1")


@pytest.mark.parametrize(
    "invalid_name",
    [
        "multiple@at@signs",
        "white space@hostname",
        "extra$!characters@hostname",
        "@nouser",
        "nohost@",
        "",
    ],
)
def test_slurm_cluster_parse_cluster_name_invalid(invalid_name):
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Invalid cluster name: {invalid_name}. "
            "Must be in the format 'user@hostname'."
        ),
    ):
        SlurmCluster.parse_cluster_name(invalid_name)


def test_slurm_cluster_get_slurm_connections(mock_os):
    mock_os.getenv.return_value = "user@host1,user@host2"
    connections = SlurmCluster.get_slurm_connections()
    assert connections == [
        SlurmCluster.ConnectionInfo(user="user", hostname="host1"),
        SlurmCluster.ConnectionInfo(user="user", hostname="host2"),
    ]


def test_slurm_cluster_get_slurm_connections_whitespace(mock_os):
    mock_os.getenv.return_value = "user@host1 , user2@host2"
    connections = SlurmCluster.get_slurm_connections()
    assert connections == [
        SlurmCluster.ConnectionInfo(user="user", hostname="host1"),
        SlurmCluster.ConnectionInfo(user="user2", hostname="host2"),
    ]


def test_slurm_cluster_get_slurm_connections_empty(mock_os):
    mock_os.getenv.return_value = ""
    connections = SlurmCluster.get_slurm_connections()
    assert connections == []


def test_slurm_cluster_get_slurm_connections_skips_malformed(mock_os):
    mock_os.getenv.return_value = (
        "user1@host1,foob@@@ar, user2@host2 , \", ', user3@host3"
    )
    connections = SlurmCluster.get_slurm_connections()
    assert connections == [
        SlurmCluster.ConnectionInfo(user="user1", hostname="host1"),
        SlurmCluster.ConnectionInfo(user="user2", hostname="host2"),
        SlurmCluster.ConnectionInfo(user="user3", hostname="host3"),
    ]


def test_slurm_cluster_name(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("demand@einstein", mock_slurm_client)
    assert cluster.name() == "demand@einstein"

    cluster = SlurmCluster("user@192.168.0.1", mock_slurm_client)
    assert cluster.name() == "user@192.168.0.1"

    cluster = SlurmCluster("debug-scaling@a", mock_slurm_client)
    assert cluster.name() == "debug-scaling@a"

    cluster = SlurmCluster("1-.foo@2-.foobar", mock_slurm_client)
    assert cluster.name() == "1-.foo@2-.foobar"


def test_slurm_cluster_get_job_valid_id(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug@host", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
        ),
    ]
    job = cluster.get_job("myjob")
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job is not None
    assert job.id == "myjob"
    assert job.cluster == "debug@host"


def test_slurm_cluster_get_job_invalid_id_empty(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug@host", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = []
    job = cluster.get_job("myjob")
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job is None


def test_slurm_cluster_get_job_invalid_id_nonempty(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug@host", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
        ),
    ]
    job = cluster.get_job("wrong job")
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job is None


def test_slurm_cluster_get_jobs_nonempty(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug@host", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
        ),
    ]
    jobs = cluster.get_jobs()
    mock_slurm_client.list_jobs.assert_called_once_with()
    expected_jobs = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="debug@host",
            done=False,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="debug@host",
            done=False,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="debug@host",
            done=False,
        ),
    ]
    assert jobs == expected_jobs


def test_slurm_cluster_get_jobs_empty(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug@host", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = []
    jobs = cluster.get_jobs()
    mock_slurm_client.list_jobs.assert_called_once_with()
    expected_jobs = []
    assert jobs == expected_jobs


def test_slurm_cluster_cancel_job(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("prod@host", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="debug@host",
            done=False,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="debug@host",
            done=False,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="debug@host",
            done=False,
        ),
    ]
    job_status = cluster.cancel_job("job2")
    expected_status = JobStatus(
        id="job2",
        name="some",
        status="running",
        metadata="",
        cluster="prod@host",
        done=False,
    )
    mock_slurm_client.cancel.assert_called_once_with(
        "job2",
    )
    assert job_status == expected_status


def test_slurm_cluster_cancel_job_fails(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("prod@host", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="debug@host",
            done=False,
        ),
    ]
    with pytest.raises(RuntimeError):
        _ = cluster.cancel_job("myjobid")


def test_slurm_cluster_run_job(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug@host", mock_slurm_client)
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_slurm_client.run_commands.return_value = mock_successful_cmd
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="RUNNING",
            metadata="",
            cluster="mycluster",
            done=False,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="RUNNING",
        metadata="",
        cluster="debug@host",
        done=False,
    )
    job_status = cluster.run_job(_get_default_job("slurm"))
    mock_slurm_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "~/oumi_launcher/20241009_130424513094",
            ),
            call(
                "~/local/path.bar",
                "~/home/remote/path.bar",
            ),
            call(
                "~/local/path2.txt",
                "~/home/remote/path2.txt",
            ),
        ],
    )
    mock_slurm_client.run_commands.assert_has_calls(
        [
            call(["chmod +x ~/oumi_launcher/20241009_130424513094/oumi_job.sh"]),
        ]
    )
    job_script = (
        "#!/bin/bash\n#SBATCH --gpus-per-task=8 \n#SBATCH --cpus-per-task=4\n\n"
        "export var1=val1\n\n"
        "pip install -r requirements.txt\n./hello_world.sh\n"
    )
    mock_slurm_client.put.assert_called_once_with(
        job_script, "~/oumi_launcher/20241009_130424513094/oumi_job.sh"
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "~/oumi_launcher/20241009_130424513094",
        2,
        "myjob",
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_slurm_cluster_run_job_with_polling_succeeds(
    mock_time, mock_datetime, mock_slurm_client
):
    mock_time.sleep.side_effect = [None, None, None, None, None]
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_failed_cmd = Mock()
    mock_failed_cmd.exit_code = 1
    mock_slurm_client.run_commands.side_effect = [
        mock_failed_cmd,
        mock_successful_cmd,
        mock_successful_cmd,
        mock_successful_cmd,
    ]
    cluster = SlurmCluster("debug@host", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.side_effect = [
        [],
        [
            JobStatus(
                id="1",
                name="some name",
                status="RUNNING",
                metadata="",
                cluster="mycluster",
                done=False,
            )
        ],
        [
            JobStatus(
                id="1234",
                name="some name",
                status="RUNNING",
                metadata="",
                cluster="mycluster",
                done=False,
            )
        ],
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="RUNNING",
        metadata="",
        cluster="debug@host",
        done=False,
    )
    job_status = cluster.run_job(_get_default_job("slurm"))
    mock_slurm_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "~/oumi_launcher/20241009_130424513094",
            ),
            call(
                "~/local/path.bar",
                "~/home/remote/path.bar",
            ),
            call(
                "~/local/path2.txt",
                "~/home/remote/path2.txt",
            ),
        ],
    )
    mock_slurm_client.run_commands.assert_has_calls(
        [
            call(["chmod +x ~/oumi_launcher/20241009_130424513094/oumi_job.sh"]),
        ]
    )
    job_script = (
        "#!/bin/bash\n#SBATCH --gpus-per-task=8 \n#SBATCH --cpus-per-task=4\n\n"
        "export var1=val1\n\n"
        "pip install -r requirements.txt\n./hello_world.sh\n"
    )
    mock_slurm_client.put.assert_called_once_with(
        job_script, "~/oumi_launcher/20241009_130424513094/oumi_job.sh"
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "~/oumi_launcher/20241009_130424513094",
        2,
        "myjob",
    )
    mock_slurm_client.list_jobs.assert_has_calls([call(), call(), call()])
    mock_time.sleep.assert_has_calls([call(5), call(5)])
    assert job_status == expected_status


def test_slurm_cluster_run_job_no_name(mock_datetime, mock_slurm_client):
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_slurm_client.run_commands.return_value = mock_successful_cmd
    cluster = SlurmCluster("debug@host", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="RUNNING",
            metadata="",
            cluster="mycluster",
            done=False,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="RUNNING",
        metadata="",
        cluster="debug@host",
        done=False,
    )
    job = _get_default_job("slurm")
    job.name = None
    with patch("oumi.launcher.clusters.slurm_cluster.uuid") as mock_uuid:
        mock_hex = Mock()
        mock_hex.hex = "1-2-3"
        mock_uuid.uuid1.return_value = mock_hex
        job_status = cluster.run_job(job)
    mock_slurm_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "~/oumi_launcher/20241009_130424513094",
            ),
            call(
                "~/local/path.bar",
                "~/home/remote/path.bar",
            ),
            call(
                "~/local/path2.txt",
                "~/home/remote/path2.txt",
            ),
        ],
    )
    mock_slurm_client.run_commands.assert_has_calls(
        [
            call(["chmod +x ~/oumi_launcher/20241009_130424513094/oumi_job.sh"]),
        ]
    )
    job_script = (
        "#!/bin/bash\n#SBATCH --gpus-per-task=8 \n#SBATCH --cpus-per-task=4\n\n"
        "export var1=val1\n\n"
        "pip install -r requirements.txt\n./hello_world.sh\n"
    )
    mock_slurm_client.put.assert_called_once_with(
        job_script, "~/oumi_launcher/20241009_130424513094/oumi_job.sh"
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "~/oumi_launcher/20241009_130424513094",
        2,
        "1-2-3",
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_slurm_cluster_run_job_no_mounts(mock_datetime, mock_slurm_client):
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_slurm_client.run_commands.return_value = mock_successful_cmd
    cluster = SlurmCluster("debug@host", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="RUNNING",
            metadata="",
            cluster="mycluster",
            done=False,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="RUNNING",
        metadata="",
        cluster="debug@host",
        done=False,
    )
    job = _get_default_job("slurm")
    job.file_mounts = {}
    job_status = cluster.run_job(job)
    mock_slurm_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "~/oumi_launcher/20241009_130424513094",
            ),
        ],
    )
    mock_slurm_client.run_commands.assert_has_calls(
        [
            call(["chmod +x ~/oumi_launcher/20241009_130424513094/oumi_job.sh"]),
        ]
    )
    job_script = (
        "#!/bin/bash\n#SBATCH --gpus-per-task=8 \n#SBATCH --cpus-per-task=4\n\n"
        "export var1=val1\n\n"
        "pip install -r requirements.txt\n./hello_world.sh\n"
    )
    mock_slurm_client.put.assert_called_once_with(
        job_script, "~/oumi_launcher/20241009_130424513094/oumi_job.sh"
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "~/oumi_launcher/20241009_130424513094",
        2,
        "myjob",
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_slurm_cluster_run_job_no_pbs(mock_datetime, mock_slurm_client):
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_slurm_client.run_commands.return_value = mock_successful_cmd
    cluster = SlurmCluster("debug@host", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="RUNNING",
            metadata="",
            cluster="mycluster",
            done=False,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="RUNNING",
        metadata="",
        cluster="debug@host",
        done=False,
    )
    job = _get_default_job("slurm")
    job.file_mounts = {}
    job.setup = "small setup"
    job.run = "./hello_world.sh"
    job_status = cluster.run_job(job)
    mock_slurm_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "~/oumi_launcher/20241009_130424513094",
            ),
        ],
    )
    mock_slurm_client.run_commands.assert_has_calls(
        [
            call(["chmod +x ~/oumi_launcher/20241009_130424513094/oumi_job.sh"]),
        ]
    )
    job_script = "#!/bin/bash\n\nexport var1=val1\n\nsmall setup\n./hello_world.sh\n"
    mock_slurm_client.put.assert_called_once_with(
        job_script, "~/oumi_launcher/20241009_130424513094/oumi_job.sh"
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "~/oumi_launcher/20241009_130424513094",
        2,
        "myjob",
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_slurm_cluster_run_job_no_setup(mock_datetime, mock_slurm_client):
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_slurm_client.run_commands.return_value = mock_successful_cmd
    cluster = SlurmCluster("debug@host", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="RUNNING",
            metadata="",
            cluster="mycluster",
            done=False,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="RUNNING",
        metadata="",
        cluster="debug@host",
        done=False,
    )
    job = _get_default_job("slurm")
    job.file_mounts = {}
    job.setup = None
    job.run = "./hello_world.sh"
    job_status = cluster.run_job(job)
    mock_slurm_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "~/oumi_launcher/20241009_130424513094",
            ),
        ],
    )
    mock_slurm_client.run_commands.assert_has_calls(
        [
            call(["chmod +x ~/oumi_launcher/20241009_130424513094/oumi_job.sh"]),
        ]
    )
    job_script = "#!/bin/bash\n\nexport var1=val1\n\n./hello_world.sh\n"
    mock_slurm_client.put.assert_called_once_with(
        job_script, "~/oumi_launcher/20241009_130424513094/oumi_job.sh"
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "~/oumi_launcher/20241009_130424513094",
        2,
        "myjob",
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_slurm_cluster_run_job_fails(mock_time, mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug@host", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "234"
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="RUNNING",
            metadata="",
            cluster="mycluster",
            done=False,
        )
    ]
    with pytest.raises(RuntimeError):
        _ = cluster.run_job(_get_default_job("slurm"))
    mock_time.sleep.assert_has_calls([call(5), call(5), call(5)])


def test_slurm_cluster_down(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug-scaling@host", mock_slurm_client)
    cluster.down()
    # Nothing to assert, this method is a no-op.


def test_slurm_cluster_stop(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug-scaling@host", mock_slurm_client)
    cluster.stop()
    # Nothing to assert, this method is a no-op.
