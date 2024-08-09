from unittest.mock import Mock, call, patch

import pytest

from lema.core.types.base_cluster import JobStatus
from lema.core.types.configs import JobConfig
from lema.core.types.params.job_resources import JobResources, StorageMount
from lema.launcher.clients.polaris_client import PolarisClient
from lema.launcher.clusters.polaris_cluster import PolarisCluster


#
# Fixtures
#
@pytest.fixture
def mock_polaris_client():
    yield Mock(spec=PolarisClient)


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
            "#PBS -o some/log \n#PBE -l wow\n#PBS -e run/log\n"
            "pip install -r requirements.txt"
        ),
        run="./hello_world.sh",
    )


#
# Tests
#
def test_polaris_cluster_name(mock_polaris_client):
    cluster = PolarisCluster("demand.einstein", mock_polaris_client)
    assert cluster.name() == "demand.einstein"

    cluster = PolarisCluster("debug.einstein", mock_polaris_client)
    assert cluster.name() == "debug.einstein"

    cluster = PolarisCluster("debug-scaling.einstein", mock_polaris_client)
    assert cluster.name() == "debug-scaling.einstein"

    cluster = PolarisCluster("preemptable.einstein", mock_polaris_client)
    assert cluster.name() == "preemptable.einstein"

    cluster = PolarisCluster("prod.einstein", mock_polaris_client)
    assert cluster.name() == "prod.einstein"


def test_polaris_cluster_invalid_name(mock_polaris_client):
    with pytest.raises(ValueError):
        PolarisCluster("einstein", mock_polaris_client)


def test_polaris_cluster_invalid_queue(mock_polaris_client):
    with pytest.raises(ValueError):
        PolarisCluster("albert.einstein", mock_polaris_client)


def test_polaris_cluster_get_job_valid_id(mock_polaris_client):
    cluster = PolarisCluster("debug.name", mock_polaris_client)
    mock_polaris_client.list_jobs.return_value = [
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
    mock_polaris_client.list_jobs.assert_called_once_with(
        PolarisClient.SupportedQueues.DEBUG
    )
    assert job is not None
    assert job.id == "myjob"
    assert job.cluster == "debug.name"


def test_polaris_cluster_get_job_invalid_id_empty(mock_polaris_client):
    cluster = PolarisCluster("debug.name", mock_polaris_client)
    mock_polaris_client.list_jobs.return_value = []
    job = cluster.get_job("myjob")
    mock_polaris_client.list_jobs.assert_called_once_with(
        PolarisClient.SupportedQueues.DEBUG
    )
    assert job is None


def test_polaris_cluster_get_job_invalid_id_nonempty(mock_polaris_client):
    cluster = PolarisCluster("debug.name", mock_polaris_client)
    mock_polaris_client.list_jobs.return_value = [
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
    mock_polaris_client.list_jobs.assert_called_once_with(
        PolarisClient.SupportedQueues.DEBUG
    )
    assert job is None


def test_polaris_cluster_get_jobs_nonempty(mock_polaris_client):
    cluster = PolarisCluster("debug.name", mock_polaris_client)
    mock_polaris_client.list_jobs.return_value = [
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
    mock_polaris_client.list_jobs.assert_called_once_with(
        PolarisClient.SupportedQueues.DEBUG
    )
    expected_jobs = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="debug.name",
            done=False,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="debug.name",
            done=False,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="debug.name",
            done=False,
        ),
    ]
    assert jobs == expected_jobs


def test_polaris_cluster_get_jobs_empty(mock_polaris_client):
    cluster = PolarisCluster("debug.name", mock_polaris_client)
    mock_polaris_client.list_jobs.return_value = []
    jobs = cluster.get_jobs()
    mock_polaris_client.list_jobs.assert_called_once_with(
        PolarisClient.SupportedQueues.DEBUG
    )
    expected_jobs = []
    assert jobs == expected_jobs


def test_polaris_cluster_stop_job(mock_polaris_client):
    cluster = PolarisCluster("prod.name", mock_polaris_client)
    mock_polaris_client.list_jobs.return_value = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="debug.name",
            done=False,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="debug.name",
            done=False,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="debug.name",
            done=False,
        ),
    ]
    job_status = cluster.stop_job("job2")
    expected_status = JobStatus(
        id="job2",
        name="some",
        status="running",
        metadata="",
        cluster="prod.name",
        done=False,
    )
    mock_polaris_client.cancel.assert_called_once_with(
        "job2",
        PolarisClient.SupportedQueues.PROD,
    )
    assert job_status == expected_status


def test_polaris_cluster_stop_job_fails(mock_polaris_client):
    cluster = PolarisCluster("prod.name", mock_polaris_client)
    mock_polaris_client.list_jobs.return_value = [
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="debug.name",
            done=False,
        ),
    ]
    with pytest.raises(RuntimeError):
        _ = cluster.stop_job("myjobid")


def test_polaris_cluster_run_job(mock_polaris_client):
    cluster = PolarisCluster("debug.name", mock_polaris_client)
    mock_polaris_client.submit_job.return_value = "1234"
    mock_polaris_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="queued",
            metadata="",
            cluster="mycluster",
            done=False,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="debug.name",
        done=False,
    )
    job_status = cluster.run_job(_get_default_job("polaris"))
    mock_polaris_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "/home/user/lema_launcher/myjob",
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
    mock_polaris_client.run_commands.assert_has_calls(
        [
            call(["mkdir -p /home/user/lema_launcher/myjob"]),
            call(["test -d /home/$USER/miniconda3/envs/lema"]),
            call(["chmod a+x /home/user/lema_launcher/myjob/lema_job.sh"]),
            call(
                [
                    "mkdir -p some/log",
                    "mkdir -p run/log",
                ]
            ),
        ]
    )
    job_script = (
        "#!/bin/bash\n#PBS -o some/log \n#PBE -l wow\n#PBS -e run/log\n\n"
        "export var1=val1\n\n"
        "pip install -r requirements.txt\n./hello_world.sh\n"
    )
    mock_polaris_client.put.assert_called_once_with(
        job_script, "/home/user/lema_launcher/myjob/lema_job.sh"
    )
    mock_polaris_client.submit_job.assert_called_once_with(
        "/home/user/lema_launcher/myjob/lema_job.sh",
        "/home/user/lema_launcher/myjob",
        2,
        PolarisClient.SupportedQueues.DEBUG,
        "myjob",
    )
    mock_polaris_client.list_jobs.assert_called_once_with(
        PolarisClient.SupportedQueues.DEBUG
    )
    assert job_status == expected_status


def test_polaris_cluster_run_job_with_conda_setup(mock_polaris_client):
    cluster = PolarisCluster("debug.name", mock_polaris_client)
    mock_polaris_client.submit_job.return_value = "1234"
    mock_polaris_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="queued",
            metadata="",
            cluster="mycluster",
            done=False,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="debug.name",
        done=False,
    )
    mock_polaris_client.run_commands.side_effect = [
        None,
        RuntimeError,
        None,
        None,
        None,
        None,
        None,
    ]
    job_status = cluster.run_job(_get_default_job("polaris"))
    mock_polaris_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "/home/user/lema_launcher/myjob",
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
    mock_polaris_client.run_commands.assert_has_calls(
        [
            call(["mkdir -p /home/user/lema_launcher/myjob"]),
            call(["test -d /home/$USER/miniconda3/envs/lema"]),
            call(
                [
                    "module use /soft/modulefiles && "
                    "module load conda && "
                    'echo "Creating LeMa Conda environment... -------------------------'
                    '--"'
                    " && conda create -y python=3.11 --prefix "
                    "/home/$USER/miniconda3/envs/lema"
                    " && conda activate /home/$USER/miniconda3/envs/lema && "
                    "pip install flash-attn --no-build-isolation"
                ]
            ),
            call(
                [
                    "cd /home/user/lema_launcher/myjob && "
                    "module use /soft/modulefiles && "
                    "module load conda && "
                    "conda activate /home/$USER/miniconda3/envs/lema && "
                    'echo "Installing packages... -------------------------------------'
                    '--"'
                    " && pip install -e '.[train]'"
                ]
            ),
            call(["chmod a+x /home/user/lema_launcher/myjob/lema_job.sh"]),
            call(
                [
                    "mkdir -p some/log",
                    "mkdir -p run/log",
                ]
            ),
        ]
    )
    job_script = (
        "#!/bin/bash\n#PBS -o some/log \n#PBE -l wow\n#PBS -e run/log\n\n"
        "export var1=val1\n\n"
        "pip install -r requirements.txt\n./hello_world.sh\n"
    )
    mock_polaris_client.put.assert_called_once_with(
        job_script, "/home/user/lema_launcher/myjob/lema_job.sh"
    )
    mock_polaris_client.submit_job.assert_called_once_with(
        "/home/user/lema_launcher/myjob/lema_job.sh",
        "/home/user/lema_launcher/myjob",
        2,
        PolarisClient.SupportedQueues.DEBUG,
        "myjob",
    )
    mock_polaris_client.list_jobs.assert_called_once_with(
        PolarisClient.SupportedQueues.DEBUG
    )
    assert job_status == expected_status


def test_polaris_cluster_run_job_no_name(mock_polaris_client):
    cluster = PolarisCluster("debug.name", mock_polaris_client)
    mock_polaris_client.submit_job.return_value = "1234"
    mock_polaris_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="queued",
            metadata="",
            cluster="mycluster",
            done=False,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="debug.name",
        done=False,
    )
    job = _get_default_job("polaris")
    job.name = None
    with patch("lema.launcher.clusters.polaris_cluster.uuid") as mock_uuid:
        mock_hex = Mock()
        mock_hex.hex = "1-2-3"
        mock_uuid.uuid1.return_value = mock_hex
        job_status = cluster.run_job(job)
    mock_polaris_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "/home/user/lema_launcher/1-2-3",
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
    mock_polaris_client.run_commands.assert_has_calls(
        [
            call(["mkdir -p /home/user/lema_launcher/1-2-3"]),
            call(["test -d /home/$USER/miniconda3/envs/lema"]),
            call(["chmod a+x /home/user/lema_launcher/1-2-3/lema_job.sh"]),
            call(
                [
                    "mkdir -p some/log",
                    "mkdir -p run/log",
                ]
            ),
        ]
    )
    job_script = (
        "#!/bin/bash\n#PBS -o some/log \n#PBE -l wow\n#PBS -e run/log\n\n"
        "export var1=val1\n\n"
        "pip install -r requirements.txt\n./hello_world.sh\n"
    )
    mock_polaris_client.put.assert_called_once_with(
        job_script, "/home/user/lema_launcher/1-2-3/lema_job.sh"
    )
    mock_polaris_client.submit_job.assert_called_once_with(
        "/home/user/lema_launcher/1-2-3/lema_job.sh",
        "/home/user/lema_launcher/1-2-3",
        2,
        PolarisClient.SupportedQueues.DEBUG,
        "1-2-3",
    )
    mock_polaris_client.list_jobs.assert_called_once_with(
        PolarisClient.SupportedQueues.DEBUG
    )
    assert job_status == expected_status


def test_polaris_cluster_run_job_no_mounts(mock_polaris_client):
    cluster = PolarisCluster("debug.name", mock_polaris_client)
    mock_polaris_client.submit_job.return_value = "1234"
    mock_polaris_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="queued",
            metadata="",
            cluster="mycluster",
            done=False,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="debug.name",
        done=False,
    )
    job = _get_default_job("polaris")
    job.file_mounts = {}
    job_status = cluster.run_job(job)
    mock_polaris_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "/home/user/lema_launcher/myjob",
            ),
        ],
    )
    mock_polaris_client.run_commands.assert_has_calls(
        [
            call(["mkdir -p /home/user/lema_launcher/myjob"]),
            call(["test -d /home/$USER/miniconda3/envs/lema"]),
            call(["chmod a+x /home/user/lema_launcher/myjob/lema_job.sh"]),
            call(
                [
                    "mkdir -p some/log",
                    "mkdir -p run/log",
                ]
            ),
        ]
    )
    job_script = (
        "#!/bin/bash\n#PBS -o some/log \n#PBE -l wow\n#PBS -e run/log\n\n"
        "export var1=val1\n\n"
        "pip install -r requirements.txt\n./hello_world.sh\n"
    )
    mock_polaris_client.put.assert_called_once_with(
        job_script, "/home/user/lema_launcher/myjob/lema_job.sh"
    )
    mock_polaris_client.submit_job.assert_called_once_with(
        "/home/user/lema_launcher/myjob/lema_job.sh",
        "/home/user/lema_launcher/myjob",
        2,
        PolarisClient.SupportedQueues.DEBUG,
        "myjob",
    )
    mock_polaris_client.list_jobs.assert_called_once_with(
        PolarisClient.SupportedQueues.DEBUG
    )
    assert job_status == expected_status


def test_polaris_cluster_run_job_no_pbs(mock_polaris_client):
    cluster = PolarisCluster("debug.name", mock_polaris_client)
    mock_polaris_client.submit_job.return_value = "1234"
    mock_polaris_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="queued",
            metadata="",
            cluster="mycluster",
            done=False,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="debug.name",
        done=False,
    )
    job = _get_default_job("polaris")
    job.file_mounts = {}
    job.setup = "small setup"
    job.run = "./hello_world.sh"
    job_status = cluster.run_job(job)
    mock_polaris_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "/home/user/lema_launcher/myjob",
            ),
        ],
    )
    mock_polaris_client.run_commands.assert_has_calls(
        [
            call(["mkdir -p /home/user/lema_launcher/myjob"]),
            call(["test -d /home/$USER/miniconda3/envs/lema"]),
            call(["chmod a+x /home/user/lema_launcher/myjob/lema_job.sh"]),
        ]
    )
    job_script = (
        "#!/bin/bash\n\n" "export var1=val1\n\n" "small setup\n./hello_world.sh\n"
    )
    mock_polaris_client.put.assert_called_once_with(
        job_script, "/home/user/lema_launcher/myjob/lema_job.sh"
    )
    mock_polaris_client.submit_job.assert_called_once_with(
        "/home/user/lema_launcher/myjob/lema_job.sh",
        "/home/user/lema_launcher/myjob",
        2,
        PolarisClient.SupportedQueues.DEBUG,
        "myjob",
    )
    mock_polaris_client.list_jobs.assert_called_once_with(
        PolarisClient.SupportedQueues.DEBUG
    )
    assert job_status == expected_status


def test_polaris_cluster_run_job_no_setup(mock_polaris_client):
    cluster = PolarisCluster("debug.name", mock_polaris_client)
    mock_polaris_client.submit_job.return_value = "1234"
    mock_polaris_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="queued",
            metadata="",
            cluster="mycluster",
            done=False,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="debug.name",
        done=False,
    )
    job = _get_default_job("polaris")
    job.file_mounts = {}
    job.setup = None
    job.run = "./hello_world.sh"
    job_status = cluster.run_job(job)
    mock_polaris_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "/home/user/lema_launcher/myjob",
            ),
        ],
    )
    mock_polaris_client.run_commands.assert_has_calls(
        [
            call(["mkdir -p /home/user/lema_launcher/myjob"]),
            call(["test -d /home/$USER/miniconda3/envs/lema"]),
            call(["chmod a+x /home/user/lema_launcher/myjob/lema_job.sh"]),
        ]
    )
    job_script = "#!/bin/bash\n\n" "export var1=val1\n\n" "./hello_world.sh\n"
    mock_polaris_client.put.assert_called_once_with(
        job_script, "/home/user/lema_launcher/myjob/lema_job.sh"
    )
    mock_polaris_client.submit_job.assert_called_once_with(
        "/home/user/lema_launcher/myjob/lema_job.sh",
        "/home/user/lema_launcher/myjob",
        2,
        PolarisClient.SupportedQueues.DEBUG,
        "myjob",
    )
    mock_polaris_client.list_jobs.assert_called_once_with(
        PolarisClient.SupportedQueues.DEBUG
    )
    assert job_status == expected_status


def test_polaris_cluster_run_job_fails(mock_polaris_client):
    cluster = PolarisCluster("debug.name", mock_polaris_client)
    mock_polaris_client.submit_job.return_value = "234"
    mock_polaris_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="queued",
            metadata="",
            cluster="mycluster",
            done=False,
        )
    ]
    with pytest.raises(RuntimeError):
        _ = cluster.run_job(_get_default_job("polaris"))


def test_polaris_cluster_down(mock_polaris_client):
    cluster = PolarisCluster("debug-scaling.name", mock_polaris_client)
    cluster.down()
    # Nothing to assert, this method is a no-op.
