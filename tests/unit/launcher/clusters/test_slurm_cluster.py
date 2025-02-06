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
def mock_datetime():
    with patch("oumi.launcher.clusters.slurm_cluster.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 10, 9, 13, 4, 24, 513094)
        yield mock_dt


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
def test_slurm_cluster_name(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("demand.einstein", mock_slurm_client)
    assert cluster.name() == "demand.einstein"

    cluster = SlurmCluster("debug.einstein", mock_slurm_client)
    assert cluster.name() == "debug.einstein"

    cluster = SlurmCluster("debug-scaling.einstein", mock_slurm_client)
    assert cluster.name() == "debug-scaling.einstein"

    cluster = SlurmCluster("preemptable.einstein", mock_slurm_client)
    assert cluster.name() == "preemptable.einstein"

    cluster = SlurmCluster("prod.einstein", mock_slurm_client)
    assert cluster.name() == "prod.einstein"


def test_slurm_cluster_invalid_name(mock_datetime, mock_slurm_client):
    with pytest.raises(ValueError):
        SlurmCluster("einstein", mock_slurm_client)


def test_slurm_cluster_invalid_queue(mock_datetime, mock_slurm_client):
    with pytest.raises(ValueError):
        SlurmCluster("albert.einstein", mock_slurm_client)


def test_slurm_cluster_get_job_valid_id(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug.name", mock_slurm_client)
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
    assert job.cluster == "debug.name"


def test_slurm_cluster_get_job_invalid_id_empty(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug.name", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = []
    job = cluster.get_job("myjob")
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job is None


def test_slurm_cluster_get_job_invalid_id_nonempty(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug.name", mock_slurm_client)
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
    cluster = SlurmCluster("debug.name", mock_slurm_client)
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


def test_slurm_cluster_get_jobs_empty(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug.name", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = []
    jobs = cluster.get_jobs()
    mock_slurm_client.list_jobs.assert_called_once_with()
    expected_jobs = []
    assert jobs == expected_jobs


def test_slurm_cluster_cancel_job(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("prod.name", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = [
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
    job_status = cluster.cancel_job("job2")
    expected_status = JobStatus(
        id="job2",
        name="some",
        status="running",
        metadata="",
        cluster="prod.name",
        done=False,
    )
    mock_slurm_client.cancel.assert_called_once_with(
        "job2",
    )
    assert job_status == expected_status


def test_slurm_cluster_cancel_job_fails(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("prod.name", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = [
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
        _ = cluster.cancel_job("myjobid")


def test_slurm_cluster_run_job(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug.name", mock_slurm_client)
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_slurm_client.run_commands.return_value = mock_successful_cmd
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
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
    job_status = cluster.run_job(_get_default_job("slurm"))
    mock_slurm_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "/home/user/oumi_launcher/20241009_130424513094",
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
            call(
                [
                    "cd /home/user/oumi_launcher/20241009_130424513094",
                    "module use /soft/modulefiles",
                    "module load conda",
                    "if [ ! -d /home/$USER/miniconda3/envs/oumi ]; then",
                    'echo "Creating Oumi Conda environment... '
                    '---------------------------"',
                    "conda create -y python=3.11 --prefix "
                    "/home/$USER/miniconda3/envs/oumi",
                    "fi",
                    'echo "Installing packages... '
                    '---------------------------------------"',
                    "conda activate /home/$USER/miniconda3/envs/oumi",
                    "if ! command -v uv >/dev/null 2>&1; then",
                    "pip install -U uv",
                    "fi",
                    "pip install -e '.[gpu]' vllm",
                ]
            ),
            call(
                ["chmod +x /home/user/oumi_launcher/20241009_130424513094/oumi_job.sh"]
            ),
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
    mock_slurm_client.put.assert_called_once_with(
        job_script, "/home/user/oumi_launcher/20241009_130424513094/oumi_job.sh"
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "/home/user/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "/home/user/oumi_launcher/20241009_130424513094",
        2,
        "myjob",
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_slurm_cluster_run_job_with_conda_setup(mock_datetime, mock_slurm_client):
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
    cluster = SlurmCluster("debug.name", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
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
    job_status = cluster.run_job(_get_default_job("slurm"))
    mock_slurm_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "/home/user/oumi_launcher/20241009_130424513094",
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
            call(
                [
                    "cd /home/user/oumi_launcher/20241009_130424513094",
                    "module use /soft/modulefiles",
                    "module load conda",
                    "if [ ! -d /home/$USER/miniconda3/envs/oumi ]; then",
                    'echo "Creating Oumi Conda environment... '
                    '---------------------------"',
                    "conda create -y python=3.11 --prefix "
                    "/home/$USER/miniconda3/envs/oumi",
                    "fi",
                    'echo "Installing packages... '
                    '---------------------------------------"',
                    "conda activate /home/$USER/miniconda3/envs/oumi",
                    "if ! command -v uv >/dev/null 2>&1; then",
                    "pip install -U uv",
                    "fi",
                    "pip install -e '.[gpu]' vllm",
                ]
            ),
            call(
                ["chmod +x /home/user/oumi_launcher/20241009_130424513094/oumi_job.sh"]
            ),
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
    mock_slurm_client.put.assert_called_once_with(
        job_script, "/home/user/oumi_launcher/20241009_130424513094/oumi_job.sh"
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "/home/user/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "/home/user/oumi_launcher/20241009_130424513094",
        2,
        "myjob",
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_slurm_cluster_run_job_no_name(mock_datetime, mock_slurm_client):
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_slurm_client.run_commands.return_value = mock_successful_cmd
    cluster = SlurmCluster("debug.name", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
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
                "/home/user/oumi_launcher/20241009_130424513094",
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
            call(
                [
                    "cd /home/user/oumi_launcher/20241009_130424513094",
                    "module use /soft/modulefiles",
                    "module load conda",
                    "if [ ! -d /home/$USER/miniconda3/envs/oumi ]; then",
                    'echo "Creating Oumi Conda environment... '
                    '---------------------------"',
                    "conda create -y python=3.11 --prefix "
                    "/home/$USER/miniconda3/envs/oumi",
                    "fi",
                    'echo "Installing packages... '
                    '---------------------------------------"',
                    "conda activate /home/$USER/miniconda3/envs/oumi",
                    "if ! command -v uv >/dev/null 2>&1; then",
                    "pip install -U uv",
                    "fi",
                    "pip install -e '.[gpu]' vllm",
                ]
            ),
            call(
                ["chmod +x /home/user/oumi_launcher/20241009_130424513094/oumi_job.sh"]
            ),
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
    mock_slurm_client.put.assert_called_once_with(
        job_script, "/home/user/oumi_launcher/20241009_130424513094/oumi_job.sh"
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "/home/user/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "/home/user/oumi_launcher/20241009_130424513094",
        2,
        "1-2-3",
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_slurm_cluster_run_job_no_mounts(mock_datetime, mock_slurm_client):
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_slurm_client.run_commands.return_value = mock_successful_cmd
    cluster = SlurmCluster("debug.name", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
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
    job = _get_default_job("slurm")
    job.file_mounts = {}
    job_status = cluster.run_job(job)
    mock_slurm_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "/home/user/oumi_launcher/20241009_130424513094",
            ),
        ],
    )
    mock_slurm_client.run_commands.assert_has_calls(
        [
            call(
                [
                    "cd /home/user/oumi_launcher/20241009_130424513094",
                    "module use /soft/modulefiles",
                    "module load conda",
                    "if [ ! -d /home/$USER/miniconda3/envs/oumi ]; then",
                    'echo "Creating Oumi Conda environment... '
                    '---------------------------"',
                    "conda create -y python=3.11 --prefix "
                    "/home/$USER/miniconda3/envs/oumi",
                    "fi",
                    'echo "Installing packages... '
                    '---------------------------------------"',
                    "conda activate /home/$USER/miniconda3/envs/oumi",
                    "if ! command -v uv >/dev/null 2>&1; then",
                    "pip install -U uv",
                    "fi",
                    "pip install -e '.[gpu]' vllm",
                ]
            ),
            call(
                ["chmod +x /home/user/oumi_launcher/20241009_130424513094/oumi_job.sh"]
            ),
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
    mock_slurm_client.put.assert_called_once_with(
        job_script, "/home/user/oumi_launcher/20241009_130424513094/oumi_job.sh"
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "/home/user/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "/home/user/oumi_launcher/20241009_130424513094",
        2,
        "myjob",
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_slurm_cluster_run_job_no_pbs(mock_datetime, mock_slurm_client):
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_slurm_client.run_commands.return_value = mock_successful_cmd
    cluster = SlurmCluster("debug.name", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
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
    job = _get_default_job("slurm")
    job.file_mounts = {}
    job.setup = "small setup"
    job.run = "./hello_world.sh"
    job_status = cluster.run_job(job)
    mock_slurm_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "/home/user/oumi_launcher/20241009_130424513094",
            ),
        ],
    )
    mock_slurm_client.run_commands.assert_has_calls(
        [
            call(
                [
                    "cd /home/user/oumi_launcher/20241009_130424513094",
                    "module use /soft/modulefiles",
                    "module load conda",
                    "if [ ! -d /home/$USER/miniconda3/envs/oumi ]; then",
                    'echo "Creating Oumi Conda environment... '
                    '---------------------------"',
                    "conda create -y python=3.11 --prefix "
                    "/home/$USER/miniconda3/envs/oumi",
                    "fi",
                    'echo "Installing packages... '
                    '---------------------------------------"',
                    "conda activate /home/$USER/miniconda3/envs/oumi",
                    "if ! command -v uv >/dev/null 2>&1; then",
                    "pip install -U uv",
                    "fi",
                    "pip install -e '.[gpu]' vllm",
                ]
            ),
            call(
                ["chmod +x /home/user/oumi_launcher/20241009_130424513094/oumi_job.sh"]
            ),
        ]
    )
    job_script = (
        "#!/bin/bash\n\n" "export var1=val1\n\n" "small setup\n./hello_world.sh\n"
    )
    mock_slurm_client.put.assert_called_once_with(
        job_script, "/home/user/oumi_launcher/20241009_130424513094/oumi_job.sh"
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "/home/user/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "/home/user/oumi_launcher/20241009_130424513094",
        2,
        "myjob",
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_slurm_cluster_run_job_no_setup(mock_datetime, mock_slurm_client):
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_slurm_client.run_commands.return_value = mock_successful_cmd
    cluster = SlurmCluster("debug.name", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
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
    job = _get_default_job("slurm")
    job.file_mounts = {}
    job.setup = None
    job.run = "./hello_world.sh"
    job_status = cluster.run_job(job)
    mock_slurm_client.put_recursive.assert_has_calls(
        [
            call(
                "./",
                "/home/user/oumi_launcher/20241009_130424513094",
            ),
        ],
    )
    mock_slurm_client.run_commands.assert_has_calls(
        [
            call(
                [
                    "cd /home/user/oumi_launcher/20241009_130424513094",
                    "module use /soft/modulefiles",
                    "module load conda",
                    "if [ ! -d /home/$USER/miniconda3/envs/oumi ]; then",
                    'echo "Creating Oumi Conda environment... '
                    '---------------------------"',
                    "conda create -y python=3.11 --prefix "
                    "/home/$USER/miniconda3/envs/oumi",
                    "fi",
                    'echo "Installing packages... '
                    '---------------------------------------"',
                    "conda activate /home/$USER/miniconda3/envs/oumi",
                    "if ! command -v uv >/dev/null 2>&1; then",
                    "pip install -U uv",
                    "fi",
                    "pip install -e '.[gpu]' vllm",
                ]
            ),
            call(
                ["chmod +x /home/user/oumi_launcher/20241009_130424513094/oumi_job.sh"]
            ),
        ]
    )
    job_script = "#!/bin/bash\n\n" "export var1=val1\n\n" "./hello_world.sh\n"
    mock_slurm_client.put.assert_called_once_with(
        job_script, "/home/user/oumi_launcher/20241009_130424513094/oumi_job.sh"
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "/home/user/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "/home/user/oumi_launcher/20241009_130424513094",
        2,
        "myjob",
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_slurm_cluster_run_job_fails(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug.name", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "234"
    mock_slurm_client.list_jobs.return_value = [
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
        _ = cluster.run_job(_get_default_job("slurm"))


def test_slurm_cluster_down(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug-scaling.name", mock_slurm_client)
    cluster.down()
    # Nothing to assert, this method is a no-op.


def test_slurm_cluster_stop(mock_datetime, mock_slurm_client):
    cluster = SlurmCluster("debug-scaling.name", mock_slurm_client)
    cluster.stop()
    # Nothing to assert, this method is a no-op.
