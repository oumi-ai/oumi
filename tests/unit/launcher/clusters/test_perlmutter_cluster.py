from datetime import datetime
from typing import Final
from unittest.mock import Mock, call, patch

import pytest

from oumi.core.configs import JobConfig, JobResources
from oumi.core.launcher import JobState, JobStatus
from oumi.launcher.clients.slurm_client import SlurmClient
from oumi.launcher.clusters.perlmutter_cluster import PerlmutterCluster


#
# Fixtures
#
@pytest.fixture
def mock_slurm_client():
    yield Mock(spec=SlurmClient)


@pytest.fixture
def mock_datetime():
    with patch("oumi.launcher.clusters.perlmutter_cluster.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 10, 9, 13, 4, 24, 513094)
        yield mock_dt


def _get_default_job() -> JobConfig:
    return JobConfig(
        name="myjob",
        user="user",
        working_dir="./",
        num_nodes=2,
        resources=JobResources(cloud="perlmutter"),
        envs={"var1": "val1"},
        file_mounts={
            "~/home/remote/path.bar": "~/local/path.bar",
            "~/home/remote/path2.txt": "~/local/path2.txt",
        },
        setup=(
            "#SBATCH -o some/log \n#SBATCH -l wow\n#SBATCH -e run/log\n"
            "pip install -r requirements.txt"
        ),
        run="./hello_world.sh",
    )


_COMMON_INIT_COMMANDS: Final[list[str]] = [
    "module load conda",
    'if [ ! -z "$CONDA_DEFAULT_ENV" ]; then',
    "conda deactivate",
    "fi",
    'echo "Installing packages... ---------------------------------------"',
    "conda activate oumi",
    "if ! command -v uv >/dev/null 2>&1; then",
    "pip install -U uv",
    "fi",
    "cd ~/oumi_launcher/20241009_130424513094",
    "uv pip install -e '.[gpu]' 'huggingface_hub[cli]' hf_transfer",
]


#
# Tests
#
def test_perlmutter_cluster_name(mock_datetime, mock_slurm_client):
    cluster = PerlmutterCluster("regular.saul", mock_slurm_client)
    assert cluster.name() == "regular.saul"

    cluster = PerlmutterCluster("debug.saul", mock_slurm_client)
    assert cluster.name() == "debug.saul"


def test_perlmutter_cluster_invalid_name(mock_datetime, mock_slurm_client):
    with pytest.raises(ValueError):
        PerlmutterCluster("saul", mock_slurm_client)


def test_perlmutter_cluster_invalid_queue(mock_datetime, mock_slurm_client):
    with pytest.raises(ValueError):
        PerlmutterCluster("fakequeue.saul", mock_slurm_client)


def test_perlmutter_cluster_get_job_valid_id(mock_datetime, mock_slurm_client):
    cluster = PerlmutterCluster("regular.name", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
        ),
    ]
    job = cluster.get_job("myjob")
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job is not None
    assert job.id == "myjob"
    assert job.cluster == "regular.name"


def test_perlmutter_cluster_get_job_invalid_id_empty(mock_datetime, mock_slurm_client):
    cluster = PerlmutterCluster("regular.name", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = []
    job = cluster.get_job("myjob")
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job is None


def test_perlmutter_cluster_get_job_invalid_id_nonempty(
    mock_datetime, mock_slurm_client
):
    cluster = PerlmutterCluster("regular.name", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
        ),
    ]
    job = cluster.get_job("wrong job")
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job is None


def test_perlmutter_cluster_get_jobs_nonempty(mock_datetime, mock_slurm_client):
    cluster = PerlmutterCluster("regular.name", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
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
            cluster="regular.name",
            done=False,
            state=JobState.PENDING,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="regular.name",
            done=False,
            state=JobState.PENDING,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="regular.name",
            done=False,
            state=JobState.PENDING,
        ),
    ]
    assert jobs == expected_jobs


def test_perlmutter_cluster_get_jobs_empty(mock_datetime, mock_slurm_client):
    cluster = PerlmutterCluster("regular.name", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = []
    jobs = cluster.get_jobs()
    mock_slurm_client.list_jobs.assert_called_once_with()
    expected_jobs = []
    assert jobs == expected_jobs


def test_perlmutter_cluster_cancel_job(mock_datetime, mock_slurm_client):
    cluster = PerlmutterCluster("debug.name", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="regular.name",
            done=False,
            state=JobState.PENDING,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="regular.name",
            done=False,
            state=JobState.PENDING,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="regular.name",
            done=False,
            state=JobState.PENDING,
        ),
    ]
    job_status = cluster.cancel_job("job2")
    expected_status = JobStatus(
        id="job2",
        name="some",
        status="running",
        metadata="",
        cluster="debug.name",
        done=False,
        state=JobState.PENDING,
    )
    mock_slurm_client.cancel.assert_called_once_with("job2")
    assert job_status == expected_status


def test_perlmutter_cluster_cancel_job_fails(mock_datetime, mock_slurm_client):
    cluster = PerlmutterCluster("debug.name", mock_slurm_client)
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="regular.name",
            done=False,
            state=JobState.PENDING,
        ),
    ]
    with pytest.raises(RuntimeError):
        _ = cluster.cancel_job("myjobid")


def test_perlmutter_cluster_run_job(mock_datetime, mock_slurm_client):
    cluster = PerlmutterCluster("regular.name", mock_slurm_client)
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
            state=JobState.PENDING,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="regular.name",
        done=False,
        state=JobState.PENDING,
    )
    job_status = cluster.run_job(_get_default_job())
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
            call(_COMMON_INIT_COMMANDS),
            call([("chmod +x ~/oumi_launcher/20241009_130424513094/oumi_job.sh")]),
            call(["mkdir -p run/log", "mkdir -p some/log"]),
        ]
    )
    job_script = (
        "#!/bin/bash\n#SBATCH -o some/log \n#SBATCH -l wow\n#SBATCH -e run/log\n\n"
        "export var1=val1\n\n"
        "pip install -r requirements.txt\n./hello_world.sh\n"
    )
    mock_slurm_client.put.assert_called_once_with(
        job_script,
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "~/oumi_launcher/20241009_130424513094",
        node_count=2,
        name="myjob",
        ntasks=2,
        threads_per_core=1,
        qos="regular",
        constraint="gpu",
        gpus_per_node=4,
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_perlmutter_cluster_run_job_with_conda_setup(mock_datetime, mock_slurm_client):
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
    cluster = PerlmutterCluster("regular.name", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="queued",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="regular.name",
        done=False,
        state=JobState.PENDING,
    )
    job_status = cluster.run_job(_get_default_job())
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
            call(_COMMON_INIT_COMMANDS),
            call(["chmod +x ~/oumi_launcher/20241009_130424513094/oumi_job.sh"]),
            call(["mkdir -p run/log", "mkdir -p some/log"]),
        ]
    )
    job_script = (
        "#!/bin/bash\n#SBATCH -o some/log \n#SBATCH -l wow\n#SBATCH -e run/log\n\n"
        "export var1=val1\n\n"
        "pip install -r requirements.txt\n./hello_world.sh\n"
    )
    mock_slurm_client.put.assert_called_once_with(
        job_script,
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "~/oumi_launcher/20241009_130424513094",
        node_count=2,
        name="myjob",
        ntasks=2,
        threads_per_core=1,
        qos="regular",
        constraint="gpu",
        gpus_per_node=4,
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_perlmutter_cluster_run_job_no_name(mock_datetime, mock_slurm_client):
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_slurm_client.run_commands.return_value = mock_successful_cmd
    cluster = PerlmutterCluster("regular.name", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="queued",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="regular.name",
        done=False,
        state=JobState.PENDING,
    )
    job = _get_default_job()
    job.name = None
    with patch("oumi.launcher.clusters.perlmutter_cluster.uuid") as mock_uuid:
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
            call(_COMMON_INIT_COMMANDS),
            call(["chmod +x ~/oumi_launcher/20241009_130424513094/oumi_job.sh"]),
            call(
                [
                    "mkdir -p run/log",
                    "mkdir -p some/log",
                ]
            ),
        ]
    )
    job_script = (
        "#!/bin/bash\n#SBATCH -o some/log \n#SBATCH -l wow\n#SBATCH -e run/log\n\n"
        "export var1=val1\n\n"
        "pip install -r requirements.txt\n./hello_world.sh\n"
    )
    mock_slurm_client.put.assert_called_once_with(
        job_script,
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "~/oumi_launcher/20241009_130424513094",
        node_count=2,
        name="1-2-3",
        ntasks=2,
        threads_per_core=1,
        qos="regular",
        constraint="gpu",
        gpus_per_node=4,
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_perlmutter_cluster_run_job_no_mounts(mock_datetime, mock_slurm_client):
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_slurm_client.run_commands.return_value = mock_successful_cmd
    cluster = PerlmutterCluster("regular.name", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="queued",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="regular.name",
        done=False,
        state=JobState.PENDING,
    )
    job = _get_default_job()
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
            call(_COMMON_INIT_COMMANDS),
            call(["chmod +x ~/oumi_launcher/20241009_130424513094/oumi_job.sh"]),
            call(
                [
                    "mkdir -p run/log",
                    "mkdir -p some/log",
                ]
            ),
        ]
    )
    job_script = (
        "#!/bin/bash\n#SBATCH -o some/log \n#SBATCH -l wow\n#SBATCH -e run/log\n\n"
        "export var1=val1\n\n"
        "pip install -r requirements.txt\n./hello_world.sh\n"
    )
    mock_slurm_client.put.assert_called_once_with(
        job_script,
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "~/oumi_launcher/20241009_130424513094",
        node_count=2,
        name="myjob",
        ntasks=2,
        threads_per_core=1,
        qos="regular",
        constraint="gpu",
        gpus_per_node=4,
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_perlmutter_cluster_run_job_no_sbatch(mock_datetime, mock_slurm_client):
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_slurm_client.run_commands.return_value = mock_successful_cmd
    cluster = PerlmutterCluster("regular.name", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="queued",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="regular.name",
        done=False,
        state=JobState.PENDING,
    )
    job = _get_default_job()
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
            call(_COMMON_INIT_COMMANDS),
            call(["chmod +x ~/oumi_launcher/20241009_130424513094/oumi_job.sh"]),
        ]
    )
    job_script = "#!/bin/bash\n\nexport var1=val1\n\nsmall setup\n./hello_world.sh\n"
    mock_slurm_client.put.assert_called_once_with(
        job_script,
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "~/oumi_launcher/20241009_130424513094",
        node_count=2,
        name="myjob",
        ntasks=2,
        threads_per_core=1,
        qos="regular",
        constraint="gpu",
        gpus_per_node=4,
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_perlmutter_cluster_run_job_no_setup(mock_datetime, mock_slurm_client):
    mock_successful_cmd = Mock()
    mock_successful_cmd.exit_code = 0
    mock_slurm_client.run_commands.return_value = mock_successful_cmd
    cluster = PerlmutterCluster("regular.name", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "1234"
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="queued",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
        )
    ]
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="regular.name",
        done=False,
        state=JobState.PENDING,
    )
    job = _get_default_job()
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
            call(_COMMON_INIT_COMMANDS),
            call(["chmod +x ~/oumi_launcher/20241009_130424513094/oumi_job.sh"]),
        ]
    )
    job_script = "#!/bin/bash\n\nexport var1=val1\n\n./hello_world.sh\n"
    mock_slurm_client.put.assert_called_once_with(
        job_script,
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
    )
    mock_slurm_client.submit_job.assert_called_once_with(
        "~/oumi_launcher/20241009_130424513094/oumi_job.sh",
        "~/oumi_launcher/20241009_130424513094",
        name="myjob",
        node_count=2,
        ntasks=2,
        threads_per_core=1,
        qos="regular",
        constraint="gpu",
        gpus_per_node=4,
    )
    mock_slurm_client.list_jobs.assert_called_once_with()
    assert job_status == expected_status


def test_perlmutter_cluster_run_job_fails(mock_datetime, mock_slurm_client):
    cluster = PerlmutterCluster("regular.name", mock_slurm_client)
    mock_slurm_client.submit_job.return_value = "234"
    mock_slurm_client.list_jobs.return_value = [
        JobStatus(
            id="1234",
            name="some name",
            status="queued",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.PENDING,
        )
    ]
    with pytest.raises(RuntimeError):
        _ = cluster.run_job(_get_default_job())


def test_perlmutter_cluster_down(mock_datetime, mock_slurm_client):
    cluster = PerlmutterCluster("debug.name", mock_slurm_client)
    cluster.down()
    # Nothing to assert, this method is a no-op.


def test_perlmutter_cluster_stop(mock_datetime, mock_slurm_client):
    cluster = PerlmutterCluster("debug.name", mock_slurm_client)
    cluster.stop()
    # Nothing to assert, this method is a no-op.
