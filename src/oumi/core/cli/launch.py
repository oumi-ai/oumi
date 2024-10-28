import itertools
import sys
import time
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Annotated, Callable, Optional

import typer

import oumi.core.cli.cli_utils as cli_utils
from oumi import launcher
from oumi.core.launcher import BaseCluster, JobStatus
from oumi.utils.git_utils import get_git_root_dir
from oumi.utils.logging import logger
from oumi.utils.version_utils import is_dev_build


def _get_working_dir(current: str) -> str:
    """Prompts the user to select the working directory, if relevant."""
    if not is_dev_build():
        return current
    oumi_root = get_git_root_dir()
    if not oumi_root or oumi_root == Path(current).resolve():
        return current
    use_root = typer.confirm(
        "You are using a dev build of oumi. "
        f"Use oumi's root directory ({oumi_root}) as your working directory?",
        abort=False,
        default=True,
    )
    return str(oumi_root) if use_root else current


def _print_spinner_and_sleep(
    message: str, spinner: itertools.cycle, sleep_duration: float
) -> None:
    """Prints a message with a loading spinner and sleeps."""
    _ = sys.stdout.write(f" {next(spinner)} {message}\r")
    _ = sys.stdout.flush()
    time.sleep(sleep_duration)
    # Clear the line before printing the next spinner. This makes the spinner appear
    # animated instead of printing each iteration on a new line. \r (written above)
    # moves the cursor to the beginning of the line. \033[K deletes everything from the
    # cursor to the end of the line.
    _ = sys.stdout.write("\033[K")


def _print_and_wait(message: str, task: Callable, asynchronous=True, **kwargs) -> None:
    """Prints a message with a loading spinner until the provided task is done."""
    spinner = itertools.cycle(["⠁", "⠈", "⠐", "⠠", "⢀", "⡀", "⠄", "⠂"])
    sleep_duration = 0.1
    if asynchronous:
        with Pool(processes=1) as worker_pool:
            worker_result = worker_pool.apply_async(task, kwds=kwargs)
            while not worker_result.ready():
                _print_spinner_and_sleep(message, spinner, sleep_duration)
            worker_result.wait()
            # Call get() to reraise any exceptions that occurred in the worker.
            worker_result.get()
    else:
        while not task(**kwargs):
            _print_spinner_and_sleep(message, spinner, sleep_duration)


def _is_job_done(id: str, cloud: str, cluster: str) -> bool:
    """Returns true IFF a job is no longer running."""
    running_cloud = launcher.get_cloud(cloud)
    running_cluster = running_cloud.get_cluster(cluster)
    if not running_cluster:
        return True
    status = running_cluster.get_job(id)
    return status.done


def _cancel_worker(id: str, cloud: str, cluster: str) -> None:
    """Cancels a job."""
    if not cluster:
        return
    if not id:
        return
    if not cloud:
        return
    launcher.cancel(id, cloud, cluster)


def _down_worker(cluster: str, cloud: Optional[str]):
    """Turns down a cluster."""
    if cloud:
        target_cloud = launcher.get_cloud(cloud)
        target_cluster = target_cloud.get_cluster(cluster)
        if target_cluster:
            target_cluster.down()
        else:
            print(f"Cluster {cluster} not found.")
        return
    # Make a best effort to find a single cluster to turn down without a cloud.
    clusters = []
    for name in launcher.which_clouds():
        target_cloud = launcher.get_cloud(name)
        target_cluster = target_cloud.get_cluster(cluster)
        if target_cluster:
            clusters.append(target_cluster)
    if len(clusters) == 0:
        print(f"Cluster {cluster} not found.")
        return
    if len(clusters) == 1:
        clusters[0].down()
    else:
        print(
            f"Multiple clusters found with name {cluster}. "
            "Specify a cloud to turn down with `--cloud`."
        )


def _poll_job(
    job_status: JobStatus,
    detach: bool,
    cloud: str,
    running_cluster: Optional[BaseCluster] = None,
) -> None:
    """Polls a job until it is complete.

    If the job is running in detached mode and the job is not on the local cloud,
    the function returns immediately.
    """
    is_local = cloud == "local"
    if detach and not is_local:
        print(f"Running job {job_status.id} in detached mode.")
        return
    if detach and is_local:
        print("Cannot detach from jobs in local mode.")

    if not running_cluster:
        running_cloud = launcher.get_cloud(cloud)
        running_cluster = running_cloud.get_cluster(job_status.cluster)

    assert running_cluster

    _print_and_wait(
        "Running job {job_status.id}",
        _is_job_done,
        asynchronous=not is_local,
        id=job_status.id,
        cloud=cloud,
        cluster=job_status.cluster,
    )

    final_status = running_cluster.get_job(job_status.id)
    if final_status:
        print(f"Job {final_status.id} finished with status {final_status.status}")
        print(f"Job metadata: {final_status.metadata}")


# ----------------------------
# Launch CLI subcommands
# ----------------------------


def cancel(
    cloud: Annotated[str, typer.Option(help="Filter results by this cloud.")],
    cluster: Annotated[
        str,
        typer.Option(help="Filter results by clusters matching this name."),
    ],
    id: Annotated[
        str, typer.Option(help="Filter results by jobs matching this job ID.")
    ],
) -> None:
    """Cancels a job.

    Args:
        cloud: Filter results by this cloud.
        cluster: Filter results by clusters matching this name.
        id: Filter results by jobs matching this job ID.
    """
    _print_and_wait(
        f"Canceling job {id}", _cancel_worker, id=id, cloud=cloud, cluster=cluster
    )


def down(
    cluster: Annotated[str, typer.Option(help="The cluster to turn down.")],
    cloud: Annotated[
        Optional[str],
        typer.Option(
            help="If specified, only clusters on this cloud will be affected."
        ),
    ] = None,
) -> None:
    """Turns down a cluster.

    Args:
        cluster: The cluster to turn down.
        cloud: If specified, only clusters on this cloud will be affected.
    """
    _print_and_wait(
        f"Turning down cluster `{cluster}`",
        _down_worker,
        cluster=cluster,
        cloud=cloud,
    )


def run(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS, help="Path to the configuration file for the job."
        ),
    ],
    cluster: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "The cluster to use for this job. If unspecified, a new cluster will "
                "be created."
            )
        ),
    ] = None,
    detach: Annotated[
        bool, typer.Option(help="Run the job in the background.")
    ] = False,
) -> None:
    """Runs a job on the target cluster.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for the job.
        cluster: The cluster to use for this job. If no such cluster exists, a new
            cluster will be created. If unspecified, a new cluster will be created with
            a unique name.
        detach: Run the job in the background.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)
    parsed_config: launcher.JobConfig = launcher.JobConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.validate()
    parsed_config.working_dir = _get_working_dir(parsed_config.working_dir)
    if not cluster:
        raise ValueError("No cluster specified for the `run` action.")

    job_status = launcher.run(parsed_config, cluster)
    print(f"Job {job_status.id} queued on cluster {cluster}.")

    _poll_job(job_status=job_status, detach=detach, cloud=parsed_config.resources.cloud)


def status(
    cloud: Annotated[
        Optional[str], typer.Option(help="Filter results by this cloud.")
    ] = None,
    cluster: Annotated[
        Optional[str],
        typer.Option(help="Filter results by clusters matching this name."),
    ] = None,
    id: Annotated[
        Optional[str], typer.Option(help="Filter results by jobs matching this job ID.")
    ] = None,
) -> None:
    """Prints the status of jobs launched from Oumi.

    Optionally, the caller may specify a job id, cluster, or cloud to further filter
    results.

    Args:
        cloud: Filter results by this cloud.
        cluster: Filter results by clusters matching this name.
        id: Filter results by jobs matching this job ID.
    """
    print("========================")
    print("Job status:")
    print("========================")
    filtered_jobs = {}

    for target_cloud in launcher.which_clouds():
        cloud_obj = launcher.get_cloud(target_cloud)
        # Ignore clouds not matching the filter criteria.
        if cloud and target_cloud != cloud:
            continue
        filtered_jobs[cloud] = {}
        for target_cluster in cloud_obj.list_clusters():
            # Ignore clusters not matching the filter criteria.
            if cluster and target_cluster.name() != cluster:
                continue
            filtered_jobs[cloud][target_cluster.name()] = []
            for job in target_cluster.get_jobs():
                # Ignore jobs not matching the filter criteria.
                if id and job.id != id:
                    continue
                filtered_jobs[cloud][target_cluster.name()].append(job)
    # Print the filtered jobs.
    if not filtered_jobs.items():
        print("No jobs found for the specified filter criteria: ")
        if cloud:
            print(f"Cloud: {cloud}")
        if cluster:
            print(f"Cluster: {cluster}")
        if id:
            print(f"Job ID: {id}")
    for cloud, clusters in filtered_jobs.items():
        print(f"Cloud: {cloud}")
        if not clusters.items():
            print("No matching clusters found.")
        for cluster, jobs in clusters.items():
            print(f"Cluster: {cluster}")
            if not jobs:
                print("No matching jobs found.")
            for job in jobs:
                print(f"Job: {job.id} Status: {job.status}")


def up(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS, help="Path to the configuration file for the job."
        ),
    ],
    cluster: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "The cluster to use for this job. If unspecified, a new cluster will "
                "be created."
            )
        ),
    ] = None,
    detach: Annotated[
        bool, typer.Option(help="Run the job in the background.")
    ] = False,
):
    """Launches a job.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for the job.
        cluster: The cluster to use for this job. If no such cluster exists, a new
            cluster will be created. If unspecified, a new cluster will be created with
            a unique name.
        detach: Run the job in the background.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)
    parsed_config: launcher.JobConfig = launcher.JobConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.validate()
    if cluster:
        target_cloud = launcher.get_cloud(parsed_config.resources.cloud)
        target_cluster = target_cloud.get_cluster(cluster)
        if target_cluster:
            print(f"Found an existing cluster: {target_cluster.name()}.")
            run(ctx, config, cluster, detach)
            return
    parsed_config.working_dir = _get_working_dir(parsed_config.working_dir)
    # Start the job
    running_cluster, job_status = launcher.up(parsed_config, cluster)
    print(f"Job {job_status.id} queued on cluster {running_cluster.name()}.")

    _poll_job(
        job_status=job_status,
        detach=detach,
        cloud=parsed_config.resources.cloud,
        running_cluster=running_cluster,
    )


def which() -> None:
    """Prints the available clouds."""
    clouds = launcher.which_clouds()
    print("========================")
    print("Available clouds:")
    print("========================")
    for cloud in clouds:
        print(cloud)
