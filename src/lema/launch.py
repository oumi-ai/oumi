import argparse
import itertools
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.pool import ThreadPool
from typing import Callable, List, Optional

import lema.launcher as launcher
from lema.core.types.base_cluster import BaseCluster, JobStatus
from lema.utils.logging import logger


class _LauncherAction(Enum):
    """An enumeration of actions that can be taken by the launcher."""

    UP = "up"
    DOWN = "down"
    STATUS = "status"
    STOP = "stop"
    RUN = "run"
    WHICH_CLOUDS = "which"


@dataclass
class _LaunchArgs:
    """Dataclass to hold launch arguments."""

    # The action to take.
    action: _LauncherAction

    # The path to the configuration file to run.
    job: Optional[str] = None

    # The cluster to use for the job.
    cluster: Optional[str] = None

    # Additional arguments to pass to the job.
    additional_args: List[str] = field(default_factory=list)

    # The cloud to use for the specific action.
    cloud: Optional[str] = None

    # The job id for the specific action.
    job_id: Optional[str] = None

    # If false, poll until the started job is complete.
    detach: bool = False


def _print_and_wait(message: str, is_done: Callable[[], bool]) -> None:
    """Prints a message with a loading spinner until is_done returns True."""
    spinner = itertools.cycle(["⠁", "⠈", "⠐", "⠠", "⢀", "⡀", "⠄", "⠂"])
    while not is_done():
        _ = sys.stdout.write(f" {next(spinner)} {message}\r")
        _ = sys.stdout.flush()
        time.sleep(0.1)
        # Clear the line before printing the next spinner. This makes the spinner
        # appear animated instead of printing each iteration on a new line.
        # \r (written above) moves the cursor to the beginning of the line.
        # \033[K deletes everything from the cursor to the end of the line.
        _ = sys.stdout.write("\033[K")


def _create_job_poller(
    job_status: JobStatus, cluster: BaseCluster
) -> Callable[[], bool]:
    """Creates a function that polls the job status."""

    def is_done() -> bool:
        """Returns True if the job is done."""
        status = cluster.get_job(job_status.id)
        if status:
            return status.done
        return True

    return is_done


def _down_worker(launch_args: _LaunchArgs) -> None:
    """Turns down a cluster. Executed in a worker thread."""
    if not launch_args.cluster:
        raise ValueError("No cluster specified for `down` action.")
    if launch_args.cloud:
        cloud = launcher.get_cloud(launch_args.cloud)
        cluster = cloud.get_cluster(launch_args.cluster)
        if cluster:
            cluster.down()
        else:
            print(f"Cluster {launch_args.cluster} not found.")
        return
    # Make a best effort to find a single cluster to turn down without a cloud.
    clusters = []
    for name in launcher.which_clouds():
        cloud = launcher.get_cloud(name)
        cluster = cloud.get_cluster(launch_args.cluster)
        if cluster:
            clusters.append(cluster)
    if len(clusters) == 0:
        print(f"Cluster {launch_args.cluster} not found.")
        return
    if len(clusters) == 1:
        clusters[0].down()
    else:
        print(
            f"Multiple clusters found with name {launch_args.cluster}. "
            "Specify a cloud to turn down with `--cloud`."
        )


def _poll_job(
    job_status: JobStatus,
    launch_args: _LaunchArgs,
    cloud: str,
    running_cluster: Optional[BaseCluster] = None,
) -> None:
    """Polls a job until it is complete.

    If the job is running in detached mode and the job is not on the local cloud,
    the function returns immediately.
    """
    if launch_args.detach and cloud != "local":
        print(f"Running job {job_status.id} in detached mode.")
        return
    if launch_args.detach and cloud == "local":
        print("Cannot detach from jobs in local mode.")

    if not running_cluster:
        running_cluster = launcher.get_cloud(cloud).get_cluster(job_status.cluster)

    assert running_cluster

    _print_and_wait(
        f"Running job {job_status.id}", _create_job_poller(job_status, running_cluster)
    )
    final_status = running_cluster.get_job(job_status.id)
    if final_status:
        print(f"Job {final_status.id} finished with status {final_status.status}")
        print(f"Job metadata: {final_status.metadata}")


def _parse_action(action: Optional[str]) -> _LauncherAction:
    """Parses the action from the command line arguments."""
    if not action:
        return _LauncherAction.UP
    try:
        return _LauncherAction(action)
    except ValueError:
        raise ValueError(f"Invalid action: {action}")


def parse_cli() -> _LaunchArgs:
    """Parses command line arguments and returns the configuration filename."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-j", "--job", default=None, help="The job id for the specific action."
    )

    parser.add_argument(
        "-p", "--path", default=None, help="Path to the job configuration file."
    )

    parser.add_argument(
        "-c", "--cluster", default=None, help="The cluster name to use for the job."
    )

    parser.add_argument(
        "-a",
        "--action",
        default=None,
        choices=[a.value for a in _LauncherAction],
        help="The action to take. "
        "Supported actions: up, down, status, stop, run, which. "
        "Defaults to `up` if not specified.",
    )

    parser.add_argument(
        "--cloud", default=None, help="The cloud to use for the specific action."
    )

    parser.add_argument(
        "-d", "--detach", default=False, help="Whether to detach from the job."
    )

    args, unknown = parser.parse_known_args()
    return _LaunchArgs(
        job=args.path,
        cluster=args.cluster,
        action=_parse_action(args.action),
        cloud=args.cloud,
        job_id=args.job,
        detach=args.detach,
        additional_args=unknown,
    )


def status(launch_args: _LaunchArgs) -> None:
    """Prints the status of jobs on LeMa.

    Optionally, the caller may specify a job id, cluster, or cloud to further filter
    results.
    """
    print("========================")
    print("Job status:")
    print("========================")
    filtered_jobs = {}

    for cloud in launcher.which_clouds():
        cloud_obj = launcher.get_cloud(cloud)
        # Ignore clouds not matching the filter criteria.
        if launch_args.cloud and cloud != launch_args.cloud:
            continue
        filtered_jobs[cloud] = {}
        for cluster in cloud_obj.list_clusters():
            # Ignore clusters not matching the filter criteria.
            if launch_args.cluster and cluster.name != launch_args.cluster:
                continue
            filtered_jobs[cloud][cluster.name] = []
            for job in cluster.get_jobs():
                # Ignore jobs not matching the filter criteria.
                if launch_args.job_id and job.id != launch_args.job_id:
                    continue
                filtered_jobs[cloud][cluster.name].append(job)
    # Print the filtered jobs.
    if not filtered_jobs.items():
        print("No jobs found for the specified filter criteria: ")
        if launch_args.cloud:
            print(f"Cloud: {launch_args.cloud}")
        if launch_args.cluster:
            print(f"Cluster: {launch_args.cluster}")
        if launch_args.job_id:
            print(f"Job ID: {launch_args.job_id}")
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


def which() -> None:
    """Prints the available clouds."""
    clouds = launcher.which_clouds()
    print("========================")
    print("Available clouds:")
    print("========================")
    for cloud in clouds:
        print(cloud)


def stop(launch_args: _LaunchArgs) -> None:
    """Stops a job on LeMa."""
    if not launch_args.cluster:
        raise ValueError("No cluster specified for `stop` action.")
    if not launch_args.job_id:
        raise ValueError("No job ID specified for `stop` action.")
    if not launch_args.cloud:
        raise ValueError("No cloud specified for `stop` action.")
    launcher.stop(launch_args.job_id, launch_args.cloud, launch_args.cluster)


def down(launch_args: _LaunchArgs) -> None:
    """Turns down a cluster."""
    if not launch_args.cluster:
        raise ValueError("No cluster specified for `down` action.")
    worker_pool = ThreadPool(processes=1)
    worker_result = worker_pool.apply_async(_down_worker, (launch_args,))
    _print_and_wait(
        f"Turning down cluster `{launch_args.cluster}`", worker_result.ready
    )
    worker_result.wait()


def run(launch_args: _LaunchArgs) -> None:
    """Runs a job on the target cluster."""
    config: launcher.JobConfig = launcher.JobConfig.from_yaml_and_arg_list(
        launch_args.job, launch_args.additional_args, logger=logger
    )
    config.validate()
    if not launch_args.cluster:
        raise ValueError("No cluster specified for the `run` action.")

    job_status = launcher.run(config, launch_args.cluster)
    print(f"Job {job_status.id} queued on cluster {launch_args.cluster}.")

    _poll_job(job_status, launch_args, config.resources.cloud)


def launch(launch_args: _LaunchArgs) -> None:
    """Launches a job on LeMa."""
    config: launcher.JobConfig = launcher.JobConfig.from_yaml_and_arg_list(
        launch_args.job, launch_args.additional_args, logger=logger
    )
    config.validate()

    # Start the job
    running_cluster, job_status = launcher.up(config, launch_args.cluster)
    print(f"Job {job_status.id} queued on cluster {running_cluster.name}.")

    _poll_job(
        job_status, launch_args, config.resources.cloud, running_cluster=running_cluster
    )


def main() -> None:
    """Main entry point for launching jobs on LeMa.

    Arguments are fetched from the following sources, ordered by
    decreasing priority:
    1. [Optional] Arguments provided as CLI arguments, in dotfile format
    2. [Optional] Arguments provided in a yaml config file
    3. Default arguments values defined in the data class
    """
    launch_args = parse_cli()
    if launch_args.action == _LauncherAction.UP:
        launch(launch_args)
    elif launch_args.action == _LauncherAction.DOWN:
        down(launch_args)
    elif launch_args.action == _LauncherAction.STATUS:
        status(launch_args)
    elif launch_args.action == _LauncherAction.STOP:
        stop(launch_args)
    elif launch_args.action == _LauncherAction.RUN:
        run(launch_args)
    elif launch_args.action == _LauncherAction.WHICH_CLOUDS:
        which()
    else:
        raise ValueError(f"Invalid action: {launch_args.action}")


if __name__ == "__main__":
    main()
