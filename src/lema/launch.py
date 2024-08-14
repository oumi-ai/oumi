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

    # The user for a job or cluster. Only used by Polaris.
    user: Optional[str] = None

    # The job id for the specific action.
    job_id: Optional[str] = None


def _print_and_wait(message: str, is_done: Callable[[], bool]) -> None:
    """Prints a message with a loading spinner until is_done returns True."""
    spinner = itertools.cycle(["⠁", "⠈", "⠐", "⠠", "⢀", "⡀", "⠄", "⠂"])
    while not is_done():
        _ = sys.stdout.write(f" {next(spinner)} {message}\r")
        _ = sys.stdout.flush()
        time.sleep(0.1)
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
        default=False,
        help="The action to take. "
        "Supported actions: up, down, status, stop, run, which. "
        "Defaults to `up` if not specified.",
    )

    parser.add_argument(
        "--cloud", default=None, help="The cloud to use for the specific action."
    )

    parser.add_argument(
        "-u", "--user", default=None, help="The user for the specific action."
    )

    args, unknown = parser.parse_known_args()
    return _LaunchArgs(
        job=args.path,
        cluster=args.cluster,
        action=_parse_action(args.action),
        cloud=args.cloud,
        user=args.user,
        job_id=args.job,
        additional_args=unknown,
    )


def _down_worker(launch_args: _LaunchArgs) -> None:
    """Turns down a cluster. Executed in a worker thread."""
    if not launch_args.cluster:
        raise ValueError("No cluster specified for down action.")
    if launch_args.cloud:
        cloud = launcher.get_cloud(launch_args.cloud)
        cluster = cloud.get_cluster(launch_args.cluster)
        if cluster:
            cluster.down()
        else:
            logger.warn(f"Cluster {launch_args.cluster} not found.")
        return
    # Make a best effort to find a single cluster to turn down without a cloud.
    clusters = []
    for name in launcher.which_clouds():
        cloud = launcher.get_cloud(name)
        cluster = cloud.get_cluster(launch_args.cluster)
        if cluster:
            clusters.append(cluster)
    if len(clusters) == 0:
        return
    if len(clusters) == 1:
        clusters[0].down()
    else:
        logger.warn(
            f"Multiple clusters found with name {launch_args.cluster}. "
            "Specify a cloud to turn down with `--cloud`."
        )


def stop(launch_args: _LaunchArgs) -> None:
    """Stops a job on LeMa."""
    if not launch_args.cluster:
        raise ValueError("No cluster specified for stop action.")
    if not launch_args.job_id:
        raise ValueError("No job specified for stop action.")
    if not launch_args.cloud:
        raise ValueError("No cloud specified for stop action.")
    launcher.stop(launch_args.job_id, launch_args.cloud, launch_args.cluster)


def down(launch_args: _LaunchArgs) -> None:
    """Turns down a cluster."""
    if not launch_args.cluster:
        raise ValueError("No cluster specified for down action.")
    worker_pool = ThreadPool(processes=1)
    worker_result = worker_pool.apply_async(_down_worker, (launch_args,))
    _print_and_wait(
        f"Turning down cluster `{launch_args.cluster}`", worker_result.ready
    )
    worker_result.wait()


def launch(launch_args: _LaunchArgs) -> None:
    """Launches a job on LeMa."""
    config: launcher.JobConfig = launcher.JobConfig.from_yaml_and_arg_list(
        launch_args.job, launch_args.additional_args, logger=logger
    )
    config.validate()

    # Start the job
    running_cluster, job_status = launcher.up(config, launch_args.cluster)

    _print_and_wait(
        f"Running job {job_status.id}", _create_job_poller(job_status, running_cluster)
    )
    final_status = running_cluster.get_job(job_status.id)
    if final_status:
        logger.info(f"Job {final_status.id} finished with status {final_status.status}")
        logger.info(f"Job metadata: {final_status.metadata}")


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
        pass
    elif launch_args.action == _LauncherAction.STATUS:
        pass
    elif launch_args.action == _LauncherAction.STOP:
        pass
    elif launch_args.action == _LauncherAction.RUN:
        pass
    elif launch_args.action == _LauncherAction.WHICH_CLOUDS:
        pass
    else:
        raise ValueError(f"Invalid action: {launch_args.action}")


if __name__ == "__main__":
    main()
