import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import lema.launcher as launcher
from lema.core.types.base_cluster import BaseCluster, JobStatus
from lema.utils.logging import logger

_START_TIME = -1.0


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


@dataclass
class _LaunchArgs:
    """Dataclass to hold launch arguments."""

    # The path to the configuration file to run.
    job: Optional[str]

    # The cluster to use for the job.
    cluster: Optional[str]

    # A flag indicating whether to detach from the job after starting.
    detach: bool

    # Additional arguments to pass to the job.
    additional_args: List[str]


def parse_cli() -> _LaunchArgs:
    """Parses command line arguments and returns the configuration filename."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j", "--job", default=None, help="Path to the configuration file"
    )

    parser.add_argument(
        "-c", "--cluster", default=None, help="The cluster name to use for the job"
    )

    parser.add_argument(
        "-d", "--detach", default=False, help="Detach from the job after starting"
    )

    args, unknown = parser.parse_known_args()
    return _LaunchArgs(args.job, args.cluster, args.detach, unknown)


def main() -> None:
    """Main entry point for launching jobs on LeMa.

    Arguments are fetched from the following sources, ordered by
    decreasing priority:
    1. [Optional] Arguments provided as CLI arguments, in dotfile format
    2. [Optional] Arguments provided in a yaml config file
    3. Default arguments values defined in the data class
    """
    # Load configuration
    launch_args = parse_cli()

    config: launcher.JobConfig = launcher.JobConfig.from_yaml_and_arg_list(
        launch_args.job, launch_args.additional_args, logger=logger
    )
    config.validate()

    # Start the job
    running_cluster, job_status = launcher.up(config, launch_args.cluster)

    if launch_args.detach:
        logger.info(f"Detached from job {job_status.id}")
        return
    # Otherwise, wait for the job to finish.
    _print_and_wait(
        f"Running job {job_status.id}", _create_job_poller(job_status, running_cluster)
    )
    logger.info(f"Job {job_status.id} finished with status {job_status.status}")
    logger.info(f"Job metadata: {job_status.metadata}")


if __name__ == "__main__":
    main()
