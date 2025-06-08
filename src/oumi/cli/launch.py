# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from collections import defaultdict
from multiprocessing.pool import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional

import typer
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.utils.git_utils import get_git_root_dir
from oumi.utils.logging import logger
from oumi.utils.version_utils import is_dev_build

if TYPE_CHECKING:
    from oumi.core.launcher import BaseCluster, JobStatus


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


def _print_and_wait(
    message: str, task: Callable[..., bool], asynchronous=True, **kwargs
) -> None:
    """Prints a message with a loading spinner until the provided task is done."""
    with cli_utils.CONSOLE.status(message):
        if asynchronous:
            with Pool(processes=1) as worker_pool:
                task_done = False
                while not task_done:
                    worker_result = worker_pool.apply_async(task, kwds=kwargs)
                    worker_result.wait()
                    # Call get() to reraise any exceptions that occurred in the worker.
                    task_done = worker_result.get()
        else:
            # Synchronous tasks should be atomic and not block for a significant amount
            # of time. If a task is blocking, it should be run asynchronously.
            while not task(**kwargs):
                sleep_duration = 0.1
                time.sleep(sleep_duration)


def _is_job_done(id: str, cloud: str, cluster: str) -> bool:
    """Returns true IFF a job is no longer running."""
    from oumi import launcher

    running_cloud = launcher.get_cloud(cloud)
    running_cluster = running_cloud.get_cluster(cluster)
    if not running_cluster:
        return True
    status = running_cluster.get_job(id)
    return status.done


def _cancel_worker(id: str, cloud: str, cluster: str) -> bool:
    """Cancels a job.

    All workers must return a boolean to indicate whether the task is done.
    Cancel has no intermediate states, so it always returns True.
    """
    from oumi import launcher

    if not cluster:
        return True
    if not id:
        return True
    if not cloud:
        return True
    launcher.cancel(id, cloud, cluster)
    return True  # Always return true to indicate that the task is done.


def _down_worker(cluster: str, cloud: Optional[str]) -> bool:
    """Turns down a cluster.

    All workers must return a boolean to indicate whether the task is done.
    Down has no intermediate states, so it always returns True.
    """
    from oumi import launcher

    if cloud:
        target_cloud = launcher.get_cloud(cloud)
        target_cluster = target_cloud.get_cluster(cluster)
        if target_cluster:
            target_cluster.down()
        else:
            cli_utils.CONSOLE.print(
                f"[red]Cluster [yellow]{cluster}[/yellow] not found.[/red]"
            )
        return True
    # Make a best effort to find a single cluster to turn down without a cloud.
    clusters = []
    for name in launcher.which_clouds():
        target_cloud = launcher.get_cloud(name)
        target_cluster = target_cloud.get_cluster(cluster)
        if target_cluster:
            clusters.append(target_cluster)
    if len(clusters) == 0:
        cli_utils.CONSOLE.print(
            f"[red]Cluster [yellow]{cluster}[/yellow] not found.[/red]"
        )
        return True
    if len(clusters) == 1:
        clusters[0].down()
    else:
        cli_utils.CONSOLE.print(
            f"[red]Multiple clusters found with name [yellow]{cluster}[/yellow]. "
            "Specify a cloud to turn down with `--cloud`.[/red]"
        )
    return True  # Always return true to indicate that the task is done.


def _stop_worker(cluster: str, cloud: Optional[str]) -> bool:
    """Stops a cluster.

    All workers must return a boolean to indicate whether the task is done.
    Stop has no intermediate states, so it always returns True.
    """
    from oumi import launcher

    if cloud:
        target_cloud = launcher.get_cloud(cloud)
        target_cluster = target_cloud.get_cluster(cluster)
        if target_cluster:
            target_cluster.stop()
        else:
            cli_utils.CONSOLE.print(
                f"[red]Cluster [yellow]{cluster}[/yellow] not found.[/red]"
            )
        return True
    # Make a best effort to find a single cluster to stop without a cloud.
    clusters = []
    for name in launcher.which_clouds():
        target_cloud = launcher.get_cloud(name)
        target_cluster = target_cloud.get_cluster(cluster)
        if target_cluster:
            clusters.append(target_cluster)
    if len(clusters) == 0:
        cli_utils.CONSOLE.print(
            f"[red]Cluster [yellow]{cluster}[/yellow] not found.[/red]"
        )
        return True
    if len(clusters) == 1:
        clusters[0].stop()
    else:
        cli_utils.CONSOLE.print(
            f"[red]Multiple clusters found with name [yellow]{cluster}[/yellow]. "
            "Specify a cloud to stop with `--cloud`.[/red]"
        )
    return True  # Always return true to indicate that the task is done.


def _poll_job(
    job_status: "JobStatus",
    detach: bool,
    cloud: str,
    running_cluster: Optional["BaseCluster"] = None,
) -> None:
    """Polls a job until it is complete.

    If the job is running in detached mode and the job is not on the local cloud,
    the function returns immediately.
    """
    from oumi import launcher

    is_local = cloud == "local"
    if detach and not is_local:
        cli_utils.CONSOLE.print(
            f"Running job [yellow]{job_status.id}[/yellow] in detached mode."
        )
        return
    if detach and is_local:
        cli_utils.CONSOLE.print("Cannot detach from jobs in local mode.")

    if not running_cluster:
        running_cloud = launcher.get_cloud(cloud)
        running_cluster = running_cloud.get_cluster(job_status.cluster)

    assert running_cluster

    _print_and_wait(
        f"Running job [yellow]{job_status.id}[/yellow]",
        _is_job_done,
        asynchronous=not is_local,
        id=job_status.id,
        cloud=cloud,
        cluster=job_status.cluster,
    )

    final_status = running_cluster.get_job(job_status.id)
    if final_status:
        cli_utils.CONSOLE.print(
            f"Job [yellow]{final_status.id}[/yellow] finished with "
            f"status [yellow]{final_status.status}[/yellow]"
        )
        cli_utils.CONSOLE.print(
            f"Job metadata: [yellow]{final_status.metadata}[/yellow]"
        )


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
    level: cli_utils.LOG_LEVEL_TYPE = None,
) -> None:
    """Cancels a job.

    Args:
        cloud: Filter results by this cloud.
        cluster: Filter results by clusters matching this name.
        id: Filter results by jobs matching this job ID.
        level: The logging level for the specified command.
    """
    _print_and_wait(
        f"Canceling job [yellow]{id}[/yellow]",
        _cancel_worker,
        id=id,
        cloud=cloud,
        cluster=cluster,
    )


def down(
    cluster: Annotated[str, typer.Option(help="The cluster to turn down.")],
    cloud: Annotated[
        Optional[str],
        typer.Option(
            help="If specified, only clusters on this cloud will be affected."
        ),
    ] = None,
    level: cli_utils.LOG_LEVEL_TYPE = None,
) -> None:
    """Turns down a cluster.

    Args:
        cluster: The cluster to turn down.
        cloud: If specified, only clusters on this cloud will be affected.
        level: The logging level for the specified command.
    """
    _print_and_wait(
        f"Turning down cluster [yellow]{cluster}[/yellow]",
        _down_worker,
        cluster=cluster,
        cloud=cloud,
    )
    cli_utils.CONSOLE.print(f"Cluster [yellow]{cluster}[/yellow] turned down!")


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
    level: cli_utils.LOG_LEVEL_TYPE = None,
) -> None:
    """Runs a job on the target cluster.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for the job.
        cluster: The cluster to use for this job. If no such cluster exists, a new
            cluster will be created. If unspecified, a new cluster will be created with
            a unique name.
        detach: Run the job in the background.
        level: The logging level for the specified command.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.JOB),
        )
    )

    # Delayed imports
    from oumi import launcher

    # End imports
    parsed_config: launcher.JobConfig = launcher.JobConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.finalize_and_validate()
    parsed_config.working_dir = _get_working_dir(parsed_config.working_dir)
    if not cluster:
        raise ValueError("No cluster specified for the `run` action.")

    job_status = launcher.run(parsed_config, cluster)
    cli_utils.CONSOLE.print(
        f"Job [yellow]{job_status.id}[/yellow] queued on cluster "
        f"[yellow]{cluster}[/yellow]."
    )

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
    level: cli_utils.LOG_LEVEL_TYPE = None,
) -> None:
    """Prints the status of jobs launched from Oumi.

    Optionally, the caller may specify a job id, cluster, or cloud to further filter
    results.

    Args:
        cloud: Filter results by this cloud.
        cluster: Filter results by clusters matching this name.
        id: Filter results by jobs matching this job ID.
        level: The logging level for the specified command.
    """
    # Delayed imports
    from oumi import launcher

    # End imports
    filtered_jobs = launcher.status(cloud=cloud, cluster=cluster, id=id)
    num_jobs = sum(len(cloud_jobs) for cloud_jobs in filtered_jobs.keys())
    # Print the filtered jobs.
    if num_jobs == 0 and (cloud or cluster or id):
        cli_utils.CONSOLE.print(
            "[red]No jobs found for the specified filter criteria: [/red]"
        )
        if cloud:
            cli_utils.CONSOLE.print(f"Cloud: [yellow]{cloud}[/yellow]")
        if cluster:
            cli_utils.CONSOLE.print(f"Cluster: [yellow]{cluster}[/yellow]")
        if id:
            cli_utils.CONSOLE.print(f"Job ID: [yellow]{id}[/yellow]")
    for target_cloud, job_list in filtered_jobs.items():
        cli_utils.section_header(f"Cloud: [yellow]{target_cloud}[/yellow]")
        cluster_name_list = [
            c.name() for c in launcher.get_cloud(target_cloud).list_clusters()
        ]
        if len(cluster_name_list) == 0:
            cli_utils.CONSOLE.print("[red]No matching clusters found.[/red]")
            continue
        # Organize all jobs by cluster.
        jobs_by_cluster: dict[str, list[JobStatus]] = defaultdict(list)
        # List all clusters, even if they don't have jobs.
        for cluster_name in cluster_name_list:
            if not cluster or cluster == cluster_name:
                jobs_by_cluster[cluster_name] = []
        for job in job_list:
            jobs_by_cluster[job.cluster].append(job)
        for target_cluster, jobs in jobs_by_cluster.items():
            title = f"[cyan]Cluster: [yellow]{target_cluster}[/yellow][/cyan]"
            if not jobs:
                body = Text("[red]No matching jobs found.[/red]")
            else:
                jobs_table = Table(show_header=True, show_lines=False)
                jobs_table.add_column("Job", justify="left", style="green")
                jobs_table.add_column("Status", justify="left", style="yellow")
                for job in jobs:
                    jobs_table.add_row(job.id, job.status)
                body = jobs_table
            cli_utils.CONSOLE.print(Panel(body, title=title, border_style="blue"))


def stop(
    cluster: Annotated[str, typer.Option(help="The cluster to stop.")],
    cloud: Annotated[
        Optional[str],
        typer.Option(
            help="If specified, only clusters on this cloud will be affected."
        ),
    ] = None,
    level: cli_utils.LOG_LEVEL_TYPE = None,
) -> None:
    """Stops a cluster.

    Args:
        cluster: The cluster to stop.
        cloud: If specified, only clusters on this cloud will be affected.
        level: The logging level for the specified command.
    """
    _print_and_wait(
        f"Stopping cluster [yellow]{cluster}[/yellow]",
        _stop_worker,
        cluster=cluster,
        cloud=cloud,
    )
    cli_utils.CONSOLE.print(
        f"Cluster [yellow]{cluster}[/yellow] stopped!\n"
        "Use [green]oumi launch down[/green] to turn it down."
    )


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
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Launches a job.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for the job.
        cluster: The cluster to use for this job. If no such cluster exists, a new
            cluster will be created. If unspecified, a new cluster will be created with
            a unique name.
        detach: Run the job in the background.
        level: The logging level for the specified command.
    """
    # Delayed imports
    from oumi import launcher

    # End imports
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.JOB),
        )
    )

    parsed_config: launcher.JobConfig = launcher.JobConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.finalize_and_validate()
    if cluster:
        target_cloud = launcher.get_cloud(parsed_config.resources.cloud)
        target_cluster = target_cloud.get_cluster(cluster)
        if target_cluster:
            cli_utils.CONSOLE.print(
                f"Found an existing cluster: [yellow]{target_cluster.name()}[/yellow]."
            )
            run(ctx, config, cluster, detach)
            return
    parsed_config.working_dir = _get_working_dir(parsed_config.working_dir)
    # Start the job
    running_cluster, job_status = launcher.up(parsed_config, cluster)
    cli_utils.CONSOLE.print(
        f"Job [yellow]{job_status.id}[/yellow] queued on cluster "
        f"[yellow]{running_cluster.name()}[/yellow]."
    )

    _poll_job(
        job_status=job_status,
        detach=detach,
        cloud=parsed_config.resources.cloud,
        running_cluster=running_cluster,
    )


def which(level: cli_utils.LOG_LEVEL_TYPE = None) -> None:
    """Prints the available clouds."""
    # Delayed imports
    from oumi import launcher

    # End imports
    clouds = launcher.which_clouds()
    cloud_options = [Text(f"{cloud}", style="bold cyan") for cloud in clouds]
    cli_utils.CONSOLE.print(
        Panel(
            Columns(cloud_options, equal=True, expand=True, padding=(0, 2)),
            title="[yellow]Available Clouds[/yellow]",
            border_style="blue",
        )
    )
