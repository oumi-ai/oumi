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

"""CLI commands for deploying models to inference providers."""

import asyncio
import logging
import os
import time
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import httpx
import requests
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from oumi.cli.cli_utils import LOG_LEVEL_TYPE, section_header
from oumi.deploy import (
    AutoscalingConfig,
    DeploymentProvider,
    Endpoint,
    EndpointState,
    FireworksDeploymentClient,
    HardwareConfig,
    Model,
    ModelType,
    ParasailDeploymentClient,
    UploadedModel,
)
from oumi.deploy.base_client import BaseDeploymentClient

logger = logging.getLogger(__name__)

CONSOLE = Console()
_DEFAULT_POLL_TIMEOUT_S = 1800  # 30 minutes


def _kv(label: str, value: Any) -> None:
    """Prints a ``[cyan]Label:[/cyan] value`` line to the console."""
    CONSOLE.print(f"[cyan]{label}:[/cyan] {value}")


# ----- Shared colour maps for Rich output -----

_ENDPOINT_STATE_COLORS: dict[EndpointState, str] = {
    EndpointState.RUNNING: "green",
    EndpointState.PENDING: "yellow",
    EndpointState.STARTING: "yellow",
    EndpointState.ERROR: "red",
    EndpointState.STOPPED: "dim",
}

_MODEL_STATUS_COLORS: dict[str, str] = {
    "ready": "green",
    "completed": "green",
    "succeeded": "green",
    "pending": "yellow",
    "queued": "yellow",
    "processing": "yellow",
    "running": "yellow",
    "uploading": "yellow",
    "failed": "red",
    "error": "red",
    "cancelled": "dim",
    "canceled": "dim",
}


# ----- Async helper -----


def _run_async(
    provider: str | None,
    fn: (
        Callable[[BaseDeploymentClient], Coroutine[Any, Any, None]]
        | Callable[[], Coroutine[Any, Any, None]]
    ),
) -> None:
    """Creates a deployment client (when *provider* is given), runs *fn*, then closes.

    When *provider* is ``None`` the callable is invoked with no arguments
    (used by commands that manage their own client lifecycle, e.g. ``list_models``).
    """

    async def _inner() -> None:
        if provider is not None:
            client = _get_deployment_client(provider)
            async with client:
                await fn(client)  # type: ignore[call-arg]
        else:
            await fn()  # type: ignore[call-arg]

    try:
        asyncio.run(_inner())
    except (
        ValueError,
        NotImplementedError,
        httpx.HTTPStatusError,
        requests.HTTPError,
    ) as exc:
        CONSOLE.print(f"\n[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None


def _get_deployment_client(
    provider: str,
) -> BaseDeploymentClient:
    """Gets a deployment client for the specified provider.

    Args:
        provider: Provider name ("fireworks", "parasail")

    Returns:
        Deployment client instance

    Raises:
        ValueError: If provider is not supported
    """
    provider = provider.lower()
    if provider == DeploymentProvider.FIREWORKS.value:
        return FireworksDeploymentClient()
    elif provider == DeploymentProvider.PARASAIL.value:
        return ParasailDeploymentClient()
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {[p.value for p in DeploymentProvider]}"
        )


def _get_available_providers() -> list[str]:
    """Gets a list of providers that have API keys configured.

    Returns:
        List of provider names that can be used
    """
    available = []

    # Check Fireworks.ai
    if os.environ.get("FIREWORKS_API_KEY") and os.environ.get("FIREWORKS_ACCOUNT_ID"):
        available.append("fireworks")

    if os.environ.get("PARASAIL_API_KEY"):
        available.append("parasail")

    return available


async def _poll_model_until_ready(
    client: BaseDeploymentClient,
    provider_model_id: str,
    job_id: str | None,
    console: Console,
    timeout_s: int = _DEFAULT_POLL_TIMEOUT_S,
) -> None:
    """Polls until the model is ready or fails. Raises typer.Exit(1) on failure."""
    console.print("\n[yellow]Waiting for model to be ready...[/yellow]")
    use_job = hasattr(client, "get_job_status") and job_id
    start = time.monotonic()

    while True:
        if time.monotonic() - start > timeout_s:
            console.print(
                f"[red]Error:[/red] Timed out after {timeout_s}s "
                "waiting for model to be ready."
            )
            raise typer.Exit(1)

        if use_job:
            job_status = await client.get_job_status(job_id)  # type: ignore[arg-type]
            cur_status = job_status.get("status", "").lower()
            error_msg = job_status.get("error", "Unknown error")
        else:
            cur_status = (await client.get_model_status(provider_model_id)).lower()
            error_msg = "Unknown error"

        console.print(f"Status: {cur_status}")

        if cur_status in ("ready", "completed", "success"):
            console.print("[green]✓[/green] Model is ready for deployment!")
            return
        if cur_status in ("failed", "error"):
            console.print(f"[red]Error:[/red] Model upload failed: {error_msg}")
            raise typer.Exit(1)

        await asyncio.sleep(10)


async def _poll_endpoint_until_state(
    client: BaseDeploymentClient,
    endpoint_id: str,
    target_state: EndpointState,
    console: Console,
    timeout_s: int = _DEFAULT_POLL_TIMEOUT_S,
) -> Endpoint:
    """Polls until the endpoint reaches *target_state* or ERROR.

    Returns the final Endpoint; raises typer.Exit(1) on ERROR or timeout.
    """
    state_label = target_state.value.lower()
    logger.debug(
        "_poll_endpoint_until_state called: endpoint_id=%s, target=%s, timeout=%s",
        endpoint_id,
        target_state,
        timeout_s,
    )
    console.print(f"\n[yellow]Waiting for endpoint to be {state_label}...[/yellow]")
    start = time.monotonic()

    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout_s:
            console.print(
                f"[red]Error:[/red] Timed out after {timeout_s}s "
                f"waiting for endpoint to be {state_label}."
            )
            raise typer.Exit(1)

        endpoint = await client.get_endpoint(endpoint_id)
        logger.debug(
            "Endpoint %s state=%s (target=%s, elapsed=%.1fs)",
            endpoint_id,
            endpoint.state,
            target_state,
            elapsed,
        )
        console.print(f"State: {endpoint.state.value}")

        if endpoint.state == target_state:
            console.print(f"[green]✓[/green] Endpoint is {state_label}!")
            if endpoint.endpoint_url:
                console.print(f"[cyan]URL:[/cyan] {endpoint.endpoint_url}")
            return endpoint
        if endpoint.state == EndpointState.ERROR:
            console.print("[red]Error:[/red] Endpoint entered error state")
            raise typer.Exit(1)

        await asyncio.sleep(10)


async def _upload_model_and_wait(
    client: BaseDeploymentClient,
    model_source: str,
    model_name: str,
    model_type: ModelType,
    base_model: str | None,
    wait: bool,
    console: Console,
) -> UploadedModel:
    """Uploads a model and optionally polls until ready. Returns the UploadedModel."""
    with console.status("[bold green]Uploading model..."):
        result = await client.upload_model(
            model_source=model_source,
            model_name=model_name,
            model_type=model_type,
            base_model=base_model,
        )
    if wait:
        await _poll_model_until_ready(
            client, result.provider_model_id, result.job_id, console
        )
    return result


def _print_endpoint_table(endpoints: list[Endpoint]) -> None:
    """Prints endpoints in a formatted table.

    Args:
        endpoints: List of endpoints to display
    """
    if not endpoints:
        CONSOLE.print("[yellow]No endpoints found.[/yellow]")
        return

    table = Table(title="Deployments", show_header=True, header_style="bold magenta")
    table.add_column("Endpoint ID", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Model ID", style="blue")
    table.add_column("State", style="yellow")
    table.add_column("Hardware", style="white")
    table.add_column("URL", style="dim", overflow="fold")

    for endpoint in endpoints:
        state_color = _ENDPOINT_STATE_COLORS.get(endpoint.state, "white")

        hw_str = f"{endpoint.hardware.count}x {endpoint.hardware.accelerator}"
        url_str = endpoint.endpoint_url or "-"

        table.add_row(
            endpoint.endpoint_id,
            endpoint.provider.value,
            endpoint.model_id,
            f"[{state_color}]{endpoint.state.value}[/{state_color}]",
            hw_str,
            url_str,
        )

    CONSOLE.print(table)


def _print_hardware_table(hardware_list: list[HardwareConfig]) -> None:
    """Prints available hardware in a formatted table.

    Args:
        hardware_list: List of hardware configurations
    """
    if not hardware_list:
        CONSOLE.print("[yellow]No hardware configurations available.[/yellow]")
        return

    table = Table(
        title="Available Hardware", show_header=True, header_style="bold magenta"
    )
    table.add_column("Accelerator", style="cyan")
    table.add_column("Default Count", style="green")

    for hw in hardware_list:
        table.add_row(hw.accelerator, str(hw.count))

    CONSOLE.print(table)


def _print_models_table(models: list[Model]) -> None:
    """Prints models in a formatted table.

    Args:
        models: List of models to display
    """
    if not models:
        CONSOLE.print("[yellow]No models found.[/yellow]")
        return

    table = Table(
        title="Uploaded Models",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Model ID", style="cyan", no_wrap=False, overflow="fold")
    table.add_column("Name", style="blue", no_wrap=False, overflow="fold")
    table.add_column("Status", style="yellow")
    table.add_column("Type", style="green")
    table.add_column("Provider", style="white")
    table.add_column("Created", style="dim")

    for model in models:
        status_color = _MODEL_STATUS_COLORS.get(model.status.lower(), "white")

        model_type_str = model.model_type.value if model.model_type else "-"
        created_str = (
            model.created_at.strftime("%Y-%m-%d %H:%M") if model.created_at else "-"
        )

        table.add_row(
            model.model_id,
            model.model_name,
            f"[{status_color}]{model.status}[/{status_color}]",
            model_type_str,
            model.provider.value,
            created_str,
        )

    CONSOLE.print(table)


def _print_test_result(result: dict[str, Any]) -> None:
    """Prints the response from a test_endpoint call."""
    choices = result.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content") or choices[0].get(
            "text", ""
        )
        CONSOLE.print("[bold green]Response:[/bold green]")
        CONSOLE.print(Panel(content.strip() or "(empty)", border_style="green"))
    else:
        CONSOLE.print("[cyan]Raw response:[/cyan]", result)


def upload(
    model_path: Annotated[
        str,
        typer.Option(
            "--model-path",
            "-m",
            help="Path to local model directory",
        ),
    ],
    provider: Annotated[
        DeploymentProvider,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider",
        ),
    ],
    model_name: Annotated[
        str,
        typer.Option(
            "--model-name",
            "-n",
            help="Display name for the model",
        ),
    ],
    model_type: Annotated[
        str,
        typer.Option(
            "--model-type",
            "-t",
            help="Model type: full or adapter",
        ),
    ] = "full",
    base_model: Annotated[
        str | None,
        typer.Option(
            "--base-model",
            "-b",
            help="Base model for LoRA adapters (required if model-type=adapter)",
        ),
    ] = None,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            "-w",
            help="Wait for upload to complete",
        ),
    ] = False,
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """Uploads a model to an inference provider.

    Example::

        oumi deploy upload --model-path /path/to/model/
            --provider fireworks --model-name my-model
    """
    section_header("Upload Model", CONSOLE)

    # Validate inputs
    if model_type not in ["full", "adapter"]:
        CONSOLE.print(
            f"[red]Error:[/red] Invalid model type: {model_type}. "
            "Must be 'full' or 'adapter'"
        )
        raise typer.Exit(1)

    if model_type == "adapter" and not base_model:
        CONSOLE.print(
            "[red]Error:[/red] --base-model is required when --model-type=adapter"
        )
        raise typer.Exit(1)

    _kv("Model Path", model_path)
    _kv("Provider", provider)
    _kv("Model Name", model_name)
    _kv("Model Type", model_type)
    if base_model:
        _kv("Base Model", base_model)

    async def _upload(client: BaseDeploymentClient) -> None:
        result = await _upload_model_and_wait(
            client,
            model_path,
            model_name,
            ModelType(model_type),
            base_model,
            wait,
            CONSOLE,
        )

        CONSOLE.print(
            f"\n[green]✓[/green] Model uploaded successfully!\n"
            f"[cyan]Provider Model ID:[/cyan] {result.provider_model_id}"
        )

        if result.job_id:
            _kv("Job ID", result.job_id)

    _run_async(provider, _upload)


def create_endpoint(
    model_id: Annotated[
        str,
        typer.Option(
            "--model-id",
            "-m",
            help="Provider-specific model ID",
        ),
    ],
    provider: Annotated[
        DeploymentProvider,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider",
        ),
    ],
    hardware: Annotated[
        str,
        typer.Option(
            "--hardware",
            "-hw",
            help="Hardware accelerator (e.g., nvidia_a100_80gb)",
        ),
    ],
    gpu_count: Annotated[
        int,
        typer.Option(
            "--gpu-count",
            "-g",
            help="Number of GPUs",
        ),
    ] = 1,
    min_replicas: Annotated[
        int,
        typer.Option(
            "--min-replicas",
            help="Minimum number of replicas for autoscaling",
        ),
    ] = 1,
    max_replicas: Annotated[
        int,
        typer.Option(
            "--max-replicas",
            help="Maximum number of replicas for autoscaling",
        ),
    ] = 1,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            "-n",
            help="Display name for the endpoint",
        ),
    ] = None,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            "-w",
            help="Wait for endpoint to be ready",
        ),
    ] = False,
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """Creates an inference endpoint for a model.

    Example::

        oumi deploy create-endpoint --model-id my-model
            --provider fireworks --hardware nvidia_a100_80gb
    """
    section_header("Create Endpoint", CONSOLE)

    _kv("Model ID", model_id)
    _kv("Provider", provider)
    _kv("Hardware", f"{gpu_count}x {hardware}")
    _kv("Autoscaling", f"{min_replicas}-{max_replicas} replicas")

    async def _create(client: BaseDeploymentClient) -> None:
        hw_config = HardwareConfig(accelerator=hardware, count=gpu_count)
        autoscaling_config = AutoscalingConfig(
            min_replicas=min_replicas, max_replicas=max_replicas
        )

        with CONSOLE.status("[bold green]Creating endpoint..."):
            endpoint = await client.create_endpoint(
                model_id=model_id,
                hardware=hw_config,
                autoscaling=autoscaling_config,
                display_name=name,
            )

        CONSOLE.print(
            f"\n[green]✓[/green] Endpoint created successfully!\n"
            f"[cyan]Endpoint ID:[/cyan] {endpoint.endpoint_id}\n"
            f"[cyan]State:[/cyan] {endpoint.state.value}"
        )

        if endpoint.endpoint_url:
            _kv("URL", endpoint.endpoint_url)

        if wait:
            endpoint = await _poll_endpoint_until_state(
                client, endpoint.endpoint_id, EndpointState.RUNNING, CONSOLE
            )

    _run_async(provider, _create)


def status(
    endpoint_id: Annotated[
        str,
        typer.Option(
            "--endpoint-id",
            "-e",
            help="Endpoint ID to check status",
        ),
    ],
    provider: Annotated[
        DeploymentProvider,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider",
        ),
    ],
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            "-w",
            help="Watch endpoint status until it's ready",
        ),
    ] = False,
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """Gets deployment status for a specific endpoint.

    Example:
        oumi deploy status --endpoint-id ep-123 --provider fireworks
    """
    section_header("Deployment Status", CONSOLE)

    async def _status(client: BaseDeploymentClient) -> None:
        endpoint = await client.get_endpoint(endpoint_id)

        CONSOLE.print(
            Panel(
                f"[cyan]Endpoint ID:[/cyan] {endpoint.endpoint_id}\n"
                f"[cyan]Provider:[/cyan] {endpoint.provider.value}\n"
                f"[cyan]Model ID:[/cyan] {endpoint.model_id}\n"
                f"[cyan]State:[/cyan] {endpoint.state.value}\n"
                f"[cyan]Hardware:[/cyan] "
                f"{endpoint.hardware.count}x "
                f"{endpoint.hardware.accelerator}\n"
                f"[cyan]Autoscaling:[/cyan] "
                f"{endpoint.autoscaling.min_replicas}"
                f"-{endpoint.autoscaling.max_replicas}"
                f" replicas\n"
                f"[cyan]URL:[/cyan] {endpoint.endpoint_url or 'N/A'}\n"
                f"[cyan]Created:[/cyan] {endpoint.created_at or 'N/A'}",
                title="Endpoint Details",
                border_style="blue",
            )
        )

        if watch and endpoint.state not in (
            EndpointState.RUNNING,
            EndpointState.ERROR,
        ):
            await _poll_endpoint_until_state(
                client, endpoint_id, EndpointState.RUNNING, CONSOLE
            )

    _run_async(provider, _status)


def list_deployments(
    provider: Annotated[
        DeploymentProvider,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider",
        ),
    ],
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """Lists all deployments for a provider.

    Example:
        oumi deploy list --provider fireworks
    """
    section_header("List Deployments", CONSOLE)

    async def _list(client: BaseDeploymentClient) -> None:
        with CONSOLE.status("[bold green]Fetching deployments..."):
            endpoints = await client.list_endpoints()

        _print_endpoint_table(endpoints)

        if endpoints:
            CONSOLE.print(f"\n[cyan]Total endpoints:[/cyan] {len(endpoints)}")

    _run_async(provider, _list)


def list_models(
    provider: Annotated[
        DeploymentProvider | None,
        typer.Option(
            "--provider",
            "-p",
            help=(
                "Deployment provider. "
                "If not specified, shows all providers "
                "with API keys configured."
            ),
        ),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option(
            "--all",
            "-a",
            help=(
                "Include public/platform models "
                "(default: only show your uploaded models)"
            ),
        ),
    ] = False,
    status: Annotated[
        str | None,
        typer.Option(
            "--status",
            "-s",
            help="Filter by status (pending, ready, processing, failed, error)",
        ),
    ] = None,
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """Lists uploaded models for providers, including pending ones.

    By default, shows models from all providers with API keys configured.
    Use --provider to limit to a specific provider.
    Use --all to include public platform models.
    Use --status to filter by specific status (e.g., pending for ongoing uploads).

    Example:
        oumi deploy list-models
        oumi deploy list-models --provider fireworks
        oumi deploy list-models --provider fireworks --all
        oumi deploy list-models --status pending
    """
    section_header("Uploaded Models", CONSOLE)

    async def _list() -> None:
        # Determine which providers to query
        if provider:
            providers = [provider]
        else:
            providers = _get_available_providers()
            if not providers:
                CONSOLE.print(
                    "[yellow]No deployment providers configured. "
                    "Please set FIREWORKS_API_KEY or PARASAIL_API_KEY "
                    "environment variables.[/yellow]"
                )
                return

        # Fetch models from all providers
        all_models_list: list[Model] = []
        for prov in providers:
            try:
                client = _get_deployment_client(prov)
                async with client:
                    with CONSOLE.status(f"[bold green]Fetching models from {prov}..."):
                        provider_models = await client.list_models(
                            include_public=all_models
                        )
                        all_models_list.extend(provider_models)
            except Exception as e:
                CONSOLE.print(
                    f"[yellow]Failed to fetch models from {prov}: {e}[/yellow]"
                )

        # Sort by created_at (most recent first), placing None values at the end
        all_models_list.sort(
            key=lambda m: (
                m.created_at
                if m.created_at
                else datetime.min.replace(tzinfo=timezone.utc)
            ),
            reverse=True,
        )

        # Filter by status if specified
        if status:
            status_lower = status.lower()
            all_models_list = [
                m for m in all_models_list if m.status.lower() == status_lower
            ]
            if not all_models_list:
                CONSOLE.print(
                    f"[yellow]No models found with status '{status}'.[/yellow]"
                )
                return

        _print_models_table(all_models_list)

        if all_models_list:
            CONSOLE.print(f"\n[cyan]Total models:[/cyan] {len(all_models_list)}")

            # Print summary by status
            status_counts: dict[str, int] = {}
            for model in all_models_list:
                model_status = model.status.lower()
                status_counts[model_status] = status_counts.get(model_status, 0) + 1

            if status_counts:
                CONSOLE.print("\n[cyan]Status Summary:[/cyan]")
                for stat, count in sorted(status_counts.items()):
                    status_color = _MODEL_STATUS_COLORS.get(stat, "white")
                    tag = f"[{status_color}]"
                    end = f"[/{status_color}]"
                    label = stat.capitalize()
                    CONSOLE.print(f"  {tag}{label}{end}: {count}")

            if not all_models and not status:
                CONSOLE.print(
                    "\n[dim]Tip: Use --all to include public platform models[/dim]"
                )
                CONSOLE.print(
                    "[dim]Tip: Use --status pending to show only ongoing uploads[/dim]"
                )

    _run_async(None, _list)


def delete(
    endpoint_id: Annotated[
        str,
        typer.Option(
            "--endpoint-id",
            "-e",
            help="Endpoint ID to delete",
        ),
    ],
    provider: Annotated[
        DeploymentProvider,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Skip confirmation prompt and force deletion "
            "(e.g. bypass checks for recent inference requests)",
        ),
    ] = False,
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """Deletes an endpoint.

    Example:
        oumi deploy delete --endpoint-id ep-123 --provider fireworks
    """
    section_header("Delete Endpoint", CONSOLE)

    if not force:
        typer.confirm(
            f"Are you sure you want to delete endpoint {endpoint_id}?",
            abort=True,
        )

    async def _delete(client: BaseDeploymentClient) -> None:
        with CONSOLE.status("[bold red]Deleting endpoint..."):
            await client.delete_endpoint(endpoint_id, force=force)

        CONSOLE.print(f"[green]✓[/green] Endpoint {endpoint_id} deleted successfully!")

    _run_async(provider, _delete)


def start(
    endpoint_id: Annotated[
        str,
        typer.Option(
            "--endpoint-id",
            "-e",
            help="Endpoint ID to start",
        ),
    ],
    provider: Annotated[
        DeploymentProvider,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider",
        ),
    ],
    min_replicas: Annotated[
        int,
        typer.Option(
            "--min-replicas",
            help="Minimum replicas when started",
        ),
    ] = 1,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            "-w",
            help="Wait for endpoint to reach RUNNING state",
        ),
    ] = False,
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """Starts a stopped endpoint (saves cost when resuming).

    Example:
        oumi deploy start --endpoint-id ep-123 --provider fireworks --min-replicas 2
    """
    section_header("Start Endpoint", CONSOLE)

    async def _start(client: BaseDeploymentClient) -> None:
        with CONSOLE.status("[bold green]Starting endpoint..."):
            endpoint = await client.start_endpoint(
                endpoint_id, min_replicas=min_replicas
            )
        CONSOLE.print(
            f"[green]✓[/green] Endpoint {endpoint_id} start requested "
            f"(min_replicas={endpoint.autoscaling.min_replicas})"
        )
        if wait:
            await _poll_endpoint_until_state(
                client, endpoint_id, EndpointState.RUNNING, CONSOLE
            )

    _run_async(provider, _start)


def stop(
    endpoint_id: Annotated[
        str,
        typer.Option(
            "--endpoint-id",
            "-e",
            help="Endpoint ID to stop",
        ),
    ],
    provider: Annotated[
        DeploymentProvider,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider",
        ),
    ],
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            "-w",
            help="Wait for endpoint to reach STOPPED state",
        ),
    ] = False,
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """Stops an endpoint by scaling to 0 replicas (cost savings).

    Example:
        oumi deploy stop --endpoint-id ep-123 --provider fireworks
    """
    section_header("Stop Endpoint", CONSOLE)

    async def _stop(client: BaseDeploymentClient) -> None:
        with CONSOLE.status("[bold yellow]Stopping endpoint..."):
            await client.stop_endpoint(endpoint_id)
        CONSOLE.print(
            f"[green]✓[/green] Endpoint {endpoint_id} stop requested (0 replicas)"
        )
        if wait:
            await _poll_endpoint_until_state(
                client, endpoint_id, EndpointState.STOPPED, CONSOLE
            )

    _run_async(provider, _stop)


def delete_model(
    model_id: Annotated[
        str,
        typer.Option(
            "--model-id",
            "-m",
            help="Model ID to delete",
        ),
    ],
    provider: Annotated[
        DeploymentProvider,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Skip confirmation prompt",
        ),
    ] = False,
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """Deletes an uploaded model from the provider.

    WARNING: This permanently deletes the model. Any deployments using this
    model must be deleted first.

    Examples:
        # Delete a Fireworks model
        oumi deploy delete-model --model-id my-model --provider fireworks

        # Delete with auto-confirmation
        oumi deploy delete-model --model-id my-model --provider fireworks -f
    """
    section_header("Delete Model", CONSOLE)

    if not force:
        typer.confirm(
            f"Are you sure you want to delete model '{model_id}' from {provider}? "
            "This action cannot be undone.",
            abort=True,
        )

    async def _delete_model_fn(client: BaseDeploymentClient) -> None:
        with CONSOLE.status("[bold red]Deleting model..."):
            await client.delete_model(model_id)
        CONSOLE.print(
            f"[green]✓[/green] Model '{model_id}' deleted successfully from {provider}!"
        )

    _run_async(provider, _delete_model_fn)


def list_hardware(
    provider: Annotated[
        DeploymentProvider,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider",
        ),
    ],
    model_id: Annotated[
        str | None,
        typer.Option(
            "--model-id",
            "-m",
            help="Filter hardware compatible with this model",
        ),
    ] = None,
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """Lists available hardware configurations.

    Example:
        oumi deploy list-hardware --provider fireworks
    """
    section_header("Available Hardware", CONSOLE)

    async def _list_hw(client: BaseDeploymentClient) -> None:
        with CONSOLE.status("[bold green]Fetching hardware options..."):
            hardware_list = await client.list_hardware(model_id=model_id)

        _print_hardware_table(hardware_list)

        if hardware_list:
            CONSOLE.print(f"\n[cyan]Total configurations:[/cyan] {len(hardware_list)}")

    _run_async(provider, _list_hw)


def test(
    endpoint_id: Annotated[
        str,
        typer.Option(
            "--endpoint-id",
            "-e",
            help="Endpoint ID to test",
        ),
    ],
    provider: Annotated[
        DeploymentProvider,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider",
        ),
    ],
    prompt: Annotated[
        str,
        typer.Option(
            "--prompt",
            help="Test prompt to send to the endpoint",
        ),
    ] = "Hello, how are you?",
    max_tokens: Annotated[
        int,
        typer.Option(
            "--max-tokens",
            help="Maximum tokens to generate",
        ),
    ] = 100,
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """Tests an endpoint with a sample request.

    Example:
        oumi deploy test --endpoint-id ep-123 --provider fireworks --prompt "Hello!"
    """
    section_header("Test Endpoint", CONSOLE)

    async def _test(client: BaseDeploymentClient) -> None:
        endpoint = await client.get_endpoint(endpoint_id)

        if endpoint.state != EndpointState.RUNNING:
            CONSOLE.print(
                "[red]Error:[/red] Endpoint is not running "
                f"(state: {endpoint.state.value})"
            )
            raise typer.Exit(1)

        if not endpoint.endpoint_url:
            CONSOLE.print(
                "[red]Error:[/red] Endpoint URL not available. Cannot test endpoint."
            )
            raise typer.Exit(1)

        _kv("Endpoint", endpoint.endpoint_url)
        _kv("Model", endpoint.model_id)
        CONSOLE.print(f"[cyan]Prompt:[/cyan] {prompt}\n")  # trailing newline

        with CONSOLE.status("[bold green]Sending test request..."):
            result = await client.test_endpoint(
                endpoint_url=endpoint.endpoint_url,
                prompt=prompt,
                model_id=endpoint.inference_model_name or endpoint.model_id,
                max_tokens=max_tokens,
            )
        _print_test_result(result)

    _run_async(provider, _test)


def up(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to deployment config YAML file",
        ),
    ] = None,
    model_path: Annotated[
        str | None,
        typer.Option(
            "--model-path",
            "-m",
            help="Path to local model directory (overrides config)",
        ),
    ] = None,
    provider: Annotated[
        DeploymentProvider | None,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider (overrides config)",
        ),
    ] = None,
    hardware: Annotated[
        str | None,
        typer.Option(
            "--hardware",
            help="Hardware accelerator (overrides config)",
        ),
    ] = None,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait/--no-wait",
            help="Wait for deployment to be ready",
        ),
    ] = True,
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """Deploys a model end-to-end (upload + create endpoint).

    Example::

        oumi deploy up --config deploy_config.yaml
        oumi deploy up --model-path /path/to/model/
            --provider fireworks --hardware nvidia_a100_80gb
    """
    from oumi.deploy.deploy_config import DeploymentConfig

    section_header("Deploy Model", CONSOLE)

    # Load and validate config
    if config:
        try:
            deploy_cfg = DeploymentConfig.from_yaml(config)
        except FileNotFoundError:
            CONSOLE.print(f"[red]Error:[/red] Config file not found: {config}")
            raise typer.Exit(1) from None
        except ValueError as exc:
            CONSOLE.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from None
        CONSOLE.print(f"[cyan]Loaded config from:[/cyan] {config}\n")
    else:
        deploy_cfg = DeploymentConfig()

    deploy_cfg.apply_cli_overrides(
        model_path=model_path, provider=provider, hardware=hardware
    )

    try:
        deploy_cfg.finalize_and_validate()
    except ValueError as exc:
        CONSOLE.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

    assert deploy_cfg.model_source is not None  # guaranteed by validation
    assert deploy_cfg.provider is not None
    model_source: str = deploy_cfg.model_source

    _kv("Model Path", deploy_cfg.model_source)
    _kv("Provider", deploy_cfg.provider)
    CONSOLE.print(
        f"[cyan]Hardware:[/cyan] "
        f"{deploy_cfg.hardware.count}x {deploy_cfg.hardware.accelerator}"
    )

    async def _deploy(client: BaseDeploymentClient) -> None:
        assert deploy_cfg.model_source is not None  # guaranteed by validation
        CONSOLE.print("\n[bold]Step 1: Uploading model...[/bold]")
        upload_result = await _upload_model_and_wait(
            client,
            model_source,
            deploy_cfg.model_name,
            ModelType(deploy_cfg.model_type),
            deploy_cfg.base_model,
            wait,
            CONSOLE,
        )

        CONSOLE.print(
            f"[green]✓[/green] Model uploaded: {upload_result.provider_model_id}"
        )

        CONSOLE.print("\n[bold]Step 2: Creating endpoint...[/bold]")

        with CONSOLE.status("[bold green]Creating endpoint..."):
            endpoint = await client.create_endpoint(
                model_id=upload_result.provider_model_id,
                hardware=deploy_cfg.hardware,
                autoscaling=deploy_cfg.autoscaling,
                display_name=deploy_cfg.model_name,
            )

        CONSOLE.print(f"[green]✓[/green] Endpoint created: {endpoint.endpoint_id}")

        if wait:
            endpoint = await _poll_endpoint_until_state(
                client, endpoint.endpoint_id, EndpointState.RUNNING, CONSOLE
            )

        CONSOLE.print("\n" + "=" * 60)
        CONSOLE.print("[bold green]Deployment Complete![/bold green]")
        CONSOLE.print("=" * 60)
        _kv("Endpoint ID", endpoint.endpoint_id)
        _kv("Model ID", upload_result.provider_model_id)
        _kv("State", endpoint.state.value)
        if endpoint.endpoint_url:
            _kv("URL", endpoint.endpoint_url)

        if (
            deploy_cfg.test_prompts
            and endpoint.state == EndpointState.RUNNING
            and endpoint.endpoint_url
        ):
            CONSOLE.print("\n[bold]Running test prompts...[/bold]")
            for test_prompt in deploy_cfg.test_prompts:
                CONSOLE.print(f"\n[cyan]Prompt:[/cyan] {test_prompt}")
                with CONSOLE.status("[bold green]Sending test request..."):
                    result = await client.test_endpoint(
                        endpoint_url=endpoint.endpoint_url,
                        prompt=test_prompt,
                        model_id=endpoint.inference_model_name or endpoint.model_id,
                    )
                _print_test_result(result)

    _run_async(deploy_cfg.provider, _deploy)
