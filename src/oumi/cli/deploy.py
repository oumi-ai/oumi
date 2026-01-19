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
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional, Union

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from oumi.cli.cli_utils import LOG_LEVEL_TYPE, section_header
from oumi.deploy import (
    AutoscalingConfig,
    BaseDeploymentClient,
    DeploymentProvider,
    Endpoint,
    EndpointState,
    FireworksDeploymentClient,
    HardwareConfig,
    Model,
    ModelType,
    TogetherDeploymentClient,
)
from oumi.utils.logging import logger

CONSOLE = Console()


def _get_deployment_client(
    provider: str,
) -> Union[TogetherDeploymentClient, FireworksDeploymentClient]:
    """Get deployment client for the specified provider.

    Args:
        provider: Provider name ("together" or "fireworks")

    Returns:
        Deployment client instance

    Raises:
        ValueError: If provider is not supported
    """
    provider = provider.lower()
    if provider == DeploymentProvider.TOGETHER.value:
        return TogetherDeploymentClient()
    elif provider == DeploymentProvider.FIREWORKS.value:
        return FireworksDeploymentClient()
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {[p.value for p in DeploymentProvider]}"
        )


def _get_available_providers() -> list[str]:
    """Get list of providers that have API keys configured.

    Returns:
        List of provider names that can be used
    """
    available = []

    # Check Together.ai
    if os.environ.get("TOGETHER_API_KEY"):
        available.append("together")

    # Check Fireworks.ai
    if os.environ.get("FIREWORKS_API_KEY") and os.environ.get("FIREWORKS_ACCOUNT_ID"):
        available.append("fireworks")

    return available


def _print_endpoint_table(endpoints: list[Endpoint]) -> None:
    """Print endpoints in a formatted table.

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
        state_color = {
            EndpointState.RUNNING: "green",
            EndpointState.PENDING: "yellow",
            EndpointState.STARTING: "yellow",
            EndpointState.ERROR: "red",
            EndpointState.STOPPED: "dim",
        }.get(endpoint.state, "white")

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
    """Print available hardware in a formatted table.

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
    """Print models in a formatted table.

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
        status_color = {
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
        }.get(model.status.lower(), "white")

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


def upload(
    model_path: Annotated[
        str,
        typer.Option(
            "--model-path",
            "-m",
            help="Path to model (local path, S3 URL, or HuggingFace repo)",
        ),
    ],
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider (together, fireworks)",
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
        Optional[str],
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
    """Upload a model to an inference provider.

    Example:
        oumi deploy upload --model-path s3://bucket/model/ --provider together --model-name my-model
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

    CONSOLE.print(f"[cyan]Model Path:[/cyan] {model_path}")
    CONSOLE.print(f"[cyan]Provider:[/cyan] {provider}")
    CONSOLE.print(f"[cyan]Model Name:[/cyan] {model_name}")
    CONSOLE.print(f"[cyan]Model Type:[/cyan] {model_type}")
    if base_model:
        CONSOLE.print(f"[cyan]Base Model:[/cyan] {base_model}")

    async def _upload() -> None:
        client = _get_deployment_client(provider)
        async with client:
            with CONSOLE.status("[bold green]Uploading model..."):
                result = await client.upload_model(
                    model_source=model_path,
                    model_name=model_name,
                    model_type=ModelType(model_type),
                    base_model=base_model,
                )

            CONSOLE.print(
                f"\n[green]✓[/green] Model uploaded successfully!\n"
                f"[cyan]Provider Model ID:[/cyan] {result.provider_model_id}"
            )

            if result.job_id:
                CONSOLE.print(f"[cyan]Job ID:[/cyan] {result.job_id}")

            if wait:
                CONSOLE.print("\n[yellow]Waiting for model to be ready...[/yellow]")
                if hasattr(client, "get_job_status") and result.job_id:
                    # For providers like Together that use job-based status
                    while True:
                        job_status = await client.get_job_status(result.job_id)
                        status = job_status.get("status", "").lower()
                        CONSOLE.print(f"Status: {status}")

                        if status in ["ready", "completed", "success"]:
                            CONSOLE.print(
                                "[green]✓[/green] Model is ready for deployment!"
                            )
                            break
                        elif status in ["failed", "error"]:
                            error_msg = job_status.get("error", "Unknown error")
                            CONSOLE.print(
                                f"[red]✗[/red] Model upload failed: {error_msg}"
                            )
                            raise typer.Exit(1)

                        time.sleep(10)
                else:
                    # For providers that use model-based status
                    while True:
                        status = await client.get_model_status(result.provider_model_id)
                        CONSOLE.print(f"Status: {status}")

                        if status.lower() in ["ready", "completed"]:
                            CONSOLE.print(
                                "[green]✓[/green] Model is ready for deployment!"
                            )
                            break
                        elif status.lower() in ["failed", "error"]:
                            CONSOLE.print("[red]✗[/red] Model upload failed")
                            raise typer.Exit(1)

                        time.sleep(10)

    asyncio.run(_upload())


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
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider (together, fireworks)",
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
        Optional[str],
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
    """Create an inference endpoint for a model.

    Example:
        oumi deploy create-endpoint --model-id my-model --provider together --hardware nvidia_a100_80gb
    """
    section_header("Create Endpoint", CONSOLE)

    CONSOLE.print(f"[cyan]Model ID:[/cyan] {model_id}")
    CONSOLE.print(f"[cyan]Provider:[/cyan] {provider}")
    CONSOLE.print(f"[cyan]Hardware:[/cyan] {gpu_count}x {hardware}")
    CONSOLE.print(f"[cyan]Autoscaling:[/cyan] {min_replicas}-{max_replicas} replicas")

    async def _create() -> None:
        client = _get_deployment_client(provider)
        async with client:
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
                CONSOLE.print(f"[cyan]URL:[/cyan] {endpoint.endpoint_url}")

            if wait:
                CONSOLE.print("\n[yellow]Waiting for endpoint to be ready...[/yellow]")
                while True:
                    endpoint = await client.get_endpoint(endpoint.endpoint_id)
                    CONSOLE.print(f"State: {endpoint.state.value}")

                    if endpoint.state == EndpointState.RUNNING:
                        CONSOLE.print("[green]✓[/green] Endpoint is ready!")
                        if endpoint.endpoint_url:
                            CONSOLE.print(f"[cyan]URL:[/cyan] {endpoint.endpoint_url}")
                        break
                    elif endpoint.state == EndpointState.ERROR:
                        CONSOLE.print("[red]✗[/red] Endpoint deployment failed")
                        raise typer.Exit(1)

                    time.sleep(10)

    asyncio.run(_create())


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
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider (together, fireworks)",
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
    """Get deployment status for a specific endpoint.

    Example:
        oumi deploy status --endpoint-id ep-123 --provider together
    """
    section_header("Deployment Status", CONSOLE)

    async def _status() -> None:
        client = _get_deployment_client(provider)
        async with client:
            while True:
                endpoint = await client.get_endpoint(endpoint_id)

                # Print endpoint details
                CONSOLE.print(
                    Panel(
                        f"[cyan]Endpoint ID:[/cyan] {endpoint.endpoint_id}\n"
                        f"[cyan]Provider:[/cyan] {endpoint.provider.value}\n"
                        f"[cyan]Model ID:[/cyan] {endpoint.model_id}\n"
                        f"[cyan]State:[/cyan] {endpoint.state.value}\n"
                        f"[cyan]Hardware:[/cyan] {endpoint.hardware.count}x {endpoint.hardware.accelerator}\n"
                        f"[cyan]Autoscaling:[/cyan] {endpoint.autoscaling.min_replicas}-{endpoint.autoscaling.max_replicas} replicas\n"
                        f"[cyan]URL:[/cyan] {endpoint.endpoint_url or 'N/A'}\n"
                        f"[cyan]Created:[/cyan] {endpoint.created_at or 'N/A'}",
                        title="Endpoint Details",
                        border_style="blue",
                    )
                )

                if not watch:
                    break

                if endpoint.state in [EndpointState.RUNNING, EndpointState.ERROR]:
                    break

                CONSOLE.print("[yellow]Waiting for endpoint to be ready...[/yellow]")
                time.sleep(10)

    asyncio.run(_status())


def list_deployments(
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider (together, fireworks)",
        ),
    ],
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """List all deployments for a provider.

    Example:
        oumi deploy list --provider together
    """
    section_header("List Deployments", CONSOLE)

    async def _list() -> None:
        client = _get_deployment_client(provider)
        async with client:
            with CONSOLE.status("[bold green]Fetching deployments..."):
                endpoints = await client.list_endpoints()

            _print_endpoint_table(endpoints)

            if endpoints:
                CONSOLE.print(f"\n[cyan]Total endpoints:[/cyan] {len(endpoints)}")

    asyncio.run(_list())


def list_models(
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider (together, fireworks). If not specified, shows all providers with API keys configured.",
        ),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option(
            "--all",
            "-a",
            help="Include public/platform models (default: only show your uploaded models)",
        ),
    ] = False,
    status: Annotated[
        Optional[str],
        typer.Option(
            "--status",
            "-s",
            help="Filter by status (pending, ready, processing, failed, error)",
        ),
    ] = None,
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """List uploaded models for providers, including pending ones.

    By default, shows models from all providers with API keys configured.
    Use --provider to limit to a specific provider.
    Use --all to include public platform models.
    Use --status to filter by specific status (e.g., pending for ongoing uploads).

    Example:
        oumi deploy list-models
        oumi deploy list-models --provider together
        oumi deploy list-models --provider together --all
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
                    "Please set TOGETHER_API_KEY or FIREWORKS_API_KEY environment variables.[/yellow]"
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
            key=lambda m: m.created_at
            if m.created_at
            else datetime.min.replace(tzinfo=timezone.utc),
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
                    status_color = {
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
                    }.get(stat, "white")
                    CONSOLE.print(
                        f"  [{status_color}]{stat.capitalize()}[/{status_color}]: {count}"
                    )

            if not all_models and not status:
                CONSOLE.print(
                    "\n[dim]Tip: Use --all to include public platform models[/dim]"
                )
                CONSOLE.print(
                    "[dim]Tip: Use --status pending to show only ongoing uploads[/dim]"
                )

    asyncio.run(_list())


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
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider (together, fireworks)",
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
    """Delete an endpoint.

    Example:
        oumi deploy delete --endpoint-id ep-123 --provider together
    """
    section_header("Delete Endpoint", CONSOLE)

    if not force:
        confirm = typer.confirm(
            f"Are you sure you want to delete endpoint {endpoint_id}?",
            abort=True,
        )
        if not confirm:
            raise typer.Exit(0)

    async def _delete() -> None:
        client = _get_deployment_client(provider)
        async with client:
            with CONSOLE.status("[bold red]Deleting endpoint..."):
                await client.delete_endpoint(endpoint_id)

            CONSOLE.print(
                f"[green]✓[/green] Endpoint {endpoint_id} deleted successfully!"
            )

    asyncio.run(_delete())


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
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider (together, fireworks)",
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
    """Delete an uploaded model from the provider.

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
        confirm = typer.confirm(
            f"Are you sure you want to delete model '{model_id}' from {provider}? "
            "This action cannot be undone.",
            abort=True,
        )
        if not confirm:
            raise typer.Exit(0)

    async def _delete_model() -> None:
        client = _get_deployment_client(provider)
        async with client:
            try:
                with CONSOLE.status("[bold red]Deleting model..."):
                    await client.delete_model(model_id)

                CONSOLE.print(
                    f"[green]✓[/green] Model '{model_id}' deleted successfully from {provider}!"
                )
            except NotImplementedError as e:
                CONSOLE.print(f"[yellow]⚠[/yellow] {e}")
                raise typer.Exit(1)
            except Exception as e:
                CONSOLE.print(f"[red]✗[/red] Failed to delete model: {e}")
                raise typer.Exit(1)

    asyncio.run(_delete_model())


def list_hardware(
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider (together, fireworks)",
        ),
    ],
    model_id: Annotated[
        Optional[str],
        typer.Option(
            "--model-id",
            "-m",
            help="Filter hardware compatible with this model",
        ),
    ] = None,
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """List available hardware configurations.

    Example:
        oumi deploy list-hardware --provider together
    """
    section_header("Available Hardware", CONSOLE)

    async def _list_hw() -> None:
        client = _get_deployment_client(provider)
        async with client:
            with CONSOLE.status("[bold green]Fetching hardware options..."):
                hardware_list = await client.list_hardware(model_id=model_id)

            _print_hardware_table(hardware_list)

            if hardware_list:
                CONSOLE.print(
                    f"\n[cyan]Total configurations:[/cyan] {len(hardware_list)}"
                )

    asyncio.run(_list_hw())


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
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider (together, fireworks)",
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
    """Test endpoint with a sample request.

    Example:
        oumi deploy test --endpoint-id ep-123 --provider together --prompt "Hello!"
    """
    section_header("Test Endpoint", CONSOLE)

    async def _test() -> None:
        client = _get_deployment_client(provider)
        async with client:
            # Get endpoint details
            endpoint = await client.get_endpoint(endpoint_id)

            if endpoint.state != EndpointState.RUNNING:
                CONSOLE.print(
                    f"[red]Error:[/red] Endpoint is not running (state: {endpoint.state.value})"
                )
                raise typer.Exit(1)

            if not endpoint.endpoint_url:
                CONSOLE.print(
                    "[red]Error:[/red] Endpoint URL not available. "
                    "Cannot test endpoint."
                )
                raise typer.Exit(1)

            CONSOLE.print(f"[cyan]Endpoint:[/cyan] {endpoint.endpoint_url}")
            CONSOLE.print(f"[cyan]Model:[/cyan] {endpoint.model_id}")
            CONSOLE.print(f"[cyan]Prompt:[/cyan] {prompt}\n")

            # Note: This is a placeholder - actual implementation would need
            # to use the appropriate API client (OpenAI SDK, etc.)
            CONSOLE.print(
                "[yellow]Note:[/yellow] Actual API testing requires implementing "
                "provider-specific inference calls. For now, showing endpoint details."
            )

            CONSOLE.print(
                "\n[cyan]Use the endpoint URL above with your preferred API client.[/cyan]"
            )

    asyncio.run(_test())


def up(
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to deployment config YAML file",
        ),
    ] = None,
    model_path: Annotated[
        Optional[str],
        typer.Option(
            "--model-path",
            "-m",
            help="Path to model (overrides config)",
        ),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider",
            "-p",
            help="Deployment provider (overrides config)",
        ),
    ] = None,
    hardware: Annotated[
        Optional[str],
        typer.Option(
            "--hardware",
            help="Hardware accelerator (overrides config)",
        ),
    ] = None,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            "-w",
            help="Wait for deployment to be ready",
        ),
    ] = True,
    log_level: LOG_LEVEL_TYPE = None,
) -> None:
    """Deploy a model end-to-end (upload + create endpoint).

    Example:
        oumi deploy up --config deploy_config.yaml
        oumi deploy up --model-path s3://bucket/model/ --provider together --hardware nvidia_a100_80gb
    """
    section_header("Deploy Model", CONSOLE)

    # Load config if provided
    deploy_config = {}
    if config:
        if not config.exists():
            CONSOLE.print(f"[red]Error:[/red] Config file not found: {config}")
            raise typer.Exit(1)

        with open(config) as f:
            deploy_config = yaml.safe_load(f)

        CONSOLE.print(f"[cyan]Loaded config from:[/cyan] {config}\n")

    # Override with CLI args
    final_model_path = model_path or deploy_config.get("model_source")
    final_provider = provider or deploy_config.get("provider")
    final_model_name = deploy_config.get("model_name", "deployed-model")
    final_model_type = deploy_config.get("model_type", "full")
    final_base_model = deploy_config.get("base_model")

    hw_config = deploy_config.get("hardware", {})
    final_hardware = hardware or hw_config.get("accelerator", "nvidia_a100_80gb")
    final_gpu_count = hw_config.get("count", 1)

    autoscaling_config = deploy_config.get("autoscaling", {})
    final_min_replicas = autoscaling_config.get("min_replicas", 1)
    final_max_replicas = autoscaling_config.get("max_replicas", 1)

    # Validate required params
    if not final_model_path:
        CONSOLE.print(
            "[red]Error:[/red] --model-path is required (or specify in config)"
        )
        raise typer.Exit(1)

    if not final_provider:
        CONSOLE.print("[red]Error:[/red] --provider is required (or specify in config)")
        raise typer.Exit(1)

    CONSOLE.print(f"[cyan]Model Path:[/cyan] {final_model_path}")
    CONSOLE.print(f"[cyan]Provider:[/cyan] {final_provider}")
    CONSOLE.print(f"[cyan]Hardware:[/cyan] {final_gpu_count}x {final_hardware}")

    async def _deploy() -> None:
        client = _get_deployment_client(final_provider)
        async with client:
            # Step 1: Upload model
            CONSOLE.print("\n[bold]Step 1: Uploading model...[/bold]")
            with CONSOLE.status("[bold green]Uploading..."):
                upload_result = await client.upload_model(
                    model_source=final_model_path,
                    model_name=final_model_name,
                    model_type=ModelType(final_model_type),
                    base_model=final_base_model,
                )

            CONSOLE.print(
                f"[green]✓[/green] Model uploaded: {upload_result.provider_model_id}"
            )

            # Wait for model to be ready
            if wait:
                CONSOLE.print("[yellow]Waiting for model to be ready...[/yellow]")
                if hasattr(client, "get_job_status") and upload_result.job_id:
                    while True:
                        job_status = await client.get_job_status(upload_result.job_id)
                        status = job_status.get("status", "").lower()

                        if status in ["ready", "completed", "success"]:
                            CONSOLE.print("[green]✓[/green] Model is ready!")
                            break
                        elif status in ["failed", "error"]:
                            error = job_status.get("error", "Unknown error")
                            CONSOLE.print(f"[red]✗[/red] Upload failed: {error}")
                            raise typer.Exit(1)

                        time.sleep(10)

            # Step 2: Create endpoint
            CONSOLE.print("\n[bold]Step 2: Creating endpoint...[/bold]")
            hw_cfg = HardwareConfig(accelerator=final_hardware, count=final_gpu_count)
            autoscaling_cfg = AutoscalingConfig(
                min_replicas=final_min_replicas, max_replicas=final_max_replicas
            )

            with CONSOLE.status("[bold green]Creating endpoint..."):
                endpoint = await client.create_endpoint(
                    model_id=upload_result.provider_model_id,
                    hardware=hw_cfg,
                    autoscaling=autoscaling_cfg,
                    display_name=final_model_name,
                )

            CONSOLE.print(f"[green]✓[/green] Endpoint created: {endpoint.endpoint_id}")

            # Wait for endpoint to be ready
            if wait:
                CONSOLE.print("[yellow]Waiting for endpoint to be ready...[/yellow]")
                while True:
                    endpoint = await client.get_endpoint(endpoint.endpoint_id)

                    if endpoint.state == EndpointState.RUNNING:
                        CONSOLE.print("[green]✓[/green] Endpoint is ready!")
                        break
                    elif endpoint.state == EndpointState.ERROR:
                        CONSOLE.print("[red]✗[/red] Endpoint deployment failed")
                        raise typer.Exit(1)

                    time.sleep(10)

            # Display final details
            CONSOLE.print("\n" + "=" * 60)
            CONSOLE.print("[bold green]Deployment Complete![/bold green]")
            CONSOLE.print("=" * 60)
            CONSOLE.print(f"[cyan]Endpoint ID:[/cyan] {endpoint.endpoint_id}")
            CONSOLE.print(f"[cyan]Model ID:[/cyan] {upload_result.provider_model_id}")
            CONSOLE.print(f"[cyan]State:[/cyan] {endpoint.state.value}")
            if endpoint.endpoint_url:
                CONSOLE.print(f"[cyan]URL:[/cyan] {endpoint.endpoint_url}")

            # Run test prompts if configured
            test_prompts = deploy_config.get("test_prompts", [])
            if test_prompts and endpoint.state == EndpointState.RUNNING:
                CONSOLE.print("\n[bold]Running test prompts...[/bold]")
                for test_prompt in test_prompts:
                    CONSOLE.print(f"\n[cyan]Prompt:[/cyan] {test_prompt}")
                    CONSOLE.print(
                        "[yellow]Note:[/yellow] Actual testing requires "
                        "provider-specific API implementation"
                    )

    asyncio.run(_deploy())
