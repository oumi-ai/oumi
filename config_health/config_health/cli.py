"""CLI entry point for config-health tool."""

from __future__ import annotations

import sys
import time

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from config_health.core.classifier import classify_config
from config_health.core.coverage import analyze_coverage
from config_health.core.hub_checker import HubChecker
from config_health.core.models import CheckStatus, ConfigType, HealthReport
from config_health.core.optimizer import suggest_optimizations
from config_health.core.scanner import find_repo_root, scan_config_paths
from config_health.core.static_checks import run_static_checks

console = Console()


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def _build_report(
    repo_root: str,
    *,
    offline: bool = False,
    tier0: bool = False,
    hub_check: bool = True,
    vram: bool = False,
    dry_run: bool = False,
    dry_run_steps: int = 2,
    paths: list[str] | None = None,
) -> HealthReport:
    """Scan, classify, and check all configs. Returns a HealthReport."""
    start = time.time()
    report = HealthReport()

    # Discover configs
    all_paths = paths or scan_config_paths(repo_root)
    if not all_paths:
        console.print("[yellow]No config files found.[/yellow]")
        return report

    n = len(all_paths)
    training_count = 0  # counted after classification

    with _make_progress() as progress:
        # Phase 1: Classify
        task = progress.add_task("Scanning configs", total=n)
        for yaml_path in all_paths:
            entry = classify_config(yaml_path, repo_root)
            report.entries.append(entry)
            progress.advance(task)
        training_count = sum(
            1 for e in report.entries if e.config_type == ConfigType.TRAINING
        )

        # Phase 2: Static checks
        task = progress.add_task("Static checks", total=n)
        for entry in report.entries:
            results = run_static_checks(entry, repo_root)
            report.check_results.extend(results)
            progress.advance(task)

        # Phase 3: Hub checks
        if hub_check:
            label = "Hub checks (offline)" if offline else "Hub checks"
            task = progress.add_task(label, total=n)
            hub = HubChecker(offline=offline)
            for entry in report.entries:
                results = hub.check_config(entry)
                report.check_results.extend(results)
                progress.advance(task)

        # Phase 4: Tier 0
        if tier0:
            from config_health.core.tier0_checks import run_tier0_checks

            task = progress.add_task("Tier 0 (architecture)", total=n)
            for entry in report.entries:
                results = run_tier0_checks(entry)
                report.check_results.extend(results)
                progress.advance(task)

        # Phase 5: VRAM
        if vram:
            from config_health.core.vram_estimator import estimate_vram

            task = progress.add_task("VRAM estimates", total=training_count)
            for entry in report.entries:
                if entry.config_type != ConfigType.TRAINING:
                    continue
                est = estimate_vram(entry)
                if not est.error:
                    report.vram_estimates[entry.path] = {
                        "total_vram_gb": est.total_vram_gb,
                        "minimal_total_vram_gb": est.minimal_total_vram_gb,
                        "total_params_b": round(est.total_params_b, 1),
                        "trainable_params_b": round(est.trainable_params_b, 2),
                        "model_memory_gb": round(est.model_memory_gb, 1),
                        "optimizer_memory_gb": round(est.optimizer_memory_gb, 1),
                        "gradient_memory_gb": round(est.gradient_memory_gb, 1),
                        "activation_memory_gb": round(est.activation_memory_gb, 1),
                        "is_peft": est.is_peft,
                        "is_quantized": est.is_quantized,
                        "num_gpus": est.num_gpus,
                        "batch_size": est.batch_size,
                        "seq_len": est.seq_len,
                        "notes": est.notes,
                    }
                progress.advance(task)

        # Phase 6: Dry-run
        if dry_run:
            from config_health.core.dry_run import (
                dry_run_to_check_results,
                run_dry_run,
            )

            task = progress.add_task("Dry-run training", total=training_count)
            for entry in report.entries:
                if entry.config_type != ConfigType.TRAINING:
                    continue
                dr = run_dry_run(entry, max_steps=dry_run_steps)
                report.check_results.extend(dry_run_to_check_results(dr))
                report.dry_run_results[entry.path] = {
                    "success": dr.success,
                    "steps_completed": dr.steps_completed,
                    "duration_s": round(dr.duration_s, 1),
                    "peak_memory_gb": round(dr.peak_memory_gb, 1),
                    "error": dr.error,
                    "notes": dr.notes,
                }
                progress.advance(task)

    # Coverage analysis + suggestions (fast, no progress needed)
    report.coverage_gaps = analyze_coverage(report.entries)
    for entry in report.entries:
        suggestions = suggest_optimizations(entry)
        report.suggestions.extend(suggestions)

    report.scan_duration_s = time.time() - start
    return report


def _print_summary(report: HealthReport) -> None:
    """Print a summary table to the console."""
    console.print()
    console.print(f"[bold]Config Health Report[/bold]  ({report.scan_duration_s:.1f}s)")
    console.print()

    # Summary counts
    total = report.total
    passes = sum(
        1
        for e in report.entries
        if all(
            r.status != CheckStatus.FAIL
            for r in report.results_for(e.path)
        )
    )
    failures = total - passes
    warnings = sum(
        1
        for e in report.entries
        if any(
            r.status == CheckStatus.WARN
            for r in report.results_for(e.path)
        )
    )

    console.print(f"  Total configs: {total}")
    console.print(f"  [green]Healthy:[/green]  {passes}")
    console.print(f"  [red]Failing:[/red]  {failures}")
    console.print(f"  [yellow]Warnings:[/yellow] {warnings}")
    console.print()

    # Type breakdown
    type_table = Table(title="By Config Type")
    type_table.add_column("Type", style="bold")
    type_table.add_column("Count", justify="right")
    for ctype, entries in sorted(report.entries_by_type().items(), key=lambda x: -len(x[1])):
        type_table.add_row(ctype.value, str(len(entries)))
    console.print(type_table)
    console.print()

    # GPU tier breakdown
    tier_table = Table(title="By GPU Tier")
    tier_table.add_column("Tier", style="bold")
    tier_table.add_column("Count", justify="right")
    tier_counts: dict[str, int] = {}
    for e in report.entries:
        tier_counts[e.gpu_tier.label] = tier_counts.get(e.gpu_tier.label, 0) + 1
    for tier, count in sorted(tier_counts.items()):
        tier_table.add_row(tier, str(count))
    console.print(tier_table)
    console.print()

    # Failures
    fail_results = [r for r in report.check_results if r.status == CheckStatus.FAIL]
    if fail_results:
        console.print(f"[bold red]Failures ({len(fail_results)}):[/bold red]")
        for r in fail_results[:20]:
            console.print(f"  [red]✗[/red] {r.config_path}: {r.message}")
        if len(fail_results) > 20:
            console.print(f"  ... and {len(fail_results) - 20} more")
        console.print()

    # Coverage gaps
    if report.coverage_gaps:
        console.print(f"[bold yellow]Coverage Gaps ({len(report.coverage_gaps)}):[/bold yellow]")
        for gap in report.coverage_gaps[:15]:
            console.print(
                f"  [yellow]⚠[/yellow] {gap.model_family} ({gap.category}): "
                f"missing {', '.join(gap.missing_types)}"
            )
        if len(report.coverage_gaps) > 15:
            console.print(f"  ... and {len(report.coverage_gaps) - 15} more")
        console.print()

    # Optimization suggestions summary
    if report.suggestions:
        console.print(f"[bold cyan]Optimization Suggestions: {len(report.suggestions)}[/bold cyan]")
        # Group by category
        by_cat: dict[str, int] = {}
        for s in report.suggestions:
            by_cat[s.category] = by_cat.get(s.category, 0) + 1
        for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
            console.print(f"  {cat}: {count}")
        console.print()


@click.group()
def main():
    """Config Health — health checker and dashboard for Oumi configs."""
    pass


@main.command()
@click.option("--offline", is_flag=True, help="Skip HuggingFace Hub checks")
@click.option("--tier0", is_flag=True, help="Run tier 0 checks (tokenizer/model config/architecture)")
@click.option("--vram", is_flag=True, help="Estimate VRAM for training configs")
@click.option("--dry-run", is_flag=True, help="Run training dry-runs with random weights")
@click.option("--exhaustive", is_flag=True, help="Run all checks: tier0 + hub + VRAM + dry-run")
@click.option("--no-hub", is_flag=True, help="Skip HuggingFace Hub existence checks entirely")
@click.option("--path", type=str, default=None, help="Check a specific config file")
@click.option("--output", "-o", type=str, default=None, help="Save report as JSON to this path")
@click.option("--repo-root", type=str, default=None, help="Repository root directory")
def check(
    offline: bool,
    tier0: bool,
    vram: bool,
    dry_run: bool,
    exhaustive: bool,
    no_hub: bool,
    path: str | None,
    output: str | None,
    repo_root: str | None,
):
    """Run health checks on configs.

    \b
    Examples:
      config-health check                    # static + hub checks
      config-health check --tier0            # + architecture validation
      config-health check --exhaustive       # everything
      config-health check --exhaustive -o report.json  # save results
    """
    if exhaustive:
        tier0 = True
        vram = True
        dry_run = True

    try:
        root = repo_root or find_repo_root()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    paths = None
    if path:
        import os

        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            console.print(f"[red]File not found: {path}[/red]")
            sys.exit(1)
        paths = [abs_path]

    report = _build_report(
        root,
        offline=offline,
        tier0=tier0,
        hub_check=not no_hub,
        vram=vram,
        dry_run=dry_run,
        paths=paths,
    )
    _print_summary(report)

    # Show VRAM summary if estimated
    if report.vram_estimates:
        valid_vrams = [v["total_vram_gb"] for v in report.vram_estimates.values()]
        min_vrams = [v["minimal_total_vram_gb"] for v in report.vram_estimates.values()]
        console.print(
            f"[bold cyan]VRAM Estimates:[/bold cyan] {len(valid_vrams)} training configs, "
            f"{min(min_vrams):.1f} — {max(valid_vrams):.1f} GB"
        )
        console.print()

    # Show dry-run summary
    if report.dry_run_results:
        dr_pass = sum(1 for v in report.dry_run_results.values() if v["success"])
        dr_fail = sum(1 for v in report.dry_run_results.values() if not v["success"] and v["error"])
        console.print(
            f"[bold cyan]Dry-runs:[/bold cyan] {dr_pass} passed, {dr_fail} failed"
        )
        for p, v in report.dry_run_results.items():
            if not v["success"] and v["error"]:
                console.print(f"  [red]✗[/red] {p}: {v['error'][:100]}")
        console.print()

    # Save report
    if output:
        report.to_json(output)
        console.print(f"[bold]Report saved to {output}[/bold]")

    # Exit with non-zero if there are failures
    fail_count = sum(
        1
        for r in report.check_results
        if r.status == CheckStatus.FAIL
    )
    if fail_count > 0:
        sys.exit(1)


@main.command()
@click.option("--port", type=int, default=8777, help="Port for the dashboard")
@click.option("--host", type=str, default="127.0.0.1", help="Host to bind to")
@click.option("--offline", is_flag=True, help="Skip HuggingFace Hub checks")
@click.option("--repo-root", type=str, default=None, help="Repository root directory")
def ui(port: int, host: str, offline: bool, repo_root: str | None):
    """Launch the web dashboard."""
    try:
        root = repo_root or find_repo_root()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    import uvicorn

    from config_health.ui.server import create_app

    app = create_app(root, offline=offline)
    console.print(f"[bold]Starting Config Health dashboard on http://{host}:{port}[/bold]")
    uvicorn.run(app, host=host, port=port, log_level="warning")


@main.command()
@click.option("--repo-root", type=str, default=None, help="Repository root directory")
def report(repo_root: str | None):
    """Print coverage report."""
    try:
        root = repo_root or find_repo_root()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    report_data = _build_report(root, hub_check=False)

    # Coverage matrix
    from config_health.core.coverage import build_coverage_matrix

    matrix = build_coverage_matrix(report_data.entries)
    col_types = ["training", "inference", "evaluation", "job", "judge", "synthesis"]

    table = Table(title="Coverage Matrix")
    table.add_column("Model Family", style="bold")
    for ct in col_types:
        table.add_column(ct.capitalize(), justify="center")

    for family in sorted(matrix.keys()):
        row = [family]
        for ct in col_types:
            entries = matrix[family].get(ct, [])
            if entries:
                row.append(f"[green]{len(entries)}[/green]")
            else:
                row.append("[dim]—[/dim]")
        table.add_row(*row)

    console.print(table)


@main.command()
@click.option("--path", type=str, default=None, help="Estimate for a specific config file")
@click.option("--family", type=str, default=None, help="Filter by model family")
@click.option("--repo-root", type=str, default=None, help="Repository root directory")
def vram(path: str | None, family: str | None, repo_root: str | None):
    """Estimate VRAM requirements for training configs."""
    import os

    from config_health.core.classifier import classify_config
    from config_health.core.vram_estimator import estimate_vram

    try:
        root = repo_root or find_repo_root()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    if path:
        paths = [os.path.abspath(path)]
    else:
        paths = scan_config_paths(root)

    entries = []
    for p in paths:
        entry = classify_config(p, root)
        if entry.config_type != ConfigType.TRAINING:
            continue
        if family and entry.model_family != family:
            continue
        entries.append(entry)

    if not entries:
        console.print("[yellow]No training configs found.[/yellow]")
        return

    console.print(f"Estimating VRAM for {len(entries)} training configs...\n")

    table = Table(title="VRAM Estimates")
    table.add_column("Config", style="cyan", max_width=55, overflow="ellipsis")
    table.add_column("Params", justify="right", style="blue")
    table.add_column("Type", style="magenta")
    table.add_column("VRAM", justify="right", style="bold green")
    table.add_column("Min VRAM", justify="right", style="green")
    table.add_column("GPUs", justify="right")
    table.add_column("Notes", style="dim", max_width=30, overflow="ellipsis")

    estimates = []
    for entry in entries:
        est = estimate_vram(entry)
        estimates.append((entry, est))
        if est.error:
            table.add_row(
                entry.short_path, "?", "?",
                f"[dim]{est.error}[/dim]", "", "", "",
            )
            continue

        peft_label = "LoRA" if est.is_peft else "FFT"
        if est.is_quantized:
            peft_label += f"+Q{int(est.quant_bytes * 8)}"

        table.add_row(
            entry.short_path,
            f"{est.total_params_b:.1f}B",
            peft_label,
            f"{est.total_vram_gb:.1f} GB",
            f"{est.minimal_total_vram_gb:.1f} GB",
            str(est.num_gpus),
            ", ".join(est.notes[:2]),
        )

    console.print(table)

    # Summary
    valid = [e for _, e in estimates if not e.error]
    if valid:
        console.print(f"\n[bold]Summary:[/bold] {len(valid)} configs estimated")
        max_vram = max(e.total_vram_gb for e in valid)
        min_vram = min(e.minimal_total_vram_gb for e in valid)
        console.print(f"  Range: {min_vram:.1f} GB (minimal) — {max_vram:.1f} GB (as configured)")

        # GPU tier buckets
        tiers = {"< 24 GB (consumer)": 0, "24-48 GB (A10/A6000)": 0, "48-80 GB (A100)": 0, "> 80 GB (multi-GPU)": 0}
        for e in valid:
            v = e.minimal_total_vram_gb
            if v < 24:
                tiers["< 24 GB (consumer)"] += 1
            elif v < 48:
                tiers["24-48 GB (A10/A6000)"] += 1
            elif v < 80:
                tiers["48-80 GB (A100)"] += 1
            else:
                tiers["> 80 GB (multi-GPU)"] += 1

        console.print("  By minimum GPU tier:")
        for tier, count in tiers.items():
            if count:
                console.print(f"    {tier}: {count}")


@main.command(name="dry-run")
@click.option("--path", type=str, default=None, help="Dry-run a specific training config")
@click.option("--family", type=str, default=None, help="Filter by model family")
@click.option("--steps", type=int, default=2, help="Number of training steps")
@click.option("--repo-root", type=str, default=None, help="Repository root directory")
def dry_run(path: str | None, family: str | None, steps: int, repo_root: str | None):
    """Run training dry-run with random weights (no model download)."""
    import os

    from config_health.core.dry_run import dry_run_to_check_results, run_dry_run

    try:
        root = repo_root or find_repo_root()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    if path:
        paths = [os.path.abspath(path)]
    else:
        paths = scan_config_paths(root)

    entries = []
    for p in paths:
        entry = classify_config(p, root)
        if entry.config_type != ConfigType.TRAINING:
            continue
        if family and entry.model_family != family:
            continue
        entries.append(entry)

    if not entries:
        console.print("[yellow]No training configs found.[/yellow]")
        return

    console.print(f"[bold]Dry-running {len(entries)} training configs ({steps} steps each)...[/bold]\n")

    passed = 0
    failed = 0
    skipped = 0
    for entry in entries:
        dr = run_dry_run(entry, max_steps=steps)
        if dr.error and dr.error in (
            "Remote engine", "GGUF model (not supported for dry-run)",
            "Local checkpoint path", "Not a training config", "No model_name",
        ):
            skipped += 1
            continue

        if dr.success:
            passed += 1
            mem = f" ({dr.peak_memory_gb:.1f} GB)" if dr.peak_memory_gb > 0 else ""
            console.print(
                f"  [green]✓[/green] {entry.short_path}  "
                f"[dim]{dr.duration_s:.1f}s{mem}[/dim]"
            )
        else:
            failed += 1
            console.print(f"  [red]✗[/red] {entry.short_path}")
            console.print(f"    [red]{dr.error}[/red]")

    console.print(f"\n[bold]Results:[/bold] {passed} passed, {failed} failed, {skipped} skipped")
    if failed > 0:
        sys.exit(1)


@main.command()
@click.argument("model_name")
@click.option("--tasks", type=str, default="training,inference,evaluation", help="Comma-separated task types")
@click.option("--output-dir", type=str, default=None, help="Output directory for generated configs")
@click.option("--no-lora", is_flag=True, help="Generate full finetune config (no LoRA)")
def scaffold(model_name: str, tasks: str, output_dir: str | None, no_lora: bool):
    """Generate config files from templates."""
    from config_health.core.scaffolder import scaffold_config

    task_list = [t.strip() for t in tasks.split(",")]
    for task in task_list:
        try:
            result = scaffold_config(
                model_name=model_name,
                task_type=task,
                output_dir=output_dir,
                use_lora=not no_lora,
            )
            if output_dir:
                console.print(f"  [green]✓[/green] Generated: {result}")
            else:
                console.print(f"[bold]--- {task} ---[/bold]")
                console.print(result)
                console.print()
        except ValueError as e:
            console.print(f"  [red]✗[/red] {e}")
