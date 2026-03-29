"""CLI entry point for config-health tool."""

from __future__ import annotations

import os
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
    TimeRemainingColumn,
)
from rich.table import Table

from config_health.core.classifier import classify_config
from config_health.core.coverage import analyze_coverage
from config_health.core.hub_checker import HubChecker
from config_health.core.models import (
    CheckResult,
    CheckStatus,
    ConfigEntry,
    ConfigType,
    HealthReport,
    Severity,
)
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
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def _collect_environment() -> dict[str, str]:
    """Capture runtime environment info that may affect check results."""
    import platform
    import subprocess

    env: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    # Git commit for reproducibility
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            env["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    # Key library versions
    for pkg in ("transformers", "torch", "peft", "oumi", "trl", "bitsandbytes", "deepspeed"):
        try:
            mod = __import__(pkg)
            env[f"{pkg}_version"] = getattr(mod, "__version__", "?")
        except ImportError:
            pass

    # Flash Attention version
    try:
        import flash_attn

        env["flash_attn_version"] = getattr(flash_attn, "__version__", "?")
    except ImportError:
        pass

    # CUDA / GPU details
    try:
        import torch

        env["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
            env["cuda_device_count"] = str(torch.cuda.device_count())
            cap = torch.cuda.get_device_capability(0)
            env["cuda_capability"] = f"{cap[0]}.{cap[1]}"
            total_memory = torch.cuda.get_device_properties(0).total_memory
            env["cuda_total_memory_gb"] = f"{total_memory / (1024**3):.1f}"

        # CUDA toolkit version (from torch)
        env["cuda_version"] = getattr(torch.version, "cuda", "N/A") or "N/A"

        # cuDNN version
        if torch.backends.cudnn.is_available():
            env["cudnn_version"] = str(torch.backends.cudnn.version())
            env["cudnn_enabled"] = str(torch.backends.cudnn.enabled)
    except Exception:
        pass

    # NVIDIA driver version via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            env["nvidia_driver"] = result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    # NCCL version (relevant for multi-GPU)
    try:
        import torch.cuda.nccl

        env["nccl_version"] = ".".join(str(v) for v in torch.cuda.nccl.version())
    except Exception:
        pass

    # System RAM (cgroup-aware for containers)
    from config_health.core.dry_run import _get_available_ram_gb, _read_cgroup_memory_limit_gb

    ram_gb = _get_available_ram_gb()
    if ram_gb > 0:
        env["available_ram_gb"] = f"{ram_gb:.1f}"
    cgroup_gb = _read_cgroup_memory_limit_gb()
    if cgroup_gb > 0:
        env["cgroup_memory_limit_gb"] = f"{cgroup_gb:.1f}"

    return env


def _suppress_library_warnings():
    """Suppress transformers/HF warnings that corrupt Rich progress output."""
    import contextlib
    import logging
    import os
    import warnings

    @contextlib.contextmanager
    def _ctx():
        # Suppress Python warnings (transformers/trl/torchao emit warnings at import)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Suppress noisy loggers that corrupt Rich progress output
            loggers_to_quiet = ["transformers", "oumi", "torch", "trl", "torchao"]
            old_levels = {}
            for name in loggers_to_quiet:
                logger = logging.getLogger(name)
                old_levels[name] = logger.level
                logger.setLevel(logging.ERROR)
            # Suppress HF hub / transformers env-based warnings
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            # Redirect stderr to suppress C-level warnings (torch elastic, torchao)
            # that bypass Python's warning system
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            old_stderr_fd = os.dup(2)
            os.dup2(devnull_fd, 2)
            os.close(devnull_fd)
            try:
                yield
            finally:
                # Restore stderr
                os.dup2(old_stderr_fd, 2)
                os.close(old_stderr_fd)
                for name, level in old_levels.items():
                    logging.getLogger(name).setLevel(level)

    return _ctx()


def _build_report(
    repo_root: str,
    *,
    offline: bool = False,
    tier0: bool = False,
    hub_check: bool = True,
    vram: bool = False,
    dry_run: bool = False,
    dry_run_steps: int = 2,
    quick: bool = False,
    incremental: bool = False,
    last_report_path: str | None = None,
    auto_fix: bool = False,
    output_path: str | None = None,
    paths: list[str] | None = None,
) -> HealthReport:
    """Scan, classify, and check all configs. Returns a HealthReport."""
    start = time.time()
    report = HealthReport()
    report.environment = _collect_environment()

    def _checkpoint() -> None:
        """Save partial results so interrupted runs aren't lost."""
        if output_path:
            report.scan_duration_s = time.time() - start
            report.to_json(output_path)

    # Discover configs
    all_paths = paths or scan_config_paths(repo_root)

    # C2: Incremental mode — only re-check changed configs
    if incremental and not paths:
        original_count = len(all_paths)
        all_paths = _filter_changed_configs(all_paths, repo_root, last_report_path)
        if len(all_paths) < original_count:
            console.print(f"[dim]Incremental: checking {len(all_paths)} changed configs (of {original_count})[/dim]")
        else:
            console.print(f"[dim]Incremental: no changes detected, checking all {len(all_paths)} configs[/dim]")

    if not all_paths:
        console.print("[yellow]No config files found.[/yellow]")
        return report

    n = len(all_paths)
    training_count = 0  # counted after classification

    with _make_progress() as progress, _suppress_library_warnings():
        # Phase 1: Classify
        phase_start = time.time()
        task = progress.add_task("Scanning configs", total=n)
        for yaml_path in all_paths:
            try:
                entry = classify_config(yaml_path, repo_root)
            except Exception as exc:
                entry = ConfigEntry(
                    path=os.path.relpath(yaml_path, repo_root),
                    abs_path=yaml_path,
                    parse_error=f"Classification crashed: {exc}",
                )
            report.entries.append(entry)
            progress.advance(task)
        training_count = sum(
            1 for e in report.entries if e.config_type == ConfigType.TRAINING
        )
        report.phase_durations_s["classify"] = round(time.time() - phase_start, 2)
        _checkpoint()

        # Phase 2: Static checks
        phase_start = time.time()
        label = "Static checks (quick)" if quick else "Static checks"
        task = progress.add_task(label, total=n)
        for entry in report.entries:
            check_start = time.time()
            try:
                results = run_static_checks(entry, repo_root, skip_finalize=quick)
            except Exception as exc:
                results = [CheckResult(
                    config_path=entry.path,
                    check_name="static_checks",
                    status=CheckStatus.FAIL,
                    message=f"Static checks crashed: {exc}",
                    severity=Severity.ERROR,
                )]
            elapsed = round(time.time() - check_start, 3)
            for r in results:
                r.duration_s = elapsed
            report.check_results.extend(results)
            progress.advance(task)
        report.phase_durations_s["static"] = round(time.time() - phase_start, 2)
        _checkpoint()

        # Phase 3: Hub checks
        if hub_check:
            phase_start = time.time()
            label = "Hub checks (offline)" if offline else "Hub checks"
            task = progress.add_task(label, total=n)
            hub = HubChecker(offline=offline)
            for entry in report.entries:
                check_start = time.time()
                results = hub.check_config(entry)
                elapsed = round(time.time() - check_start, 3)
                for r in results:
                    r.duration_s = elapsed
                report.check_results.extend(results)
                progress.advance(task)
            report.phase_durations_s["hub"] = round(time.time() - phase_start, 2)
            _checkpoint()

        # Phase 4: Tier 0
        if tier0:
            from config_health.core.tier0_checks import run_tier0_checks

            phase_start = time.time()
            task = progress.add_task("Tier 0 (architecture)", total=n)
            for entry in report.entries:
                check_start = time.time()
                results = run_tier0_checks(entry)
                elapsed = round(time.time() - check_start, 3)
                for r in results:
                    r.duration_s = elapsed
                report.check_results.extend(results)
                progress.advance(task)
            report.phase_durations_s["tier0"] = round(time.time() - phase_start, 2)
            _checkpoint()

        # Phase 5: VRAM
        if vram:
            from config_health.core.vram_estimator import estimate_vram

            phase_start = time.time()
            task = progress.add_task("VRAM estimates", total=training_count)
            for entry in report.entries:
                if entry.config_type != ConfigType.TRAINING:
                    continue
                check_start = time.time()
                est = estimate_vram(entry)
                elapsed = round(time.time() - check_start, 3)
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
                        "duration_s": elapsed,
                    }
                progress.advance(task)
            report.phase_durations_s["vram"] = round(time.time() - phase_start, 2)
            _checkpoint()

        # Phase 6: Dry-run
        if dry_run:
            from config_health.core.dry_run import (
                dry_run_to_check_results,
                run_dry_run,
            )

            phase_start = time.time()
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
            report.phase_durations_s["dry_run"] = round(time.time() - phase_start, 2)
            _checkpoint()

    # Coverage analysis + suggestions (fast, no progress needed)
    report.coverage_gaps = analyze_coverage(report.entries)
    for entry in report.entries:
        suggestions = suggest_optimizations(entry)
        report.suggestions.extend(suggestions)

    # C3: Auto-fix known issues
    if auto_fix:
        from config_health.core.auto_fix import apply_fixes

        fix_count = apply_fixes(report, repo_root)
        if fix_count:
            console.print(f"[bold green]Auto-fixed {fix_count} issue(s)[/bold green]")

    report.scan_duration_s = time.time() - start
    return report


def _filter_changed_configs(
    all_paths: list[str], repo_root: str, last_report_path: str | None
) -> list[str]:
    """C2: Filter to only configs changed since last report or git commit."""
    import os
    import subprocess

    # If a previous report is provided, use its timestamp
    if last_report_path and os.path.exists(last_report_path):
        try:
            mtime = os.path.getmtime(last_report_path)
            return [p for p in all_paths if os.path.getmtime(p) > mtime]
        except Exception:
            pass

    # Fall back to git: configs changed vs default branch
    try:
        # Detect default branch (main or master)
        branch_result = subprocess.run(
            ["git", "rev-parse", "--verify", "main"],
            capture_output=True, text=True, cwd=repo_root,
        )
        default_branch = "main" if branch_result.returncode == 0 else "master"

        result = subprocess.run(
            ["git", "diff", "--name-only", default_branch, "--", "configs/"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        if result.returncode == 0:
            changed = set(result.stdout.strip().splitlines())
            changed_abs = {os.path.join(repo_root, p) for p in changed}
            filtered = [p for p in all_paths if p in changed_abs]
            return filtered if filtered else all_paths
    except Exception:
        pass

    return all_paths


def _print_summary(report: HealthReport, *, target: Console | None = None) -> None:
    """Print a summary table to the console (or an alternate target)."""
    out = target or console
    out.print()
    out.print(f"[bold]Config Health Report[/bold]  ({report.scan_duration_s:.1f}s)")
    out.print()

    # Summary counts (using model properties for consistency with dashboard)
    total = report.total
    failures = report.fail_count
    warnings = report.warn_count
    passes = report.pass_count

    out.print(f"  Total configs: {total}")
    out.print(f"  [green]Healthy:[/green]  {passes}")
    out.print(f"  [red]Failing:[/red]  {failures}")
    out.print(f"  [yellow]Warnings:[/yellow] {warnings}")

    # Phase timing breakdown
    if report.phase_durations_s:
        parts = [f"{phase}: {dur}s" for phase, dur in report.phase_durations_s.items()]
        out.print(f"  [dim]Timing: {', '.join(parts)}[/dim]")
    out.print()

    # Type breakdown
    type_table = Table(title="By Config Type")
    type_table.add_column("Type", style="bold")
    type_table.add_column("Count", justify="right")
    for ctype, entries in sorted(report.entries_by_type().items(), key=lambda x: -len(x[1])):
        type_table.add_row(ctype.value, str(len(entries)))
    out.print(type_table)
    out.print()

    # GPU tier breakdown
    tier_table = Table(title="By GPU Tier")
    tier_table.add_column("Tier", style="bold")
    tier_table.add_column("Count", justify="right")
    tier_counts: dict[str, int] = {}
    for e in report.entries:
        tier_counts[e.gpu_tier.label] = tier_counts.get(e.gpu_tier.label, 0) + 1
    for tier, count in sorted(tier_counts.items()):
        tier_table.add_row(tier, str(count))
    out.print(tier_table)
    out.print()

    # Failures
    fail_results = [r for r in report.check_results if r.status == CheckStatus.FAIL]
    if fail_results:
        out.print(f"[bold red]Failures ({len(fail_results)}):[/bold red]")
        for r in fail_results[:20]:
            out.print(f"  [red]✗[/red] {r.config_path}: {r.message}")
        if len(fail_results) > 20:
            out.print(f"  ... and {len(fail_results) - 20} more")
        out.print()

    # Coverage gaps
    if report.coverage_gaps:
        out.print(f"[bold yellow]Coverage Gaps ({len(report.coverage_gaps)}):[/bold yellow]")
        for gap in report.coverage_gaps[:15]:
            out.print(
                f"  [yellow]⚠[/yellow] {gap.model_family} ({gap.category}): "
                f"missing {', '.join(gap.missing_types)}"
            )
        if len(report.coverage_gaps) > 15:
            out.print(f"  ... and {len(report.coverage_gaps) - 15} more")
        out.print()

    # Optimization suggestions summary
    if report.suggestions:
        out.print(f"[bold cyan]Optimization Suggestions: {len(report.suggestions)}[/bold cyan]")
        # Group by category
        by_cat: dict[str, int] = {}
        for s in report.suggestions:
            by_cat[s.category] = by_cat.get(s.category, 0) + 1
        for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
            out.print(f"  {cat}: {count}")
        out.print()


def _write_summary_md(report: HealthReport, path: str) -> None:
    """Write a human-readable markdown summary of the health report."""
    lines: list[str] = []
    w = lines.append

    w("# Config Health Summary")
    w("")
    w(f"Scanned **{report.total}** configs in {report.scan_duration_s:.1f}s.")
    w("")

    # Counts
    w("## Results")
    w("")
    w(f"| Status | Count |")
    w(f"|--------|-------|")
    w(f"| Healthy | {report.pass_count} |")
    w(f"| Warnings | {report.warn_count} |")
    w(f"| Failing | {report.fail_count} |")
    w(f"| **Total** | **{report.total}** |")
    w("")

    # By type
    w("## By Config Type")
    w("")
    w("| Type | Count |")
    w("|------|-------|")
    for ctype, entries in sorted(report.entries_by_type().items(), key=lambda x: -len(x[1])):
        w(f"| {ctype.value} | {len(entries)} |")
    w("")

    # Failures
    fail_results = [r for r in report.check_results if r.status == CheckStatus.FAIL]
    if fail_results:
        w(f"## Failures ({len(fail_results)})")
        w("")
        for r in fail_results:
            w(f"- **{r.config_path}**: {r.message}")
            if r.details:
                w(f"  - {r.details}")
        w("")

    # Warnings
    warn_results = [r for r in report.check_results if r.status == CheckStatus.WARN]
    if warn_results:
        w(f"## Warnings ({len(warn_results)})")
        w("")
        for r in warn_results:
            w(f"- **{r.config_path}**: {r.message}")
        w("")

    # Coverage gaps
    if report.coverage_gaps:
        w(f"## Coverage Gaps ({len(report.coverage_gaps)})")
        w("")
        for gap in report.coverage_gaps:
            w(f"- **{gap.model_family}** ({gap.category}): missing {', '.join(gap.missing_types)}")
        w("")

    # VRAM estimates
    if report.vram_estimates:
        w(f"## VRAM Estimates ({len(report.vram_estimates)} training configs)")
        w("")
        w("| Config | VRAM | Min VRAM | Params | Type |")
        w("|--------|------|----------|--------|------|")
        for cfg_path, est in sorted(report.vram_estimates.items()):
            short = cfg_path.removeprefix("configs/")
            peft = "LoRA" if est.get("is_peft") else "FFT"
            if est.get("is_quantized"):
                peft += "+Q"
            w(f"| {short} | {est['total_vram_gb']:.1f} GB | {est['minimal_total_vram_gb']:.1f} GB | {est['total_params_b']}B | {peft} |")
        w("")

    # Dry-run results
    if report.dry_run_results:
        dr_pass = sum(1 for v in report.dry_run_results.values() if v["success"])
        dr_fail = sum(1 for v in report.dry_run_results.values() if not v["success"] and v["error"])
        w(f"## Dry-Run Results ({dr_pass} passed, {dr_fail} failed)")
        w("")
        for cfg_path, dr in sorted(report.dry_run_results.items()):
            short = cfg_path.removeprefix("configs/")
            if dr["success"]:
                mem = f" ({dr['peak_memory_gb']:.1f} GB)" if dr["peak_memory_gb"] > 0 else ""
                w(f"- {short}: passed in {dr['duration_s']:.1f}s{mem}")
            elif dr["error"]:
                w(f"- {short}: **FAILED** — {dr['error']}")
        w("")

    # Suggestions summary
    if report.suggestions:
        by_cat: dict[str, int] = {}
        for s in report.suggestions:
            by_cat[s.category] = by_cat.get(s.category, 0) + 1
        w(f"## Optimization Suggestions ({len(report.suggestions)})")
        w("")
        for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
            w(f"- {cat}: {count}")
        w("")

    # Environment
    if report.environment:
        w("## Environment")
        w("")
        for k, v in sorted(report.environment.items()):
            w(f"- {k}: {v}")
        w("")

    # Phase timing
    if report.phase_durations_s:
        w("## Phase Timing")
        w("")
        for phase, dur in report.phase_durations_s.items():
            w(f"- {phase}: {dur}s")
        w("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


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
@click.option("--quick", is_flag=True, help="Skip finalize_and_validate() for faster static checks (~10s)")
@click.option("--incremental", is_flag=True, help="Only check configs changed since last report")
@click.option("--last-report", type=str, default=None, help="Path to previous report.json for incremental mode")
@click.option("--fix", is_flag=True, help="Auto-fix known issues (wrong LoRA targets, FSDP layer classes)")
@click.option("--path", type=str, default=None, help="Check a specific config file")
@click.option("--output", "-o", type=str, default=None, help="Save results to this directory (report.json, summary.md, logs.txt)")
@click.option("--repo-root", type=str, default=None, help="Repository root directory")
def check(
    offline: bool,
    tier0: bool,
    vram: bool,
    dry_run: bool,
    exhaustive: bool,
    no_hub: bool,
    quick: bool,
    incremental: bool,
    last_report: str | None,
    fix: bool,
    path: str | None,
    output: str | None,
    repo_root: str | None,
):
    """Run health checks on configs.

    \b
    Examples:
      config-health check                    # static + hub checks
      config-health check --quick            # fast static checks (~10s)
      config-health check --tier0            # + architecture validation
      config-health check --exhaustive       # everything
      config-health check --exhaustive -o results/     # save to directory
      config-health check --incremental      # only changed configs
      config-health check --fix              # auto-fix known issues
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
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            console.print(f"[red]File not found: {path}[/red]")
            sys.exit(1)
        paths = [abs_path]

    # Prepare output directory
    output_dir = None
    json_path = None
    if output:
        output_dir = os.path.abspath(output)
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "report.json")

    # If saving output, tee console to logs.txt
    log_console = None
    if output_dir:
        from rich.console import Console as RichConsole

        log_console = RichConsole(
            file=open(os.path.join(output_dir, "logs.txt"), "w"),
            force_terminal=False,
            no_color=True,
            width=120,
        )

    report = _build_report(
        root,
        offline=offline,
        tier0=tier0,
        hub_check=not no_hub,
        vram=vram,
        dry_run=dry_run,
        quick=quick,
        incremental=incremental,
        last_report_path=last_report,
        auto_fix=fix,
        output_path=json_path,
        paths=paths,
    )
    _print_summary(report)
    if log_console:
        _print_summary(report, target=log_console)

    # Show VRAM summary if estimated
    if report.vram_estimates:
        valid_vrams = [v["total_vram_gb"] for v in report.vram_estimates.values()]
        min_vrams = [v["minimal_total_vram_gb"] for v in report.vram_estimates.values()]
        msg = (
            f"[bold cyan]VRAM Estimates:[/bold cyan] {len(valid_vrams)} training configs, "
            f"{min(min_vrams):.1f} — {max(valid_vrams):.1f} GB"
        )
        console.print(msg)
        console.print()
        if log_console:
            log_console.print(msg)
            log_console.print()

    # Show dry-run summary
    if report.dry_run_results:
        dr_pass = sum(1 for v in report.dry_run_results.values() if v["success"])
        dr_fail = sum(1 for v in report.dry_run_results.values() if not v["success"] and v["error"])
        msg = f"[bold cyan]Dry-runs:[/bold cyan] {dr_pass} passed, {dr_fail} failed"
        console.print(msg)
        if log_console:
            log_console.print(msg)
        for p, v in report.dry_run_results.items():
            if not v["success"] and v["error"]:
                line = f"  [red]✗[/red] {p}: {v['error'][:100]}"
                console.print(line)
                if log_console:
                    log_console.print(line)
        console.print()
        if log_console:
            log_console.print()

    # Save outputs
    if output_dir and json_path:
        report.to_json(json_path)
        _write_summary_md(report, os.path.join(output_dir, "summary.md"))
        if log_console and log_console.file:
            log_console.file.close()
        console.print(f"[bold]Results saved to {output_dir}/[/bold]")
        console.print(f"  report.json  — machine-readable full report")
        console.print(f"  summary.md   — human-readable markdown summary")
        console.print(f"  logs.txt     — plain-text console output")

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
@click.option("--from", "from_report", type=str, default=None, help="Load pre-computed report.json instead of scanning")
@click.option("--repo-root", type=str, default=None, help="Repository root directory")
def ui(port: int, host: str, offline: bool, from_report: str | None, repo_root: str | None):
    """Launch the web dashboard.

    \b
    Examples:
      config-health ui                       # scan and serve
      config-health ui --from report.json    # load pre-computed report
    """
    try:
        root = repo_root or find_repo_root()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    import uvicorn

    from config_health.ui.server import create_app

    app = create_app(root, offline=offline, from_report=from_report)
    console.print(f"[bold]Starting Config Health dashboard on http://{host}:{port}[/bold]")
    if from_report:
        console.print(f"[dim]Loaded from: {from_report}[/dim]")
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
