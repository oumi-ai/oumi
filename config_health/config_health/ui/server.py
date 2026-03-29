"""FastAPI web dashboard for config health."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config_health.core.classifier import classify_config
from config_health.core.coverage import analyze_coverage, build_arch_coverage, build_coverage_matrix
from config_health.core.hub_checker import HubChecker
from config_health.core.models import CheckStatus, ConfigEntry, ConfigType, GpuTier, HealthReport
from config_health.core.optimizer import suggest_optimizations
from config_health.core.scaffolder import get_available_tasks, scaffold_config
from config_health.core.scanner import scan_config_paths
from config_health.core.static_checks import run_static_checks

_UI_DIR = Path(__file__).parent
_TEMPLATES_DIR = _UI_DIR / "templates"
_STATIC_DIR = _UI_DIR / "static"


def create_app(
    repo_root: str,
    *,
    offline: bool = False,
    from_report: str | None = None,
) -> FastAPI:
    # Shared state
    state: dict = {"repo_root": repo_root, "offline": offline, "report": HealthReport(), "scan_error": None}

    def _scan() -> HealthReport:
        report = HealthReport()
        paths = scan_config_paths(repo_root)
        for p in paths:
            try:
                entry = classify_config(p, repo_root)
            except Exception as exc:
                entry = ConfigEntry(
                    path=os.path.relpath(p, repo_root),
                    abs_path=p,
                    parse_error=f"Classification crashed: {exc}",
                )
            report.entries.append(entry)

        hub = HubChecker(offline=offline)
        for entry in report.entries:
            try:
                report.check_results.extend(run_static_checks(entry, repo_root))
                report.check_results.extend(hub.check_config(entry))
                report.suggestions.extend(suggest_optimizations(entry))
            except Exception as exc:
                from config_health.core.models import CheckResult as CR, Severity as Sev
                report.check_results.append(CR(
                    config_path=entry.path,
                    check_name="scan_error",
                    status=CheckStatus.FAIL,
                    message=f"Check crashed: {exc}",
                    severity=Sev.ERROR,
                ))
        hub.save_cache()

        report.coverage_gaps = analyze_coverage(report.entries)
        return report

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            if from_report:
                state["report"] = HealthReport.from_json(from_report)
            else:
                state["report"] = await asyncio.to_thread(_scan)
        except Exception as exc:
            import sys
            print(f"WARNING: startup scan failed: {exc}", file=sys.stderr)
            state["scan_error"] = str(exc)
        yield

    app = FastAPI(title="Config Health Dashboard", lifespan=lifespan)
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ── Pages ──────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(
        request: Request,
        type: str | None = Query(None),
        status: str | None = Query(None),
        tier: str | None = Query(None),
        family: str | None = Query(None),
        model: str | None = Query(None),
        q: str | None = Query(None),
    ):
        report: HealthReport = state["report"]
        entries = report.entries

        # Apply filters
        if type:
            entries = [e for e in entries if e.config_type.value == type]
        if tier is not None and tier != "":
            entries = [e for e in entries if str(e.gpu_tier.value) == tier]
        if family:
            entries = [e for e in entries if e.model_family == family]
        if model:
            entries = [e for e in entries if e.model_name == model]
        if q:
            entries = [e for e in entries if q.lower() in e.path.lower()]
        if status:
            all_fail = report.fail_paths
            all_warn = report.warn_paths  # warn-only (excludes configs that also have fails)
            if status == "fail":
                entries = [e for e in entries if e.path in all_fail]
            elif status == "warn":
                entries = [e for e in entries if e.path in all_warn]
            elif status == "pass":
                non_pass = all_fail | all_warn
                entries = [e for e in entries if e.path not in non_pass]

        # Compute summary stats from full report
        all_fail_paths = report.fail_paths
        all_warn_paths = report.warn_paths

        config_types = sorted(set(e.config_type.value for e in report.entries if e.config_type != ConfigType.UNKNOWN))
        families = sorted(set(e.model_family for e in report.entries if e.model_family))
        models = sorted(set(e.model_name for e in report.entries if e.model_name))

        ctx = {
            "report": report,
            "entries": entries,
            "total": len(report.entries),
            "fail_count": len(all_fail_paths),
            "warn_count": len(all_warn_paths),
            "pass_count": len(report.entries) - len(all_fail_paths) - len(all_warn_paths),
            "config_types": config_types,
            "families": families,
            "models": models,
            "gpu_tiers": [(t.value, t.label) for t in GpuTier],
            "current_type": type or "",
            "current_status": status or "",
            "current_tier": tier if tier is not None else "",
            "current_family": family or "",
            "current_model": model or "",
            "current_q": q or "",
            "all_fail_paths": all_fail_paths,
            "all_warn_paths": all_warn_paths,
            "scan_error": state.get("scan_error"),
        }

        if request.headers.get("HX-Request"):
            return templates.TemplateResponse(request, "partials/config_list.html", ctx)
        return templates.TemplateResponse(request, "dashboard.html", ctx)

    @app.get("/coverage", response_class=HTMLResponse)
    async def coverage_page(request: Request):
        report: HealthReport = state["report"]
        matrix = build_coverage_matrix(report.entries)
        col_types = ["training", "inference", "evaluation", "job", "judge", "synthesis"]

        # Architecture coverage (may be slow first time — resolves model_types from HF)
        covered, uncovered = await asyncio.to_thread(build_arch_coverage, report.entries)

        return templates.TemplateResponse(request, "coverage.html", {
            "matrix": matrix,
            "col_types": col_types,
            "gaps": report.coverage_gaps,
            "families": sorted(matrix.keys()),
            "arch_covered": covered,
            "arch_uncovered": uncovered,
        })

    @app.get("/scaffold", response_class=HTMLResponse)
    async def scaffold_page(request: Request):
        return templates.TemplateResponse(request, "scaffold.html", {
            "available_tasks": get_available_tasks(),
        })

    # ── API endpoints ──────────────────────────────────────────────

    @app.get("/api/config-detail/{config_path:path}", response_class=HTMLResponse)
    async def config_detail(request: Request, config_path: str):
        config_path = unquote(config_path)
        report: HealthReport = state["report"]
        entry = next((e for e in report.entries if e.path == config_path), None)
        if not entry:
            return HTMLResponse("<div class='p-4 text-red-500'>Config not found</div>")

        results = report.results_for(config_path)
        suggestions = report.suggestions_for(config_path)

        # VRAM estimate (training configs only)
        vram_est = None
        if entry.config_type == ConfigType.TRAINING:
            from config_health.core.vram_estimator import estimate_vram

            vram_est = estimate_vram(entry)
            if vram_est.error:
                vram_est = None

        # Read raw YAML
        raw_yaml = ""
        try:
            with open(entry.abs_path) as f:
                raw_yaml = f.read()
        except Exception:
            raw_yaml = "Could not read file"

        return templates.TemplateResponse(request, "config_detail.html", {
            "entry": entry,
            "results": results,
            "suggestions": suggestions,
            "vram": vram_est,
            "raw_yaml": raw_yaml,
            "CheckStatus": CheckStatus,
        })

    @app.post("/api/rescan", response_class=HTMLResponse)
    async def rescan(request: Request):
        # Clear all caches before rescanning
        from config_health.core.coverage import clear_model_type_cache
        from config_health.core.scanner import clear_yaml_cache

        clear_yaml_cache()
        clear_model_type_cache()
        state["scan_error"] = None
        state["report"] = await asyncio.to_thread(_scan)
        # Redirect to dashboard
        from starlette.responses import RedirectResponse

        return RedirectResponse(url="/", status_code=303)

    @app.post("/api/scaffold", response_class=HTMLResponse)
    async def do_scaffold(request: Request):
        form = await request.form()
        model_name = str(form.get("model_name", ""))
        tasks = form.getlist("tasks")
        use_lora = form.get("use_lora") == "on"
        raw_output_dir = str(form.get("output_dir", "")).strip() or None
        # Sanitize: resolve and ensure output_dir stays within the repo
        output_dir = None
        if raw_output_dir:
            resolved = os.path.realpath(os.path.join(repo_root, raw_output_dir))
            if resolved.startswith(os.path.realpath(repo_root)):
                output_dir = resolved
            else:
                return templates.TemplateResponse(request, "partials/scaffold_result.html", {
                    "results": [{"task": "error", "success": False, "content": "output_dir must be within the repo"}],
                })

        results: list[dict] = []
        for task in tasks:
            try:
                yaml_content = scaffold_config(
                    model_name=model_name,
                    task_type=str(task),
                    output_dir=output_dir,
                    use_lora=use_lora,
                )
                results.append(
                    {
                        "task": task,
                        "success": True,
                        "content": yaml_content,
                        "saved": output_dir is not None,
                    }
                )
            except Exception as e:
                results.append({"task": task, "success": False, "content": str(e)})

        return templates.TemplateResponse(request, "partials/scaffold_result.html", {
            "results": results,
        })

    return app
