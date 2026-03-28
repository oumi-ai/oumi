"""FastAPI web dashboard for config health."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config_health.core.classifier import classify_config
from config_health.core.coverage import analyze_coverage, build_coverage_matrix
from config_health.core.hub_checker import HubChecker
from config_health.core.models import CheckStatus, ConfigType, GpuTier, HealthReport
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
    app = FastAPI(title="Config Health Dashboard")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Shared state
    state = {"repo_root": repo_root, "offline": offline, "report": HealthReport()}

    def _scan() -> HealthReport:
        report = HealthReport()
        paths = scan_config_paths(repo_root)
        for p in paths:
            entry = classify_config(p, repo_root)
            report.entries.append(entry)

        hub = HubChecker(offline=offline)
        for entry in report.entries:
            report.check_results.extend(run_static_checks(entry, repo_root))
            report.check_results.extend(hub.check_config(entry))
            report.suggestions.extend(suggest_optimizations(entry))

        report.coverage_gaps = analyze_coverage(report.entries)
        return report

    @app.on_event("startup")
    async def startup():
        if from_report:
            state["report"] = HealthReport.from_json(from_report)
        else:
            state["report"] = _scan()

    # ── Pages ──────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(
        request: Request,
        type: str | None = Query(None),
        status: str | None = Query(None),
        tier: str | None = Query(None),
        family: str | None = Query(None),
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
        if q:
            entries = [e for e in entries if q.lower() in e.path.lower()]
        if status:
            if status == "fail":
                fail_paths = {
                    r.config_path
                    for r in report.check_results
                    if r.status == CheckStatus.FAIL
                }
                entries = [e for e in entries if e.path in fail_paths]
            elif status == "warn":
                warn_paths = {
                    r.config_path
                    for r in report.check_results
                    if r.status == CheckStatus.WARN
                }
                entries = [e for e in entries if e.path in warn_paths]
            elif status == "pass":
                fail_paths = {
                    r.config_path
                    for r in report.check_results
                    if r.status == CheckStatus.FAIL
                }
                entries = [e for e in entries if e.path not in fail_paths]

        # Compute summary stats from full report
        all_fail_paths = {
            r.config_path
            for r in report.check_results
            if r.status == CheckStatus.FAIL
        }
        all_warn_paths = {
            r.config_path
            for r in report.check_results
            if r.status == CheckStatus.WARN
        }

        config_types = sorted(set(e.config_type.value for e in report.entries if e.config_type != ConfigType.UNKNOWN))
        families = sorted(set(e.model_family for e in report.entries if e.model_family))

        ctx = {
            "request": request,
            "report": report,
            "entries": entries,
            "total": len(report.entries),
            "fail_count": len(all_fail_paths),
            "warn_count": len(all_warn_paths - all_fail_paths),
            "pass_count": len(report.entries) - len(all_fail_paths),
            "config_types": config_types,
            "families": families,
            "gpu_tiers": [(t.value, t.label) for t in GpuTier],
            "current_type": type or "",
            "current_status": status or "",
            "current_tier": tier if tier is not None else "",
            "current_family": family or "",
            "current_q": q or "",
            "all_fail_paths": all_fail_paths,
            "all_warn_paths": all_warn_paths,
        }

        if request.headers.get("HX-Request"):
            return templates.TemplateResponse("partials/config_list.html", ctx)
        return templates.TemplateResponse("dashboard.html", ctx)

    @app.get("/coverage", response_class=HTMLResponse)
    async def coverage_page(request: Request):
        report: HealthReport = state["report"]
        matrix = build_coverage_matrix(report.entries)
        col_types = ["training", "inference", "evaluation", "job", "judge", "synthesis"]
        return templates.TemplateResponse(
            "coverage.html",
            {
                "request": request,
                "matrix": matrix,
                "col_types": col_types,
                "gaps": report.coverage_gaps,
                "families": sorted(matrix.keys()),
            },
        )

    @app.get("/scaffold", response_class=HTMLResponse)
    async def scaffold_page(request: Request):
        return templates.TemplateResponse(
            "scaffold.html",
            {
                "request": request,
                "available_tasks": get_available_tasks(),
            },
        )

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

        return templates.TemplateResponse(
            "config_detail.html",
            {
                "request": request,
                "entry": entry,
                "results": results,
                "suggestions": suggestions,
                "vram": vram_est,
                "raw_yaml": raw_yaml,
                "CheckStatus": CheckStatus,
            },
        )

    @app.post("/api/rescan", response_class=HTMLResponse)
    async def rescan(request: Request):
        state["report"] = _scan()
        # Redirect to dashboard
        from starlette.responses import RedirectResponse

        return RedirectResponse(url="/", status_code=303)

    @app.post("/api/scaffold", response_class=HTMLResponse)
    async def do_scaffold(request: Request):
        form = await request.form()
        model_name = str(form.get("model_name", ""))
        tasks = form.getlist("tasks")
        use_lora = form.get("use_lora") == "on"
        output_dir = str(form.get("output_dir", "")).strip() or None

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

        return templates.TemplateResponse(
            "partials/scaffold_result.html",
            {"request": request, "results": results},
        )

    return app
