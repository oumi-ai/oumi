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
from config_health.core.coverage import analyze_coverage, build_arch_coverage, build_coverage_matrix, build_scale_coverage
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

    def _build_scale_type_breakdown(
        entries: list, sizes: list[str]
    ) -> dict[str, dict[str, int]]:
        """Build size -> {config_type: count} for drilldown view."""
        result: dict[str, dict[str, int]] = {s: {} for s in sizes}
        for e in entries:
            if not e.model_meta or not e.model_meta.size_label:
                continue
            size = e.model_meta.size_label
            if size not in result:
                continue
            ctype = e.config_type.value
            result[size][ctype] = result[size].get(ctype, 0) + 1
        return result

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
        col_types = ["training", "inference", "evaluation"]
        all_col_types = ["training", "inference", "evaluation", "job", "judge", "synthesis"]

        # Architecture coverage
        covered, uncovered = await asyncio.to_thread(build_arch_coverage, report.entries)

        # Scale coverage
        scale_matrix = build_scale_coverage(report.entries)

        # Build per-family summaries for the drilldown view
        gap_by_family = {g.model_family: g for g in report.coverage_gaps}
        family_summaries = []
        for family in sorted(matrix.keys()):
            types_present = set(matrix[family].keys())
            config_count = sum(len(v) for v in matrix[family].values())
            expected = set(col_types)
            present = types_present & expected
            score = len(present) / len(expected) if expected else 1.0
            gap = gap_by_family.get(family)
            scales = scale_matrix.get(family, {})
            sorted_scales = sorted(scales.keys(), key=lambda s: float(s.rstrip("B"))) if scales else []

            # Find arch info and ALL sizes for this family's models
            family_entries = [e for e in report.entries if e.model_family == family]
            arch_type = ""
            model_name_sample = ""
            all_family_sizes: set[str] = set()
            for e in family_entries:
                if e.model_meta and e.model_meta.model_type and not arch_type:
                    arch_type = e.model_meta.model_type
                if e.model_name and not model_name_sample:
                    model_name_sample = e.model_name
                # Collect ALL sizes seen for this family (from any config type)
                if e.model_meta and e.model_meta.size_label:
                    all_family_sizes.add(e.model_meta.size_label)

            # Discover all sizes that exist on HF Hub for this family
            from config_health.core.enrichment import discover_hub_sizes

            hub_sizes: dict[str, str] = {}  # size_label -> model_id
            if model_name_sample:
                for mid, size_b in discover_hub_sizes(model_name_sample):
                    if size_b >= 1:
                        label = f"{int(size_b)}B" if size_b == int(size_b) else f"{size_b:.1f}B"
                    else:
                        label = f"{size_b:.1f}B"
                    hub_sizes[label] = mid

            covered_sizes = set(scales.keys()) if scales else set()
            # Missing = sizes on HF Hub that have no configs in recipes/projects
            missing_from_hub = {s: mid for s, mid in hub_sizes.items() if s not in all_family_sizes}
            sorted_missing = sorted(missing_from_hub.keys(), key=lambda s: float(s.rstrip("B")))

            family_summaries.append({
                "name": family,
                "score": score,
                "config_count": config_count,
                "types_present": sorted(types_present),
                "types_missing": sorted(expected - present),
                "all_types": {t: len(matrix[family].get(t, [])) for t in all_col_types},
                "scales": {s: len(scales[s]) for s in sorted_scales},
                "missing_scales": sorted_missing,
                "missing_scale_models": {s: missing_from_hub[s] for s in sorted_missing},
                # Per-size per-type breakdown for the drilldown
                "scale_types": _build_scale_type_breakdown(family_entries, sorted_scales or sorted(all_family_sizes, key=lambda s: float(s.rstrip("B")))),
                "arch_type": arch_type,
                "model_name_sample": model_name_sample,
                "category": gap.category if gap else "recipes",
            })

        # Sort: incomplete families first, then by name
        family_summaries.sort(key=lambda f: (f["score"], f["name"]))

        # Architectures: covered, Oumi-registered uncovered, and all others
        popular_covered = covered[:20]
        oumi_uncovered = [a for a in uncovered if a.in_oumi_registry]
        other_uncovered = [a for a in uncovered if not a.in_oumi_registry]

        return templates.TemplateResponse(request, "coverage.html", {
            "families": family_summaries,
            "col_types": all_col_types,
            "arch_covered": popular_covered,
            "arch_uncovered_oumi": oumi_uncovered,
            "arch_uncovered_other": other_uncovered,
            "total_arch": len(covered) + len(uncovered),
            "covered_count": len(covered),
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
        from config_health.core.enrichment import clear_metadata_cache
        from config_health.core.scanner import clear_yaml_cache

        clear_yaml_cache()
        clear_model_type_cache()
        clear_metadata_cache()
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
