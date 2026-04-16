from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .github import cache as gh_cache
from .github.client import GitHubClient, GitHubRateLimitError
from .routers import commits, contributors, issues, overview, pulls, releases, stargazers

logger = logging.getLogger(__name__)

templates: Jinja2Templates | None = None


def _relative_time(dt: datetime | str | None) -> str:
    if dt is None:
        return ""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    diff = now - dt
    seconds = int(diff.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        m = seconds // 60
        return f"{m} minute{'s' if m != 1 else ''} ago"
    if seconds < 86400:
        h = seconds // 3600
        return f"{h} hour{'s' if h != 1 else ''} ago"
    if seconds < 86400 * 30:
        d = seconds // 86400
        return f"{d} day{'s' if d != 1 else ''} ago"
    if seconds < 86400 * 365:
        mo = seconds // (86400 * 30)
        return f"{mo} month{'s' if mo != 1 else ''} ago"
    y = seconds // (86400 * 365)
    return f"{y} year{'s' if y != 1 else ''} ago"


def _format_number(n: int | None) -> str:
    if n is None:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _format_bytes(n: int) -> str:
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


@asynccontextmanager
async def lifespan(app: FastAPI):
    gh_cache.init_cache(settings.cache_dir)
    client = GitHubClient(settings)
    app.state.github = client
    try:
        await client.get_repo_stats()
    except Exception as e:
        logger.warning("Failed to warm cache on startup: %s", e)
    yield
    await client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="oumi GitHub Dashboard",
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory="static"), name="static")

    global templates
    templates = Jinja2Templates(directory="templates")
    templates.env.filters["relative_time"] = _relative_time
    templates.env.filters["format_number"] = _format_number
    templates.env.filters["format_bytes"] = _format_bytes
    templates.env.globals["repo_name"] = settings.github_repo
    templates.env.globals["current_year"] = datetime.now().year

    app.state.templates = templates

    app.include_router(overview.router)
    app.include_router(issues.router)
    app.include_router(pulls.router)
    app.include_router(contributors.router)
    app.include_router(commits.router)
    app.include_router(releases.router)
    app.include_router(stargazers.router)

    @app.exception_handler(GitHubRateLimitError)
    async def rate_limit_handler(request: FastAPIRequest, exc: GitHubRateLimitError):
        msg = (
            f"GitHub rate limit exceeded. Resets at {exc.reset_at.strftime('%H:%M')}. "
            "Add GITHUB_TOKEN to .env for 5,000 req/hr."
        )
        return HTMLResponse(
            f'<html><body style="font-family:system-ui;padding:2rem;color:#92400e;background:#fffbeb">'
            f'<h2>Rate limit exceeded</h2><p>{msg}</p></body></html>',
            status_code=429,
        )

    return app


app = create_app()
