from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..dependencies import get_github
from ..github.client import GitHubClient

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def overview(request: Request, gh: GitHubClient = Depends(get_github)):
    templates = request.app.state.templates

    stats, contributors, commits, releases = await _gather(gh)

    return templates.TemplateResponse(
        request,
        "pages/overview.html",
        {
            "active": "overview",
            "stats": stats,
            "top_contributors": contributors.items[:8],
            "recent_commits": commits.items[:6],
            "recent_releases": releases.items[:3],
        },
    )


async def _gather(gh: GitHubClient):
    import asyncio
    return await asyncio.gather(
        gh.get_repo_stats(),
        gh.get_contributors(per_page=8),
        gh.get_commits(per_page=6),
        gh.get_releases(per_page=3),
    )
