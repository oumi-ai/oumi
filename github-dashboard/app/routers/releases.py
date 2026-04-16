from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..dependencies import get_github
from ..github.client import GitHubClient

router = APIRouter()


@router.get("/releases", response_class=HTMLResponse)
async def releases(
    request: Request,
    page: int = 1,
    per_page: int = 10,
    gh: GitHubClient = Depends(get_github),
):
    templates = request.app.state.templates
    data = await gh.get_releases(page=page, per_page=per_page)

    ctx = {
        "active": "releases",
        "releases": data.items,
        "pagination": data,
        "per_page": per_page,
    }

    return templates.TemplateResponse(request, "pages/releases.html", ctx)
