from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..dependencies import get_github
from ..github.client import GitHubClient

router = APIRouter()


@router.get("/contributors", response_class=HTMLResponse)
async def contributors(
    request: Request,
    page: int = 1,
    per_page: int = 30,
    gh: GitHubClient = Depends(get_github),
):
    templates = request.app.state.templates
    data = await gh.get_contributors(page=page, per_page=per_page)

    ctx = {
        "active": "contributors",
        "contributors": data.items,
        "pagination": data,
        "per_page": per_page,
    }

    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(request, "partials/contributors_grid.html", ctx)

    return templates.TemplateResponse(request, "pages/contributors.html", ctx)
