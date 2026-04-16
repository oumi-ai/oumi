from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..dependencies import get_github
from ..github.client import GitHubClient

router = APIRouter()


@router.get("/commits", response_class=HTMLResponse)
async def commits(
    request: Request,
    author: str = "",
    page: int = 1,
    per_page: int = 25,
    gh: GitHubClient = Depends(get_github),
):
    templates = request.app.state.templates
    data = await gh.get_commits(author=author, page=page, per_page=per_page)

    ctx = {
        "active": "commits",
        "commits": data.items,
        "pagination": data,
        "author": author,
        "per_page": per_page,
    }

    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(request, "partials/commits_list.html", ctx)

    return templates.TemplateResponse(request, "pages/commits.html", ctx)
