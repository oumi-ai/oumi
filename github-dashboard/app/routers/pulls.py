from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..dependencies import get_github
from ..github.client import GitHubClient

router = APIRouter()


@router.get("/pulls", response_class=HTMLResponse)
async def pulls(
    request: Request,
    q: str = "",
    state: str = "open",
    page: int = 1,
    per_page: int = 25,
    gh: GitHubClient = Depends(get_github),
):
    templates = request.app.state.templates
    data = await gh.get_pulls(state=state, search=q, page=page, per_page=per_page)

    ctx = {
        "active": "pulls",
        "pulls": data.items,
        "pagination": data,
        "q": q,
        "state": state,
        "per_page": per_page,
    }

    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(request, "partials/pulls_list.html", ctx)

    return templates.TemplateResponse(request, "pages/pulls.html", ctx)
