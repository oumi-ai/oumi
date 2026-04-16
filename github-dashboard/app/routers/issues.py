from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..dependencies import get_github
from ..github.client import GitHubClient

router = APIRouter()


@router.get("/issues", response_class=HTMLResponse)
async def issues(
    request: Request,
    q: str = "",
    state: str = "open",
    label: str = "",
    page: int = 1,
    per_page: int = 25,
    gh: GitHubClient = Depends(get_github),
):
    templates = request.app.state.templates
    data = await gh.get_issues(
        state=state,
        labels=[label] if label else [],
        search=q,
        page=page,
        per_page=per_page,
    )

    ctx = {
        "active": "issues",
        "issues": data.items,
        "pagination": data,
        "q": q,
        "state": state,
        "label": label,
        "per_page": per_page,
    }

    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(request, "partials/issues_list.html", ctx)

    ctx["labels"] = await gh.get_labels()
    return templates.TemplateResponse(request, "pages/issues.html", ctx)
