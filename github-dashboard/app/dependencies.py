from fastapi import Request

from .github.client import GitHubClient


async def get_github(request: Request) -> GitHubClient:
    return request.app.state.github
