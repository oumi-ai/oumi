from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from ..config import Settings
from .models import (
    Commit,
    Contributor,
    Issue,
    Label,
    PaginatedResponse,
    PullRequest,
    Release,
    RepoStats,
)


class GitHubRateLimitError(Exception):
    def __init__(self, reset_at: datetime):
        self.reset_at = reset_at
        super().__init__(f"GitHub rate limit exceeded. Resets at {reset_at}")


@dataclass
class RateLimitState:
    remaining: int = 5000
    limit: int = 5000
    reset_at: datetime | None = None

    def update(self, headers: httpx.Headers) -> None:
        if "X-RateLimit-Remaining" in headers:
            self.remaining = int(headers["X-RateLimit-Remaining"])
        if "X-RateLimit-Limit" in headers:
            self.limit = int(headers["X-RateLimit-Limit"])
        if "X-RateLimit-Reset" in headers:
            self.reset_at = datetime.fromtimestamp(int(headers["X-RateLimit-Reset"]))


def _parse_last_page(link_header: str | None) -> int | None:
    if not link_header:
        return None
    match = re.search(r'<[^>]+[?&]page=(\d+)[^>]*>;\s*rel="last"', link_header)
    if match:
        return int(match.group(1))
    return None


class GitHubClient:
    BASE_URL = "https://api.github.com"

    def __init__(self, settings: Settings):
        self.repo = settings.github_repo
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if settings.github_token:
            headers["Authorization"] = f"Bearer {settings.github_token}"

        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers=headers,
            timeout=15.0,
            follow_redirects=True,
        )
        self.rate_limit = RateLimitState()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict | None = None, accept: str | None = None) -> tuple[Any, httpx.Headers]:
        extra_headers = {"Accept": accept} if accept else {}
        response = await self._client.get(path, params=params, headers=extra_headers)
        self.rate_limit.update(response.headers)

        if response.status_code == 403 and self.rate_limit.remaining == 0:
            raise GitHubRateLimitError(self.rate_limit.reset_at or datetime.now())

        response.raise_for_status()
        return response.json(), response.headers

    def _paginated(self, items: list, headers: httpx.Headers, page: int, per_page: int, total_count: int | None = None) -> PaginatedResponse:
        last = _parse_last_page(headers.get("link"))
        total_pages = last if last else (1 if not items else page)
        return PaginatedResponse(
            items=items,
            total_count=total_count,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
        )

    async def get_repo_stats(self) -> RepoStats:
        data, _ = await self._get(f"/repos/{self.repo}")
        return RepoStats.from_api(data)

    async def get_labels(self) -> list[Label]:
        data, _ = await self._get(f"/repos/{self.repo}/labels", {"per_page": 100})
        return [Label(name=l["name"], color=l["color"], description=l.get("description")) for l in data]

    async def get_issues(
        self,
        state: str = "open",
        labels: list[str] | None = None,
        search: str = "",
        page: int = 1,
        per_page: int = 25,
    ) -> PaginatedResponse[Issue]:
        if search:
            q = f"repo:{self.repo} is:issue {search}"
            if state != "all":
                q += f" is:{state}"
            data, headers = await self._get("/search/issues", {"q": q, "page": page, "per_page": per_page})
            items = [Issue.from_api(i) for i in data["items"] if "pull_request" not in i]
            return PaginatedResponse(
                items=items,
                total_count=data.get("total_count"),
                page=page,
                per_page=per_page,
                total_pages=max(1, -(-data.get("total_count", 0) // per_page)),
            )

        params: dict[str, Any] = {
            "state": state,
            "page": page,
            "per_page": per_page,
        }
        if labels:
            params["labels"] = ",".join(labels)

        data, headers = await self._get(f"/repos/{self.repo}/issues", params)
        # Filter out PRs (they appear in /issues endpoint)
        items = [Issue.from_api(i) for i in data if "pull_request" not in i]
        return self._paginated(items, headers, page, per_page)

    async def get_pulls(
        self,
        state: str = "open",
        search: str = "",
        page: int = 1,
        per_page: int = 25,
    ) -> PaginatedResponse[PullRequest]:
        if search:
            q = f"repo:{self.repo} is:pr {search}"
            if state == "merged":
                q += " is:merged"
            elif state != "all":
                q += f" is:{state}"
            data, headers = await self._get("/search/issues", {"q": q, "page": page, "per_page": per_page})
            items = [PullRequest.from_api(i) for i in data["items"] if "pull_request" in i]
            return PaginatedResponse(
                items=items,
                total_count=data.get("total_count"),
                page=page,
                per_page=per_page,
                total_pages=max(1, -(-data.get("total_count", 0) // per_page)),
            )

        api_state = "closed" if state == "merged" else state
        data, headers = await self._get(
            f"/repos/{self.repo}/pulls",
            {"state": api_state, "page": page, "per_page": per_page},
        )
        items = [PullRequest.from_api(i) for i in data]
        if state == "merged":
            items = [p for p in items if p.merged_at is not None]
        return self._paginated(items, headers, page, per_page)

    async def get_contributors(self, page: int = 1, per_page: int = 30) -> PaginatedResponse[Contributor]:
        data, headers = await self._get(
            f"/repos/{self.repo}/contributors",
            {"page": page, "per_page": per_page},
        )
        items = [Contributor.from_api(c) for c in data]
        return self._paginated(items, headers, page, per_page)

    async def get_commits(
        self,
        author: str = "",
        page: int = 1,
        per_page: int = 25,
    ) -> PaginatedResponse[Commit]:
        params: dict[str, Any] = {"page": page, "per_page": per_page}
        if author:
            params["author"] = author
        data, headers = await self._get(f"/repos/{self.repo}/commits", params)
        items = [Commit.from_api(c) for c in data]
        return self._paginated(items, headers, page, per_page)

    async def get_releases(self, page: int = 1, per_page: int = 10) -> PaginatedResponse[Release]:
        data, headers = await self._get(
            f"/repos/{self.repo}/releases",
            {"page": page, "per_page": per_page},
        )
        items = [Release.from_api(r) for r in data]
        return self._paginated(items, headers, page, per_page)

    async def get_open_prs_count(self) -> int:
        data, _ = await self._get(f"/repos/{self.repo}/pulls", {"state": "open", "per_page": 1})
        return len(data)  # approximate; accurate enough for stat card

    # ---- Stargazer methods ----

    _STAR_ACCEPT = "application/vnd.github.star+json"

    async def get_all_star_timestamps(self) -> list[dict]:
        """Fetch every star event with its timestamp. Cached 1h on disk."""
        import asyncio
        from ..github.cache import _DISK_CACHE

        cache_key = f"star_timestamps:{self.repo}"
        if _DISK_CACHE is not None and cache_key in _DISK_CACHE:
            return _DISK_CACHE[cache_key]

        # Get page 1 to discover total pages
        data, headers = await self._get(
            f"/repos/{self.repo}/stargazers",
            {"per_page": 100, "page": 1},
            accept=self._STAR_ACCEPT,
        )
        last_page = _parse_last_page(headers.get("link")) or 1

        events: list[dict] = [
            {"starred_at": item["starred_at"], "login": item["user"]["login"]}
            for item in data
        ]

        # Fetch remaining pages in batches of 15 concurrent requests
        for batch_start in range(2, last_page + 1, 15):
            batch_pages = range(batch_start, min(batch_start + 15, last_page + 1))
            results = await asyncio.gather(*[
                self._get(
                    f"/repos/{self.repo}/stargazers",
                    {"per_page": 100, "page": p},
                    accept=self._STAR_ACCEPT,
                )
                for p in batch_pages
            ])
            for page_data, _ in results:
                for item in page_data:
                    events.append({"starred_at": item["starred_at"], "login": item["user"]["login"]})

        if _DISK_CACHE is not None:
            _DISK_CACHE.set(cache_key, events, expire=3600)

        return events

    async def get_recent_enriched_stargazers(self, n: int = 200) -> list[dict]:
        """Return the N most recent stargazers with full GitHub profiles. Cached 1h."""
        import asyncio
        from ..github.cache import _DISK_CACHE

        cache_key = f"enriched_stargazers:{self.repo}:{n}"
        if _DISK_CACHE is not None and cache_key in _DISK_CACHE:
            return _DISK_CACHE[cache_key]

        # Find total pages so we can grab the MOST RECENT pages
        _, headers = await self._get(
            f"/repos/{self.repo}/stargazers",
            {"per_page": 100, "page": 1},
            accept=self._STAR_ACCEPT,
        )
        last_page = _parse_last_page(headers.get("link")) or 1
        pages_needed = max(1, -(-n // 100))  # ceiling division
        start_page = max(1, last_page - pages_needed + 1)

        raw: list[dict] = []
        results = await asyncio.gather(*[
            self._get(
                f"/repos/{self.repo}/stargazers",
                {"per_page": 100, "page": p},
                accept=self._STAR_ACCEPT,
            )
            for p in range(start_page, last_page + 1)
        ])
        for page_data, _ in results:
            for item in page_data:
                raw.append({
                    "starred_at": item["starred_at"],
                    "login": item["user"]["login"],
                    "avatar_url": item["user"].get("avatar_url", ""),
                })

        # Take most recent n, sorted descending
        raw.sort(key=lambda x: x["starred_at"], reverse=True)
        sample = raw[:n]

        # Enrich with full user profiles (10 concurrent)
        sem = asyncio.Semaphore(10)

        async def enrich(item: dict) -> dict:
            async with sem:
                try:
                    profile, _ = await self._get(f"/users/{item['login']}")
                    return {
                        **item,
                        "name": profile.get("name"),
                        "company": profile.get("company"),
                        "location": profile.get("location"),
                        "bio": profile.get("bio"),
                        "followers": profile.get("followers", 0),
                        "public_repos": profile.get("public_repos", 0),
                        "html_url": profile.get("html_url", ""),
                        "twitter_username": profile.get("twitter_username"),
                        "blog": profile.get("blog"),
                    }
                except Exception:
                    return {**item, "followers": 0, "public_repos": 0, "html_url": f"https://github.com/{item['login']}"}

        enriched = list(await asyncio.gather(*[enrich(s) for s in sample]))
        enriched.sort(key=lambda x: x.get("starred_at", ""), reverse=True)

        if _DISK_CACHE is not None:
            _DISK_CACHE.set(cache_key, enriched, expire=3600)

        return enriched
