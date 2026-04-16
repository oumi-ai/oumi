from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class RepoStats(BaseModel):
    full_name: str
    description: str | None = None
    homepage: str | None = None
    language: str | None = None
    stargazers_count: int = 0
    forks_count: int = 0
    open_issues_count: int = 0
    watchers_count: int = 0
    subscribers_count: int = 0
    topics: list[str] = []
    pushed_at: datetime | None = None
    created_at: datetime | None = None
    license_name: str | None = None
    html_url: str = ""

    @classmethod
    def from_api(cls, data: dict) -> "RepoStats":
        return cls(
            full_name=data["full_name"],
            description=data.get("description"),
            homepage=data.get("homepage"),
            language=data.get("language"),
            stargazers_count=data.get("stargazers_count", 0),
            forks_count=data.get("forks_count", 0),
            open_issues_count=data.get("open_issues_count", 0),
            watchers_count=data.get("watchers_count", 0),
            subscribers_count=data.get("subscribers_count", 0),
            topics=data.get("topics", []),
            pushed_at=data.get("pushed_at"),
            created_at=data.get("created_at"),
            license_name=data.get("license", {}).get("name") if data.get("license") else None,
            html_url=data.get("html_url", ""),
        )


class Label(BaseModel):
    name: str
    color: str
    description: str | None = None


class UserRef(BaseModel):
    login: str
    avatar_url: str = ""
    html_url: str = ""


class Issue(BaseModel):
    number: int
    title: str
    state: str
    labels: list[Label] = []
    user: UserRef | None = None
    created_at: datetime
    updated_at: datetime
    html_url: str
    comments: int = 0
    body: str | None = None

    @classmethod
    def from_api(cls, data: dict) -> "Issue":
        return cls(
            number=data["number"],
            title=data["title"],
            state=data["state"],
            labels=[Label(name=l["name"], color=l["color"], description=l.get("description")) for l in data.get("labels", [])],
            user=UserRef(login=data["user"]["login"], avatar_url=data["user"].get("avatar_url", ""), html_url=data["user"].get("html_url", "")) if data.get("user") else None,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            html_url=data["html_url"],
            comments=data.get("comments", 0),
            body=data.get("body"),
        )


class PullRequest(BaseModel):
    number: int
    title: str
    state: str
    draft: bool = False
    merged_at: datetime | None = None
    labels: list[Label] = []
    user: UserRef | None = None
    created_at: datetime
    updated_at: datetime
    html_url: str
    comments: int = 0
    head_ref: str = ""
    base_ref: str = ""

    @classmethod
    def from_api(cls, data: dict) -> "PullRequest":
        return cls(
            number=data["number"],
            title=data["title"],
            state=data["state"],
            draft=data.get("draft", False),
            merged_at=data.get("merged_at"),
            labels=[Label(name=l["name"], color=l["color"], description=l.get("description")) for l in data.get("labels", [])],
            user=UserRef(login=data["user"]["login"], avatar_url=data["user"].get("avatar_url", ""), html_url=data["user"].get("html_url", "")) if data.get("user") else None,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            html_url=data["html_url"],
            comments=data.get("comments", 0),
            head_ref=data.get("head", {}).get("ref", ""),
            base_ref=data.get("base", {}).get("ref", ""),
        )


class Contributor(BaseModel):
    login: str
    avatar_url: str = ""
    html_url: str = ""
    contributions: int = 0

    @classmethod
    def from_api(cls, data: dict) -> "Contributor":
        return cls(
            login=data["login"],
            avatar_url=data.get("avatar_url", ""),
            html_url=data.get("html_url", ""),
            contributions=data.get("contributions", 0),
        )


class CommitAuthor(BaseModel):
    name: str
    date: datetime | None = None


class Commit(BaseModel):
    sha: str
    message: str
    author_name: str = ""
    author_date: datetime | None = None
    author_login: str | None = None
    author_avatar: str | None = None
    html_url: str = ""

    @property
    def short_sha(self) -> str:
        return self.sha[:7]

    @property
    def short_message(self) -> str:
        return self.message.split("\n")[0][:80]

    @classmethod
    def from_api(cls, data: dict) -> "Commit":
        commit = data.get("commit", {})
        author_data = data.get("author")
        return cls(
            sha=data["sha"],
            message=commit.get("message", ""),
            author_name=commit.get("author", {}).get("name", ""),
            author_date=commit.get("author", {}).get("date"),
            author_login=author_data["login"] if author_data else None,
            author_avatar=author_data.get("avatar_url") if author_data else None,
            html_url=data.get("html_url", ""),
        )


class ReleaseAsset(BaseModel):
    name: str
    download_count: int = 0
    size: int = 0


class Release(BaseModel):
    id: int
    tag_name: str
    name: str | None = None
    published_at: datetime | None = None
    html_url: str = ""
    body: str | None = None
    prerelease: bool = False
    draft: bool = False
    assets: list[ReleaseAsset] = []

    @property
    def total_downloads(self) -> int:
        return sum(a.download_count for a in self.assets)

    @classmethod
    def from_api(cls, data: dict) -> "Release":
        return cls(
            id=data["id"],
            tag_name=data["tag_name"],
            name=data.get("name"),
            published_at=data.get("published_at"),
            html_url=data.get("html_url", ""),
            body=data.get("body"),
            prerelease=data.get("prerelease", False),
            draft=data.get("draft", False),
            assets=[
                ReleaseAsset(name=a["name"], download_count=a.get("download_count", 0), size=a.get("size", 0))
                for a in data.get("assets", [])
            ],
        )


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total_count: int | None = None
    page: int = 1
    per_page: int = 25
    total_pages: int = 1

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages

    @property
    def has_prev(self) -> bool:
        return self.page > 1
