"""GitHub Action script: adds a news item to README.md for a new release."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import requests

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "")
INPUT_TAG = os.environ.get("INPUT_TAG", "")

NEWS_HEADER = "## 🔥 News"
VISIBLE_ITEM_LIMIT = 12
OLDER_ITEMS_SUMMARY = "Older items"


def parse_news_items(readme: str) -> list[str]:
    """Return all news bullet lines (visible + inside <details>), in order."""
    lines = readme.splitlines()
    # Find the News section
    try:
        news_idx = next(i for i, l in enumerate(lines) if l.strip() == NEWS_HEADER)
    except StopIteration:
        raise ValueError(f"Could not find '{NEWS_HEADER}' section in README")

    # Collect lines until the next ## section
    bullets: list[str] = []
    for line in lines[news_idx + 1:]:
        if re.match(r"^## ", line):
            break
        if line.startswith("- ["):
            bullets.append(line)
    return bullets


def rewrite_readme_news(readme: str, new_item: str) -> str:
    """Prepend new_item, keep 12 visible, wrap rest under <details>."""
    lines = readme.splitlines(keepends=True)

    # Locate News section boundaries
    news_idx = next(
        (i for i, l in enumerate(lines) if l.strip() == NEWS_HEADER), None
    )
    if news_idx is None:
        raise ValueError(f"Could not find '{NEWS_HEADER}' section in README")

    # Find the line index of the next ## header after the News section
    next_section_idx = next(
        (i for i, l in enumerate(lines) if i > news_idx and re.match(r"^## ", l)),
        len(lines),
    )

    # Collect all existing bullets from the news section (visible + in <details>)
    existing_bullets: list[str] = []
    for line in lines[news_idx + 1 : next_section_idx]:
        if line.startswith("- ["):
            existing_bullets.append(line.rstrip("\r\n"))

    # Build the new ordered list (new item first)
    all_items = [new_item] + existing_bullets

    # Split into visible and older
    visible = all_items[:VISIBLE_ITEM_LIMIT]
    older = all_items[VISIBLE_ITEM_LIMIT:]

    # Reconstruct the News section
    new_section_lines: list[str] = [NEWS_HEADER + "\n", "\n"]
    for item in visible:
        new_section_lines.append(item + "\n")
    new_section_lines.append("\n")
    if older:
        new_section_lines.append("<details>\n")
        new_section_lines.append(f"<summary>{OLDER_ITEMS_SUMMARY}</summary>\n")
        new_section_lines.append("\n")
        for item in older:
            new_section_lines.append(item + "\n")
        new_section_lines.append("\n")
        new_section_lines.append("</details>\n")
        new_section_lines.append("\n")

    return "".join(lines[:news_idx] + new_section_lines + lines[next_section_idx:])


def is_release_in_news(readme: str, tag: str) -> bool:
    """Return True if tag already appears in the News section."""
    lines = readme.splitlines()
    try:
        news_idx = next(i for i, l in enumerate(lines) if l.strip() == NEWS_HEADER)
    except StopIteration:
        return False
    for line in lines[news_idx + 1:]:
        if re.match(r"^## ", line):
            break
        if tag in line:
            return True
    return False


def format_news_item(tag: str, url: str, published_at: str, summary: str) -> str:
    """Return a formatted news bullet: `- [YYYY/MM] [Oumi {tag} released]({url}) {summary}`."""
    summary = summary.strip()
    dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    date_str = dt.strftime("%Y/%m")
    base = f"- [{date_str}] [Oumi {tag} released]({url})"
    return f"{base} {summary}".rstrip() if summary else base


def get_release_info(repo: str, tag: str) -> dict:
    """Fetch release metadata from GitHub API. tag='' fetches latest."""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    if tag:
        url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    else:
        url = f"https://api.github.com/repos/{repo}/releases/latest"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def has_open_pr_for_tag(repo: str, tag: str) -> bool:
    """Return True if an open PR title already references this release tag."""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/{repo}/pulls"
    # Fetches only the first page (max 100). Sufficient for this repo in practice.
    resp = requests.get(url, headers=headers, params={"state": "open", "per_page": 100})
    resp.raise_for_status()
    prs = resp.json()
    return any(tag in pr.get("title", "") for pr in prs)


def generate_summary(release_body: str, tag: str) -> str:
    """Call Anthropic API and return a short 'with X, Y, and Z' summary."""
    prompt = (
        f"You are writing a one-line summary for the {tag} release of Oumi "
        "(an open-source foundation model platform) to appear in the project README.\n\n"
        "Release notes:\n"
        f"{release_body}\n\n"
        "Write a concise phrase (max 120 characters) starting with 'with' that highlights "
        "the 2-4 most important new features or changes. "
        "Example: 'with Python 3.13 support, `oumi analyze` CLI command, and TRL 0.26+ support'. "
        "Output only the phrase, nothing else. Do not end with a period."
    )
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"].strip().rstrip(".")
    return text


def create_pr(repo: str, branch: str, tag: str, news_item: str, release_body: str) -> str:
    """Push branch and open a PR. Returns the PR URL."""
    subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
    subprocess.run(["git", "checkout", "-b", branch], check=True)
    subprocess.run(["git", "add", "README.md"], check=True)
    subprocess.run(["git", "commit", "-m", f"chore: add news item for {tag} release"], check=True)
    subprocess.run(["git", "push", "origin", branch], check=True)

    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    pr_body = (
        f"## Proposed news item\n\n```\n{news_item}\n```\n\n"
        "Please review and edit the news item text above if needed.\n\n"
        f"---\n\n<details>\n<summary>Full release notes for {tag}</summary>\n\n"
        f"{release_body}\n\n</details>"
    )
    resp = requests.post(
        f"https://api.github.com/repos/{repo}/pulls",
        headers=headers,
        json={
            "title": f"chore: add README news item for {tag}",
            "body": pr_body,
            "head": branch,
            "base": "main",
        },
    )
    resp.raise_for_status()
    return resp.json()["html_url"]


def main() -> None:
    repo = GITHUB_REPOSITORY
    if not repo:
        print("ERROR: GITHUB_REPOSITORY not set", file=sys.stderr)
        sys.exit(1)
    if not GITHUB_TOKEN:
        print("ERROR: GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching release info for tag='{INPUT_TAG}' in {repo}...")
    release = get_release_info(repo, INPUT_TAG)
    tag = release["tag_name"]
    url = release["html_url"]
    published_at = release["published_at"]
    body = release.get("body") or ""
    print(f"Release: {tag} published at {published_at}")

    readme_path = "README.md"
    with open(readme_path, encoding="utf-8") as f:
        readme = f.read()

    if is_release_in_news(readme, tag):
        print(f"News item for {tag} already exists in README. Nothing to do.")
        return

    if has_open_pr_for_tag(repo, tag):
        print(f"An open PR for {tag} news item already exists. Nothing to do.")
        return

    print("Generating news summary via Claude...")
    summary = generate_summary(body, tag)
    print(f"Summary: {summary}")

    news_item = format_news_item(tag=tag, url=url, published_at=published_at, summary=summary)
    print(f"News item: {news_item}")

    updated_readme = rewrite_readme_news(readme, news_item)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(updated_readme)
    print("README.md updated.")

    branch = f"chore/news-{tag}"
    pr_url = create_pr(repo=repo, branch=branch, tag=tag, news_item=news_item, release_body=body)
    print(f"PR created: {pr_url}")


if __name__ == "__main__":
    main()
