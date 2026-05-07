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
            existing_bullets.append(line.rstrip("\n"))

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
    dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    date_str = dt.strftime("%Y/%m")
    base = f"- [{date_str}] [Oumi {tag} released]({url})"
    return f"{base} {summary}".rstrip() if summary else base


def get_release_info(repo: str, tag: str) -> dict:
    """Fetch release metadata from GitHub API. tag='' fetches latest."""
    raise NotImplementedError


def has_open_pr_for_tag(repo: str, tag: str) -> bool:
    """Return True if an open PR title already references this release tag."""
    raise NotImplementedError


def generate_summary(release_body: str, tag: str) -> str:
    """Call Anthropic API and return a short 'with X, Y, and Z' summary."""
    raise NotImplementedError


def create_pr(repo: str, branch: str, tag: str, news_item: str, release_body: str) -> str:
    """Push branch and open a PR. Returns the PR URL."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
