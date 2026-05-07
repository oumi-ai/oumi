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
    raise NotImplementedError


def rewrite_readme_news(readme: str, new_item: str) -> str:
    """Prepend new_item, keep 12 visible, wrap rest under <details>."""
    raise NotImplementedError


def is_release_in_news(readme: str, tag: str) -> bool:
    """Return True if tag already appears in the News section."""
    raise NotImplementedError


def format_news_item(tag: str, url: str, published_at: str, summary: str) -> str:
    """Return a formatted news bullet: `- [YYYY/MM] [Oumi {tag} released]({url}) {summary}`."""
    raise NotImplementedError


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
