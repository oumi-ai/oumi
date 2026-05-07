# README News Auto-Update GitHub Action — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A GitHub Action that fires on release publication (or manual trigger) and opens a PR adding a Claude-generated news item to `README.md`, skipping if the item or PR already exists.

**Architecture:** A Python script (`.github/scripts/update_readme_news.py`) handles all logic — README parsing, Anthropic API call, GitHub API calls, git branch creation, and PR submission. A thin workflow YAML wires up triggers, permissions, and env vars, then calls the script.

**Tech Stack:** Python 3.11 stdlib + `requests` + `anthropic` SDK (already a project dependency), GitHub Actions, `GITHUB_TOKEN`, `ANTHROPIC_API_KEY` secret.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `.github/scripts/update_readme_news.py` | Create | All script logic: parse README, call Anthropic, check duplicates, commit, create PR |
| `.github/workflows/update_readme_news.yaml` | Create | Triggers, permissions, env wiring, calls script |
| `tests/unit/test_update_readme_news.py` | Create | Unit tests for all pure functions and mocked network calls |

---

## README News Section — Current Structure (reference)

```
## 🔥 News

- [2026/03] item ...
- [2026/03] item ...
...  (12 visible bullets total currently)

<details>
<summary>Older updates</summary>

- [2025/09] older item ...
...

</details>

## 🔎 About
```

The script must handle this structure: collect all bullets (visible + inside `<details>`), prepend new item, keep first 12 visible, wrap 13+ in `<details><summary>Older items</summary>`.

---

### Task 1: Scaffold the script file and write tests for pure README functions

**Files:**
- Create: `.github/scripts/update_readme_news.py`
- Create: `tests/unit/test_update_readme_news.py`

- [ ] **Step 1: Create the script file with function stubs only**

Create `.github/scripts/update_readme_news.py`:

```python
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
```

- [ ] **Step 2: Write tests for `parse_news_items`**

Create `tests/unit/test_update_readme_news.py`:

```python
"""Tests for .github/scripts/update_readme_news.py"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".github", "scripts"))

import pytest
from update_readme_news import (
    parse_news_items,
    rewrite_readme_news,
    is_release_in_news,
    format_news_item,
)

SAMPLE_README = """\
## 🔥 News

- [2026/03] Item A
- [2026/02] Item B
- [2025/12] Item C

<details>
<summary>Older updates</summary>

- [2025/09] Item D
- [2025/08] Item E

</details>

## 🔎 About

Some text.
"""

def test_parse_news_items_returns_all_bullets():
    items = parse_news_items(SAMPLE_README)
    assert items == [
        "- [2026/03] Item A",
        "- [2026/02] Item B",
        "- [2025/12] Item C",
        "- [2025/09] Item D",
        "- [2025/08] Item E",
    ]


def test_parse_news_items_no_details_block():
    readme = "## 🔥 News\n\n- [2026/01] Only item\n\n## Next\n"
    items = parse_news_items(readme)
    assert items == ["- [2026/01] Only item"]


def test_parse_news_items_empty_section():
    readme = "## 🔥 News\n\n## Next\n"
    items = parse_news_items(readme)
    assert items == []
```

- [ ] **Step 3: Run the tests and confirm they fail with NotImplementedError**

```bash
cd /Users/stefanwebb/Code/Python/oumi
python -m pytest tests/unit/test_update_readme_news.py::test_parse_news_items_returns_all_bullets -v
```

Expected: `FAILED` with `NotImplementedError`.

- [ ] **Step 4: Write tests for `is_release_in_news`**

Append to `tests/unit/test_update_readme_news.py`:

```python
def test_is_release_in_news_true():
    readme = "## 🔥 News\n\n- [2025/12] [Oumi v0.6.0 released](https://github.com/oumi-ai/oumi/releases/tag/v0.6.0) with stuff\n\n## Next\n"
    assert is_release_in_news(readme, "v0.6.0") is True


def test_is_release_in_news_false():
    readme = "## 🔥 News\n\n- [2025/12] Some other item\n\n## Next\n"
    assert is_release_in_news(readme, "v0.6.0") is False


def test_is_release_in_news_tag_in_older_block():
    readme = (
        "## 🔥 News\n\n- [2026/01] New item\n\n"
        "<details>\n<summary>Older updates</summary>\n\n"
        "- [2025/09] [Oumi v0.4.0 released](https://github.com/oumi-ai/oumi/releases/tag/v0.4.0)\n\n"
        "</details>\n\n## Next\n"
    )
    assert is_release_in_news(readme, "v0.4.0") is True
```

- [ ] **Step 5: Write tests for `format_news_item`**

Append to `tests/unit/test_update_readme_news.py`:

```python
def test_format_news_item_with_summary():
    item = format_news_item(
        tag="v0.8",
        url="https://github.com/oumi-ai/oumi/releases/tag/v0.8",
        published_at="2026-05-01T12:00:00Z",
        summary="with new feature X and Y",
    )
    assert item == "- [2026/05] [Oumi v0.8 released](https://github.com/oumi-ai/oumi/releases/tag/v0.8) with new feature X and Y"


def test_format_news_item_empty_summary():
    item = format_news_item(
        tag="v0.9",
        url="https://github.com/oumi-ai/oumi/releases/tag/v0.9",
        published_at="2026-06-15T00:00:00Z",
        summary="",
    )
    assert item == "- [2026/06] [Oumi v0.9 released](https://github.com/oumi-ai/oumi/releases/tag/v0.9)"
```

- [ ] **Step 6: Write tests for `rewrite_readme_news`**

Append to `tests/unit/test_update_readme_news.py`:

```python
def _make_readme(visible_count: int, older_count: int = 0) -> str:
    """Build a synthetic README with given item counts."""
    lines = ["## 🔥 News", ""]
    for i in range(visible_count):
        lines.append(f"- [2026/{visible_count - i:02d}] Item {i + 1}")
    lines.append("")
    if older_count:
        lines += ["<details>", "<summary>Older updates</summary>", ""]
        for j in range(older_count):
            lines.append(f"- [2025/{older_count - j:02d}] Older {j + 1}")
        lines += ["", "</details>", ""]
    lines += ["## 🔎 About", "", "Some text.", ""]
    return "\n".join(lines)


def test_rewrite_prepends_new_item():
    readme = _make_readme(visible_count=3)
    result = rewrite_readme_news(readme, "- [2026/05] New Item")
    items = parse_news_items(result)
    assert items[0] == "- [2026/05] New Item"
    assert len(items) == 4


def test_rewrite_keeps_12_visible_wraps_rest():
    # Start with 12 visible + 5 older = 17 total; adding 1 → 18 total → 12 visible + 6 older
    readme = _make_readme(visible_count=12, older_count=5)
    result = rewrite_readme_news(readme, "- [2026/06] Brand New")
    # Exactly 12 bullets should appear before <details>
    news_start = result.index("## 🔥 News")
    details_start = result.index("<details>", news_start)
    visible_section = result[news_start:details_start]
    visible_bullets = [l for l in visible_section.splitlines() if l.startswith("- [")]
    assert len(visible_bullets) == 12
    # Remaining 6 should be inside details
    items = parse_news_items(result)
    assert len(items) == 18
    assert items[0] == "- [2026/06] Brand New"


def test_rewrite_under_12_no_details_block():
    readme = _make_readme(visible_count=5)
    result = rewrite_readme_news(readme, "- [2026/05] New Item")
    assert "<details>" not in result
    items = parse_news_items(result)
    assert len(items) == 6


def test_rewrite_normalizes_older_summary_label():
    readme = _make_readme(visible_count=12, older_count=3)
    result = rewrite_readme_news(readme, "- [2026/06] Brand New")
    assert "<summary>Older items</summary>" in result


def test_rewrite_preserves_rest_of_readme():
    readme = _make_readme(visible_count=3)
    result = rewrite_readme_news(readme, "- [2026/05] New Item")
    assert "## 🔎 About" in result
    assert "Some text." in result
```

- [ ] **Step 7: Commit scaffolding + tests**

```bash
git add .github/scripts/update_readme_news.py tests/unit/test_update_readme_news.py
git commit -m "test: add unit tests for README news script (all failing)"
```

---

### Task 2: Implement the pure README functions

**Files:**
- Modify: `.github/scripts/update_readme_news.py`

- [ ] **Step 1: Implement `parse_news_items`**

Replace the `parse_news_items` stub in `.github/scripts/update_readme_news.py`:

```python
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
    in_section = False
    for line in lines[news_idx + 1:]:
        if re.match(r"^## ", line):
            break
        if line.startswith("- ["):
            bullets.append(line)
    return bullets
```

- [ ] **Step 2: Run `parse_news_items` tests**

```bash
python -m pytest tests/unit/test_update_readme_news.py -k "parse_news_items" -v
```

Expected: 3 tests PASSED.

- [ ] **Step 3: Implement `is_release_in_news`**

Replace the `is_release_in_news` stub:

```python
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
```

- [ ] **Step 4: Run `is_release_in_news` tests**

```bash
python -m pytest tests/unit/test_update_readme_news.py -k "is_release_in_news" -v
```

Expected: 3 tests PASSED.

- [ ] **Step 5: Implement `format_news_item`**

Replace the `format_news_item` stub:

```python
def format_news_item(tag: str, url: str, published_at: str, summary: str) -> str:
    """Return a formatted news bullet."""
    dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    date_str = dt.strftime("%Y/%m")
    base = f"- [{date_str}] [Oumi {tag} released]({url})"
    return f"{base} {summary}".rstrip() if summary else base
```

- [ ] **Step 6: Run `format_news_item` tests**

```bash
python -m pytest tests/unit/test_update_readme_news.py -k "format_news_item" -v
```

Expected: 2 tests PASSED.

- [ ] **Step 7: Implement `rewrite_readme_news`**

Replace the `rewrite_readme_news` stub:

```python
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
```

- [ ] **Step 8: Run all pure function tests**

```bash
python -m pytest tests/unit/test_update_readme_news.py -v
```

Expected: all tests PASSED (no NotImplementedError failures remaining for implemented functions).

- [ ] **Step 9: Commit**

```bash
git add .github/scripts/update_readme_news.py
git commit -m "feat: implement README news parsing and rewriting functions"
```

---

### Task 3: Implement GitHub API functions and write their tests

**Files:**
- Modify: `.github/scripts/update_readme_news.py`
- Modify: `tests/unit/test_update_readme_news.py`

- [ ] **Step 1: Write tests for `get_release_info` and `has_open_pr_for_tag`**

Append to `tests/unit/test_update_readme_news.py`:

```python
from unittest.mock import patch, MagicMock
from update_readme_news import get_release_info, has_open_pr_for_tag


def _mock_response(json_data, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status = MagicMock()
    return mock


def test_get_release_info_by_tag():
    payload = {
        "tag_name": "v0.8",
        "html_url": "https://github.com/oumi-ai/oumi/releases/tag/v0.8",
        "body": "## What's new\n- Feature A\n- Feature B",
        "published_at": "2026-05-01T12:00:00Z",
    }
    with patch("update_readme_news.requests.get", return_value=_mock_response(payload)):
        info = get_release_info("oumi-ai/oumi", "v0.8")
    assert info["tag_name"] == "v0.8"
    assert info["html_url"] == "https://github.com/oumi-ai/oumi/releases/tag/v0.8"


def test_get_release_info_latest_when_no_tag():
    payload = {
        "tag_name": "v0.8",
        "html_url": "https://github.com/oumi-ai/oumi/releases/tag/v0.8",
        "body": "Latest release notes",
        "published_at": "2026-05-01T12:00:00Z",
    }
    with patch("update_readme_news.requests.get", return_value=_mock_response(payload)) as mock_get:
        info = get_release_info("oumi-ai/oumi", "")
    # Should have called the /releases/latest endpoint
    call_url = mock_get.call_args[0][0]
    assert "latest" in call_url
    assert info["tag_name"] == "v0.8"


def test_has_open_pr_for_tag_true():
    prs = [
        {"title": "chore: add README news item for v0.8", "state": "open"},
    ]
    with patch("update_readme_news.requests.get", return_value=_mock_response(prs)):
        assert has_open_pr_for_tag("oumi-ai/oumi", "v0.8") is True


def test_has_open_pr_for_tag_false():
    prs = [
        {"title": "fix: some unrelated fix", "state": "open"},
    ]
    with patch("update_readme_news.requests.get", return_value=_mock_response(prs)):
        assert has_open_pr_for_tag("oumi-ai/oumi", "v0.8") is False


def test_has_open_pr_for_tag_empty():
    with patch("update_readme_news.requests.get", return_value=_mock_response([])):
        assert has_open_pr_for_tag("oumi-ai/oumi", "v0.8") is False
```

- [ ] **Step 2: Run these tests to confirm they fail**

```bash
python -m pytest tests/unit/test_update_readme_news.py -k "get_release_info or has_open_pr" -v
```

Expected: FAILED with `NotImplementedError`.

- [ ] **Step 3: Implement `get_release_info`**

Replace the `get_release_info` stub in the script:

```python
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
```

- [ ] **Step 4: Implement `has_open_pr_for_tag`**

Replace the `has_open_pr_for_tag` stub:

```python
def has_open_pr_for_tag(repo: str, tag: str) -> bool:
    """Return True if an open PR title already references this release tag."""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/{repo}/pulls"
    resp = requests.get(url, headers=headers, params={"state": "open", "per_page": 100})
    resp.raise_for_status()
    prs = resp.json()
    return any(tag in pr.get("title", "") for pr in prs)
```

- [ ] **Step 5: Run GitHub API tests**

```bash
python -m pytest tests/unit/test_update_readme_news.py -k "get_release_info or has_open_pr" -v
```

Expected: 5 tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add .github/scripts/update_readme_news.py tests/unit/test_update_readme_news.py
git commit -m "feat: implement GitHub API functions for release lookup and PR dedup"
```

---

### Task 4: Implement `generate_summary`, `create_pr`, and `main`

**Files:**
- Modify: `.github/scripts/update_readme_news.py`
- Modify: `tests/unit/test_update_readme_news.py`

- [ ] **Step 1: Write test for `generate_summary`**

Append to `tests/unit/test_update_readme_news.py`:

```python
from update_readme_news import generate_summary


def test_generate_summary_returns_string():
    mock_response = _mock_response({
        "content": [{"text": "with feature A, feature B, and feature C"}]
    })
    with patch("update_readme_news.requests.post", return_value=mock_response):
        result = generate_summary("## What's new\n- Feature A\n- Feature B\n- Feature C", "v0.8")
    assert isinstance(result, str)
    assert len(result) > 0
    assert result.startswith("with ")


def test_generate_summary_strips_trailing_period():
    mock_response = _mock_response({
        "content": [{"text": "with feature A and feature B."}]
    })
    with patch("update_readme_news.requests.post", return_value=mock_response):
        result = generate_summary("notes", "v0.8")
    assert not result.endswith(".")
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/unit/test_update_readme_news.py -k "generate_summary" -v
```

Expected: FAILED with `NotImplementedError`.

- [ ] **Step 3: Implement `generate_summary`**

Replace the `generate_summary` stub:

```python
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
```

- [ ] **Step 4: Run `generate_summary` tests**

```bash
python -m pytest tests/unit/test_update_readme_news.py -k "generate_summary" -v
```

Expected: 2 tests PASSED.

- [ ] **Step 5: Write test for `create_pr`**

Append to `tests/unit/test_update_readme_news.py`:

```python
from update_readme_news import create_pr


def test_create_pr_returns_url():
    pr_response = _mock_response({"html_url": "https://github.com/oumi-ai/oumi/pull/9999"})
    with patch("update_readme_news.subprocess.run") as mock_run, \
         patch("update_readme_news.requests.post", return_value=pr_response):
        mock_run.return_value = MagicMock(returncode=0)
        url = create_pr(
            repo="oumi-ai/oumi",
            branch="chore/news-v0.8",
            tag="v0.8",
            news_item="- [2026/05] [Oumi v0.8 released](https://...) with stuff",
            release_body="## Notes\n- Feature A",
        )
    assert url == "https://github.com/oumi-ai/oumi/pull/9999"
```

- [ ] **Step 6: Run to confirm failure**

```bash
python -m pytest tests/unit/test_update_readme_news.py -k "create_pr" -v
```

Expected: FAILED with `NotImplementedError`.

- [ ] **Step 7: Implement `create_pr`**

Replace the `create_pr` stub:

```python
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
```

- [ ] **Step 8: Implement `main`**

Replace the `main` stub:

```python
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
```

- [ ] **Step 9: Run all tests**

```bash
python -m pytest tests/unit/test_update_readme_news.py -v
```

Expected: all tests PASSED.

- [ ] **Step 10: Commit**

```bash
git add .github/scripts/update_readme_news.py tests/unit/test_update_readme_news.py
git commit -m "feat: implement generate_summary, create_pr, and main for news update script"
```

---

### Task 5: Create the workflow YAML

**Files:**
- Create: `.github/workflows/update_readme_news.yaml`

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/update_readme_news.yaml`:

```yaml
name: Update README News

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      tag:
        description: "Release tag to create news item for (leave blank for latest release)"
        required: false
        default: ""

jobs:
  update-news:
    runs-on: ubuntu-latest

    permissions:
      contents: write
      pull-requests: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install requests anthropic

      - name: Run news update script
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GITHUB_REPOSITORY: ${{ github.repository }}
          INPUT_TAG: ${{ github.event.release.tag_name || github.event.inputs.tag || '' }}
        run: python .github/scripts/update_readme_news.py
```

- [ ] **Step 2: Verify the YAML is valid**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/update_readme_news.yaml'))" && echo "YAML valid"
```

Expected: `YAML valid`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/update_readme_news.yaml
git commit -m "feat: add update_readme_news GitHub Actions workflow"
```

---

### Task 6: End-to-end smoke test (dry run)

**Files:**
- Read: `.github/scripts/update_readme_news.py`

- [ ] **Step 1: Verify script imports cleanly**

```bash
cd /Users/stefanwebb/Code/Python/oumi
pip install requests anthropic
python -c "import sys; sys.path.insert(0, '.github/scripts'); import update_readme_news; print('Import OK')"
```

Expected: `Import OK` with no errors.

- [ ] **Step 2: Run full test suite one final time**

```bash
python -m pytest tests/unit/test_update_readme_news.py -v --tb=short
```

Expected: all tests PASSED.

- [ ] **Step 3: Verify README parse round-trips correctly on the real README**

```bash
python -c "
import sys
sys.path.insert(0, '.github/scripts')
from update_readme_news import parse_news_items, rewrite_readme_news

readme = open('README.md').read()
items = parse_news_items(readme)
print(f'Found {len(items)} news items')
print('First:', items[0])
print('Last:', items[-1])

# Dry-run rewrite with a fake item
updated = rewrite_readme_news(readme, '- [2026/05] [Oumi v0.8 released](https://github.com/oumi-ai/oumi/releases/tag/v0.8) with new stuff')
new_items = parse_news_items(updated)
print(f'After rewrite: {len(new_items)} items, first: {new_items[0]}')
assert new_items[0].startswith('- [2026/05]'), 'New item should be first'
print('Round-trip OK')
"
```

Expected output (approximately):
```
Found 24 news items
First: - [2026/03] Upgraded to Transformers v5, ...
Last: - [2025/04] Oumi now supports two new Vision-Language models: ...
After rewrite: 25 items, first: - [2026/05] [Oumi v0.8 released]...
Round-trip OK
```

- [ ] **Step 4: Final commit**

```bash
git add -u
git commit -m "chore: verify README news script round-trip on real README" --allow-empty
```

---

## Self-Review

**Spec coverage:**
- [x] Triggered on `release: [published]` + `workflow_dispatch` with optional tag input
- [x] Checks if release tag already in news → skip
- [x] Checks for existing open PR with tag in title → skip
- [x] Calls Claude API to generate summary
- [x] Formats news item as `- [YYYY/MM] [Oumi {tag} released](url) {summary}`
- [x] Keeps 12 visible, wraps 13+ under `<details><summary>Older items</summary>`
- [x] Creates branch `chore/news-{tag}`, commits, opens PR
- [x] PR body includes proposed news item + full release notes
- [x] Errors on missing secrets or missing News section

**Placeholder scan:** No TBDs or stubs remain after all tasks.

**Type consistency:** All function signatures match their call sites across tasks. `parse_news_items` returns `list[str]` used consistently in `rewrite_readme_news` and tests.
