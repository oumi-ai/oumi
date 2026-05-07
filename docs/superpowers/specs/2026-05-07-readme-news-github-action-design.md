# Design: README News Auto-Update GitHub Action

**Date:** 2026-05-07
**Status:** Approved

## Summary

A GitHub Action that triggers on release publication (and manually) to open a PR adding a new news item to `README.md`'s `## 🔥 News` section. Claude generates a concise one-line summary of the release from its notes. The PR is skipped if a news item or open PR for the release already exists.

---

## Trigger

```yaml
on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      tag:
        description: "Release tag to create news item for (defaults to latest release)"
        required: false
```

On `release`, the tag comes from `github.event.release.tag_name`. On `workflow_dispatch`, the script fetches the latest release if no tag is supplied.

---

## Files

| File | Purpose |
|------|---------|
| `.github/workflows/update_readme_news.yaml` | Workflow definition: triggers, job, permissions, env wiring |
| `.github/scripts/update_readme_news.py` | All logic: GitHub API, Anthropic API, README parsing, branch + PR creation |

---

## Script Logic (`update_readme_news.py`)

### Inputs (via environment variables)
- `GITHUB_TOKEN` — for GitHub API calls and PR creation
- `ANTHROPIC_API_KEY` — for generating the news summary
- `GITHUB_REPOSITORY` — `owner/repo` (e.g. `oumi-ai/oumi`)
- `INPUT_TAG` — release tag from workflow input (may be empty)

### Steps

1. **Resolve release** — if `INPUT_TAG` is set, fetch that release via GitHub API; otherwise fetch the latest release.

2. **Check for existing news item** — search `README.md` for the release tag string (e.g. `v0.8`). If found, print a message and exit 0 (no-op).

3. **Check for existing open PR** — call `GET /repos/{owner}/{repo}/pulls?state=open` and search titles for a string matching the release tag. If found, print a message and exit 0.

4. **Generate news summary via Claude** — call the Anthropic Messages API (`claude-sonnet-4-6`) with the release body as context. The prompt asks for a single short phrase (≤120 chars) like `with X, Y, and Z` summarizing the most important highlights. The result is used as the trailing description on the news item.

5. **Build the news item** — format:
   ```
   - [YYYY/MM] [Oumi {tag} released]({release_url}) {claude_summary}
   ```
   Where `YYYY/MM` is derived from the release's `published_at` field.

6. **Parse and rewrite the News section** — the script identifies the `## 🔥 News` section and collects all bullet items: those directly in the section and those inside the existing `<details>` block. It then:
   - Prepends the new item
   - Keeps the first 12 items as visible bullets
   - Wraps items 13+ in `<details><summary>Older items</summary>` … `</details>`

7. **Commit and open PR**:
   - Branch: `chore/news-{tag}` (e.g. `chore/news-v0.8`)
   - Commit message: `chore: add news item for {tag} release`
   - PR title: `chore: add README news item for {tag}`
   - PR body: includes the proposed news item and the full release notes for reviewer reference

---

## README Parsing Rules

The `## 🔥 News` section currently has this structure:

```markdown
## 🔥 News

- [YYYY/MM] item ...
- [YYYY/MM] item ...
...

<details>
<summary>Older updates</summary>

- [YYYY/MM] older item ...
</details>

## Next Section
```

The script reads lines from the start of the News section header until it hits the next `##`-level header. It extracts all `- [` bullet lines (both visible and inside `<details>`). The rewritten section always uses `<summary>Older items</summary>` (normalizing away `Older updates`).

---

## Permissions

```yaml
permissions:
  contents: write       # push branch
  pull-requests: write  # create PR
```

`GITHUB_TOKEN` is sufficient; no PAT needed.

---

## Secrets Required

| Secret | Where set |
|--------|-----------|
| `ANTHROPIC_API_KEY` | Repository or org secrets |
| `GITHUB_TOKEN` | Automatically provided by Actions |

---

## Error Handling

- If the Anthropic API call fails, the script exits with a non-zero code (workflow fails visibly rather than silently creating a bad PR).
- If `README.md` has no `## 🔥 News` section, the script exits with an error.
- All GitHub API calls check response status codes.

---

## Out of Scope

- Editing the news item content after PR creation (human reviewer does this in the PR).
- Generating multi-sentence release summaries.
- Updating the news section for non-release commits.
