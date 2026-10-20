# Agent instructions

For coding agents (Claude Code, Codex, Cursor, and anything similar) working in this repository. Humans should read [CONTRIBUTING.md](CONTRIBUTING.md) — everything there applies to agents too.

## Publication policy

**Never publish without the developer's explicit, per-instance approval.** Publishing means: pushing a branch (including new commits to an existing pull request branch), creating a fork or a remote repository, opening a pull request (draft included), or rewriting a published pull request's title or body.

This repository is public. Commit messages, branch names, pull request text, and code comments are visible to everyone the moment they leave the machine, and deleting them later does not un-publish them.

A general instruction to "work on this" is not approval to publish. Neither is a plan the developer approved, nor a task list they accepted. Approval is per publication, and it comes after they have seen what would go out.

Before publishing:

1. **Show the developer what would become visible** — every commit message, the pull request title and body, and the lines the diff adds. Not a summary of them.
2. **Check that content for anything that should not be public**: credentials and keys, internal hostnames and infrastructure identifiers, customer or partner names, people's contact details, and unreleased or business-internal information. Pattern matching alone misses names, so read the content as well as scanning it.
3. **Wait for the developer to say yes**, having seen the above. If something was flagged, it comes out or they explicitly accept it first.
4. **Then publish**, and share the resulting link.

If no developer is available to approve, keep the work on a local branch and say what is waiting. Do not publish and report it afterwards.

## Working with contributions

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming, the fork-and-pull-request flow, tests, and style. Run `pre-commit` before proposing a change. Every pull request still needs a review from a member of `oumi-ai/oumi-staff`.
