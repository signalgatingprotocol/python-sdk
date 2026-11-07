---
name: sdk-delivery
description: Commit scoped SDK changes, open pull requests, watch CI, and complete authorized merges while keeping unrelated local work out of the PR.
---

# SDK delivery

Follow `AGENTS.md` for project scope and guardrails. Use
[SDK maintenance](../sdk-maintenance/SKILL.md) for repairs and validation.

## Isolate the change

Inspect the intended base, working tree, and untracked paths before staging.
Record hashes for files the user explicitly asks to preserve. Fetch the target
branch and inspect the prospective commit range and diff. A previously merged
feature branch can differ from current `main` after a squash merge.

Reuse a suitable current branch. For a new PR from a branch with unrelated
history, preserve that branch and start from the current target, carrying only
intended edits. Account for concurrent commits before proceeding. Stage
explicit paths and inspect the staged diff. Verify the proposed files match
the validated copy.

Before opening a PR, compare the pushed branch with its target. Check both
changed filenames and unique commits. Confirm any user-excluded commit is
absent from the PR's base-to-head commit range; a small file diff alone does
not establish scope.
Describe the concrete problem, resulting behavior, and checks that actually ran.

## Watch CI and merge

Track runs for the current PR head SHA. Inspect every expected job from the
workflow, including the Python matrix and packaging. After a new commit or
rebase, use checks for the new head. Inspect failed-job logs and reproduce the
failure before fixing it through the maintenance workflow.

Check review and merge status as well as CI. Report missing code-owner approval
as the merge blocker even when every CI job passes. Use PR auto-merge only when
repository settings already allow it; changing those settings needs its own
authorization. Preserve branch and environment protections.

Use authorization already given for the current task before asking again.
Apply an admin override only when that authorization covers the override;
approval naming another PR is not standing permission. If a normal merge is
blocked and the override is unauthorized, identify GitHub's specific rule and
ask once for the missing decision. A repeated unchanged review blocker needs
human action, not another CI run or merge retry.

Pin the merge to the verified head SHA using the tool's expected-head field or
`gh pr merge --match-head-commit`. Use a merge method allowed by the repository.
When a connector lacks a required capability, use the existing CLI with scoped
arguments. An authorized CLI override uses `--admin`; keep the head check.

## Close out

Verify GitHub reports the PR merged and compare the resulting merge commit with
its base to confirm scope. Verify preserved file hashes and keep branches
holding unrelated work recoverable. Merge remotely without resetting the
working checkout or sweeping untracked files into cleanup.

Report the PR link, commit, checks, and any remaining human action. Distinguish
local skill or code edits from changes already committed, published, or merged.
