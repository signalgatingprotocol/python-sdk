# SDK improvement

One GitHub Actions workflow handles failed main CI, weekly simplification,
and maintainer-directed tasks. It uses the official Codex action and the same
[maintenance skill](../.github/skills/sdk-maintenance/SKILL.md) local agents use.
No custom agent service, session database, or provider wrapper is required.

## Enable

1. Merge the workflow. In **Settings → Environments → sdk-improvement**, allow
   deployments only from branch `main` and add environment secret
   `OPENAI_API_KEY`. Use an OpenAI project scoped to this repository with an
   appropriate spend budget. Keep this key out of repository-level secrets so
   a workflow on another branch cannot request it.
2. Enable **Settings → Actions → General → Workflow permissions → Allow GitHub
   Actions to create and approve pull requests**. This setting permits creation;
   the workflow does not approve or merge PRs.
3. Set repository variable `SDK_IMPROVEMENT_ENABLED` to `true` to enable automatic
   runs. Leave it unset for manual runs only. Optional `SDK_IMPROVEMENT_MODEL`
   overrides the default `gpt-6.1-sol`.
4. Run **Actions → SDK improvement → Run workflow**, on `main`. Supply a concrete
   task or leave it blank to inspect repeated maintenance work.

```bash
gh workflow run improve.yml --ref main -f task='Reproduce and remove a recurring SDK failure'
```

Automatic runs occur Monday at 14:23 UTC and after a failed push CI run on main.
Set `SDK_IMPROVEMENT_ENABLED=false` to stop automatic inference. Disable the
workflow in Actions to stop all runs. Timeouts bound runner time; they are not
a hard cap on API spend. No inference credentials are included in the repository.

## Execution contract

- Only repository `maintain` or `admin` users pass the live permission check.
  The original actor, rerun actor, and originating CI actor must all qualify.
  Ordinary write access, organization membership, and bot status do not qualify.
- Only the main branch can start an improvement. Failed fork or pull-request
  runs cannot feed the failure trigger. Normal CI remains available to PRs.
  The inference environment also restricts credential access to `main`.
- Runs are serialized. An open `maintenance/improve-*` PR stops another model
  call. Healthy CI does not start inference. Automatic runs require opt-in.
- The model receives read permissions and a temporary sandbox. The official
  action proxies its inference credential and removes sudo access. Its job
  has no repository write token and checkout does not retain Git credentials.
- A fresh runner applies only a regular text patch of at most 100 KB and 10
  files. Python under `src/signal_gating`, `tests`, and `examples`, Markdown
  under `docs`, and root README/changelog are allowed. Workflow, dependency,
  script, policy, and agent-control changes fail validation. Symlinks,
  executable files, binary changes, and renames out of protected paths fail.
- Lint, types, all tests, package metadata, installed version, and the installed
  demo must pass before a fresh publishing job receives write permissions.
  That job applies the verified patch, creates one draft PR, and explicitly
  dispatches CI on its branch. No personal GitHub token is required.
- No-change runs create no PR. Proposal artifacts expire after seven days.
  Git history, tests, PR evidence, and Actions logs preserve the useful learning.
  This improves the repository through reviewed changes, not model training.

## Design basis

[Anthropic Managed Agents](https://www.anthropic.com/research/managed-agents)
separates reasoning, execution, and durable sessions. Here, those roles belong
to the maintained action, disposable runners, and GitHub records.
[OpenAI harness engineering](https://openai.com/index/harness-engineering/)
emphasizes repository knowledge, executable checks, and recurring cleanup.
Here, failures become regression tests and small refactoring PRs.
[Agent Skills](https://agentskills.io/home) provides the portable procedure
format. Keep company-specific knowledge there instead of adding another runner.

The workflow deliberately improves one SDK at a time. Share this contract and
skill across repositories before building organization-wide orchestration.
