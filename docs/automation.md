# Skills and workflows

Use the existing agent session for reasoning and GitHub Actions for repeatable
checks. Two portable skills capture the repository-specific procedures:

- [SDK maintenance](../.github/skills/sdk-maintenance/SKILL.md): reproduce a bug
  or CI failure, make the smallest repair, and preserve the learning in a test.
- [SDK release](../.github/skills/sdk-release/SKILL.md): prepare a consistent
  version, verify built distributions, and follow the existing release process.

`AGENTS.md` routes agents to these skills. Each uses the standard `SKILL.md`
format and can be read by an existing coding agent without a separate service.

The Makefile defines the checks once:

| Target | Result |
| --- | --- |
| `make check` | Lint, type checks, and tests |
| `make package` | Source/wheel builds and package metadata checks |
| `make smoke` | Package checks, wheel installation, version check, installed demo |

Run the smoke target in a virtual environment. CI runs `make check` on Python
3.10 through 3.14 and `make smoke` on Python 3.14. The release workflow retains
the `pypi` environment requirement, release-tag check, and trusted publishing.

Improve the system by turning actual failures into regression tests and updating
skills when a recurring task exposes missing instructions. Git history, PRs,
and CI results hold the evidence. This setup needs no additional inference key,
scheduled model calls, or Actions permission to create PRs.
