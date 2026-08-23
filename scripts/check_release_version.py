"""Verify that package metadata, runtime API, and a release tag agree."""

from __future__ import annotations

import argparse
from importlib.metadata import version

import signal_gating


def check_versions(
    distribution_version: str,
    runtime_version: str,
    release_tag: str | None = None,
) -> None:
    """Raise an actionable error when any release version source disagrees."""
    if distribution_version != runtime_version:
        raise ValueError(
            f"distribution version {distribution_version} does not match runtime version "
            f"{runtime_version}; update pyproject.toml and signal_gating.__version__ together"
        )
    if release_tag not in (None, distribution_version, f"v{distribution_version}"):
        raise ValueError(
            f"release tag {release_tag} does not match distribution version "
            f"{distribution_version}; publish tag {distribution_version} or "
            f"v{distribution_version}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify installed Signal Gating release versions."
    )
    parser.add_argument("--tag", help="GitHub release tag, with optional v prefix")
    args = parser.parse_args()
    distribution_version = version("signal-gating")
    runtime_version = signal_gating.__version__
    try:
        check_versions(distribution_version, runtime_version, args.tag)
    except ValueError as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Verified Signal Gating release version {distribution_version}")


if __name__ == "__main__":
    main()
