#!/usr/bin/env python3
"""Derive release and development-build versions for the release workflow.

Two subcommands, both writing ``name=value`` pairs to ``$GITHUB_OUTPUT``:

``detect``
    Compare the version in ``pyproject.toml`` at ``HEAD`` against ``HEAD^`` and
    decide whether this push to master should cut a release.

``dev``
    Derive the version for a manually dispatched development build.  Takes no
    input: the base comes from the committed version and the serial number from
    ``$GITHUB_RUN_NUMBER``, which is monotonic per workflow and assigned when a
    run is queued, so two branches dispatching at the same moment cannot collide.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.version import InvalidVersion, Version

PYPROJECT = "pyproject.toml"
DEV_VERSION = re.compile(r"\d+\.\d+\.\d+\.dev\d+")


def fail(message: str) -> "NoReturn":  # noqa: F821
    print(f"::error::{message}", file=sys.stderr)
    raise SystemExit(1)


def emit(**outputs: object) -> None:
    lines = [f"{key.replace('_', '-')}={value}" for key, value in outputs.items()]
    for line in lines:
        print(line)
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")


def parse_version(raw: bytes) -> Version:
    data = tomllib.loads(raw.decode("utf-8"))
    declared = data.get("project", {}).get("version") or data.get("tool", {}).get("poetry", {}).get("version")
    if not declared:
        fail(f"no version declared in {PYPROJECT}")
    try:
        return Version(declared)
    except InvalidVersion:
        fail(f"{declared!r} in {PYPROJECT} is not a valid PEP 440 version")


def version_at(revision: str) -> Version | None:
    """The declared version at a git revision, or None if it isn't readable there."""
    result = subprocess.run(
        ["git", "show", f"{revision}:{PYPROJECT}"],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return parse_version(result.stdout)


def tag_exists(version: Version) -> bool:
    out = subprocess.check_output(["git", "tag", "--list", f"v{version}"], text=True)
    return bool(out.strip())


def detect() -> None:
    head = parse_version(Path(PYPROJECT).read_bytes())
    if head.is_devrelease:
        fail(f"{head} is a development version and must never be committed to master")

    previous = version_at("HEAD^")
    if previous is None:
        print("no parent commit to compare against; not releasing")
        emit(changed="false", version="", is_prerelease="false")
        return

    # Comparing Version objects, not strings, so that a spelling change such as
    # 4.18.0-rc.1 -> 4.18.0rc1 is correctly seen as no change at all.
    if head == previous:
        print(f"version unchanged at {head}; not releasing")
        emit(changed="false", version="", is_prerelease="false")
        return

    if tag_exists(head):
        print(f"v{head} is already tagged; not releasing again")
        emit(changed="false", version="", is_prerelease="false")
        return

    print(f"version changed {previous} -> {head}")
    emit(
        changed="true",
        version=str(head),
        is_prerelease="true" if head.is_prerelease else "false",
    )


def dev() -> None:
    current = parse_version(Path(PYPROJECT).read_bytes())
    if current.is_devrelease:
        fail(f"{current} is a development version and must never be committed")

    # Branches never bump the version, so the committed value is the last thing
    # released.  A dev build is aimed at whatever comes next: the release being
    # stabilised if master is mid-rc, otherwise the next minor.
    if current.is_prerelease:
        base = f"{current.major}.{current.minor}.{current.micro}"
    else:
        base = f"{current.major}.{current.minor + 1}.0"

    run_number = os.environ.get("GITHUB_RUN_NUMBER")
    if not run_number:
        fail("GITHUB_RUN_NUMBER is not set")

    version = f"{base}.dev{run_number}"
    if not DEV_VERSION.fullmatch(version):
        fail(f"derived version {version!r} is not a well-formed development version")

    print(f"committed version {current}; building {version}")
    emit(version=version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["detect", "dev"])
    args = parser.parse_args()
    {"detect": detect, "dev": dev}[args.command]()


if __name__ == "__main__":
    main()
