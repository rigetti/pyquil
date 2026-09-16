#!/usr/bin/env python3
"""Print one section of CHANGELOG.md, for use as GitHub release notes.

Sections are headed ``## <version> (<YYYY-MM-DD>)``.  A prerelease has no
section of its own -- an rc pull request bumps the version and leaves the
entries under ``## Unreleased`` -- so for a prerelease the Unreleased section is
printed instead.

Exits non-zero when no usable section is found, so that a release pull request
that forgot to rename the heading fails the release rather than publishing
empty notes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

CHANGELOG = Path("CHANGELOG.md")
RELEASE_HEADING = re.compile(r"^##\s+(?P<version>\S+)\s+\((?P<date>\d{4}-\d{2}-\d{2})\)\s*$")
UNRELEASED_HEADING = re.compile(r"^##\s+Unreleased\s*$", re.IGNORECASE)


def fail(message: str) -> "NoReturn":  # noqa: F821
    print(f"::error::{message}", file=sys.stderr)
    raise SystemExit(1)


def sections(lines: list[str]) -> list[tuple[str | None, list[str]]]:
    """Split the changelog into (version or None for Unreleased, body) pairs."""
    found: list[tuple[str | None, list[str]]] = []
    current: tuple[str | None, list[str]] | None = None

    for line in lines:
        if UNRELEASED_HEADING.match(line):
            current = (None, [])
            found.append(current)
            continue
        if match := RELEASE_HEADING.match(line):
            current = (match["version"], [])
            found.append(current)
            continue
        if line.startswith("## "):
            # A level-two heading we do not recognise ends the previous section
            # rather than silently accumulating into it.
            current = None
            continue
        if current is not None:
            current[1].append(line)

    return found


def body_for(found: list[tuple[str | None, list[str]]], wanted: Version) -> list[str] | None:
    for version, body in found:
        if version is None:
            continue
        try:
            if Version(version) == wanted:
                return body
        except InvalidVersion:
            continue
    return None


def unreleased(found: list[tuple[str | None, list[str]]]) -> list[str] | None:
    for version, body in found:
        if version is None:
            return body
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="version to print notes for, without a leading v")
    args = parser.parse_args()

    try:
        wanted = Version(args.version)
    except InvalidVersion:
        fail(f"{args.version!r} is not a valid PEP 440 version")

    if not CHANGELOG.is_file():
        fail(f"{CHANGELOG} not found")

    found = sections(CHANGELOG.read_text(encoding="utf-8").splitlines())

    body = body_for(found, wanted)
    source = f"the {wanted} section"
    if body is None and wanted.is_prerelease:
        body = unreleased(found)
        source = "the Unreleased section"

    if body is None:
        fail(
            f"no section for {wanted} in {CHANGELOG}. A release pull request renames "
            f'"## Unreleased" to "## {wanted} (YYYY-MM-DD)".'
        )

    text = "\n".join(body).strip()
    if not text:
        fail(f"{source} of {CHANGELOG} is empty; there is nothing to release")

    print(text)


if __name__ == "__main__":
    main()
