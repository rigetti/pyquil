#!/usr/bin/env python3
"""Helpers for the release workflows.

Each subcommand is documented with where it runs and what it reads and writes.
Run ``python scripts/ci.py --help`` for the list.

Commands that produce values for later workflow steps append ``name=value``
lines to ``$GITHUB_OUTPUT`` and also print them, so a run's log shows what was
decided.  Failures are reported with a ``::error::`` annotation and a non-zero
exit, which surfaces on the run summary rather than only in the log.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import textwrap
import tomllib
from dataclasses import dataclass
from pathlib import Path

from packaging.version import InvalidVersion, Version

PYPROJECT = Path("pyproject.toml")
CHANGELOG = Path("CHANGELOG.md")

#: A development build version, e.g. 4.19.0.dev42.  Deliberately narrow: this
#: pattern is what publish.yml checks before allowing a publish from a branch.
DEV_VERSION = re.compile(r"\d+\.\d+\.\d+\.dev\d+")

#: A released section, e.g. "## 4.18.0 (2026-08-19)".
RELEASE_HEADING = re.compile(r"^##\s+(?P<version>\S+)\s+\((?P<date>\d{4}-\d{2}-\d{2})\)\s*$")

#: The section entries accumulate in between releases.
UNRELEASED_HEADING = re.compile(r"^##\s+Unreleased\s*$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


class CIError(Exception):
    """A condition the workflow should stop on, reported as a GitHub annotation."""


def emit(**outputs: object) -> None:
    """Append outputs to $GITHUB_OUTPUT, and print them for the run log."""
    lines = [f"{name.replace('_', '-')}={value}" for name, value in outputs.items()]
    for line in lines:
        print(line)
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")


def parse_version(raw: bytes | str) -> Version:
    """The version declared in the given pyproject.toml contents."""
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    data = tomllib.loads(text)
    declared = data.get("project", {}).get("version") or data.get("tool", {}).get("poetry", {}).get("version")
    if not declared:
        raise CIError(f"no version declared in {PYPROJECT}")
    try:
        return Version(declared)
    except InvalidVersion as error:
        raise CIError(f"{declared!r} in {PYPROJECT} is not a valid PEP 440 version") from error


def committed_version() -> Version:
    """The version in the working tree's pyproject.toml."""
    if not PYPROJECT.is_file():
        raise CIError(f"{PYPROJECT} not found; run this from the repository root")
    return parse_version(PYPROJECT.read_bytes())


def version_at(revision: str) -> Version | None:
    """The version declared at a git revision, or None if unreadable there."""
    result = subprocess.run(["git", "show", f"{revision}:{PYPROJECT}"], capture_output=True)
    if result.returncode != 0:
        return None
    return parse_version(result.stdout)


def tag_exists(version: Version) -> bool:
    """Whether v<version> is already a tag in this clone.

    Only meaningful when tags have been fetched; release.yml passes
    fetch-tags so that this is a real check rather than a vacuous one.
    """
    return bool(subprocess.check_output(["git", "tag", "--list", f"v{version}"], text=True).strip())


@dataclass(frozen=True)
class Release:
    """Whether a push to master should cut a release, and of what."""

    changed: bool
    version: Version | None = None

    @property
    def is_prerelease(self) -> bool:
        return bool(self.version and self.version.is_prerelease)


# --------------------------------------------------------------------------- #
# detect-release
# --------------------------------------------------------------------------- #


def detect_release() -> tuple[Release, str]:
    """Decide whether HEAD changed the version, with a reason for the log."""
    head = committed_version()
    if head.is_devrelease:
        raise CIError(f"{head} is a development version and must never be committed to master")

    previous = version_at("HEAD^")
    if previous is None:
        return Release(changed=False), "no parent commit to compare against"
    # Comparing parsed versions, not strings, so that a spelling change such as
    # 4.18.0-rc.1 -> 4.18.0rc1 is correctly seen as no change at all.
    if head == previous:
        return Release(changed=False), f"version unchanged at {head}"
    if tag_exists(head):
        return Release(changed=False), f"v{head} is already tagged"
    return Release(changed=True, version=head), f"version changed {previous} -> {head}"


def command_detect_release(_: argparse.Namespace) -> None:
    """Decide whether a push to master should be released.

    Invoked by: release.yml, job `detect-version` (push to master).
    Requires:   a checkout with fetch-depth >= 2 and fetch-tags, to see HEAD^
                and existing tags.
    Outputs:    changed (true/false), version, is-prerelease (true/false).

    A release happens when the version in pyproject.toml differs from the
    previous commit's and is not already tagged, which makes re-runs a no-op.
    """
    release, reason = detect_release()
    print(reason)
    emit(
        changed=str(release.changed).lower(),
        version=str(release.version or ""),
        is_prerelease=str(release.is_prerelease).lower(),
    )


# --------------------------------------------------------------------------- #
# dev-version
# --------------------------------------------------------------------------- #


def dev_version(run_number: str) -> Version:
    """The version for a development build of the current branch."""
    current = committed_version()
    if current.is_devrelease:
        raise CIError(f"{current} is a development version and must never be committed")

    # Branches never bump the version, so the committed value is the last thing
    # released.  A dev build is aimed at whatever comes next: the release being
    # stabilised if master is mid-rc, otherwise the next minor.
    if current.is_prerelease:
        base = f"{current.major}.{current.minor}.{current.micro}"
    else:
        base = f"{current.major}.{current.minor + 1}.0"

    version = f"{base}.dev{run_number}"
    if not DEV_VERSION.fullmatch(version):
        raise CIError(f"derived version {version!r} is not a well-formed development version")
    return Version(version)


def command_dev_version(_: argparse.Namespace) -> None:
    """Derive the version for a manually dispatched development build.

    Invoked by: release.yml, job `tag-dev` (workflow_dispatch, any branch).
    Requires:   $GITHUB_RUN_NUMBER.
    Outputs:    version.

    Takes no human input by design.  The serial number is the run number, which
    is unique per workflow and assigned when a run is queued, so two branches
    dispatching at the same moment cannot produce the same version.
    """
    run_number = os.environ.get("GITHUB_RUN_NUMBER")
    if not run_number:
        raise CIError("GITHUB_RUN_NUMBER is not set")
    version = dev_version(run_number)
    print(f"committed version {committed_version()}; building {version}")
    emit(version=str(version))


# --------------------------------------------------------------------------- #
# release-notes
# --------------------------------------------------------------------------- #


def changelog_sections(lines: list[str]) -> list[tuple[str | None, list[str]]]:
    """Split the changelog into (version, body) pairs; None is Unreleased."""
    found: list[tuple[str | None, list[str]]] = []
    current: tuple[str | None, list[str]] | None = None

    for line in lines:
        if UNRELEASED_HEADING.match(line):
            current = (None, [])
            found.append(current)
        elif match := RELEASE_HEADING.match(line):
            current = (match["version"], [])
            found.append(current)
        elif line.startswith("## "):
            # An unrecognised level-two heading ends the previous section rather
            # than silently accumulating into it.
            current = None
        elif current is not None:
            current[1].append(line)

    return found


def release_notes(version: Version) -> str:
    """The changelog body to publish as the release notes for a version."""
    if not CHANGELOG.is_file():
        raise CIError(f"{CHANGELOG} not found; run this from the repository root")

    sections = changelog_sections(CHANGELOG.read_text(encoding="utf-8").splitlines())

    body: list[str] | None = None
    for declared, lines in sections:
        if declared is None:
            continue
        try:
            if Version(declared) == version:
                body = lines
                break
        except InvalidVersion:
            continue

    # A prerelease has no section of its own: an rc bumps the version and leaves
    # the entries where they are, to be renamed by the final release.
    if body is None and version.is_prerelease:
        body = next((lines for declared, lines in sections if declared is None), None)

    if body is None:
        raise CIError(
            f'no section for {version} in {CHANGELOG}. A release pull request renames "## Unreleased" '
            f'to "## {version} (YYYY-MM-DD)".'
        )

    text = "\n".join(body).strip()
    if not text:
        raise CIError(f"the {version} section of {CHANGELOG} is empty; there is nothing to release")
    return text


def command_release_notes(args: argparse.Namespace) -> None:
    """Print the CHANGELOG.md section to publish as a release's notes.

    Invoked by: release.yml, job `tag-release`, piped to `gh release create
                --notes-file`.
    Requires:   the version as an argument, without a leading v.
    Outputs:    the notes on stdout.

    Fails when the section is missing, so that a release pull request which
    forgot to rename "## Unreleased" stops the release instead of publishing
    empty notes against an irreversible PyPI upload.
    """
    try:
        version = Version(args.version)
    except InvalidVersion as error:
        raise CIError(f"{args.version!r} is not a valid PEP 440 version") from error
    print(release_notes(version))


# --------------------------------------------------------------------------- #
# patch-grpc-web
# --------------------------------------------------------------------------- #


def command_patch_grpc_web(_: argparse.Namespace) -> None:
    """Rewrite pyproject.toml in place to build the pyquil-grpc-web variant.

    Invoked by: publish.yml, job `build-publish-grpc-web`, before the build.
    Requires:   the `toml` package (imported here so the other commands do not
                need it).
    Outputs:    none; edits pyproject.toml and pyquil/_version.py in the
                runner's tree only.

    Renames the published package and swaps the qcs-sdk-python dependency for
    its grpc-web build, keeping the import name as pyquil.
    """
    import toml  # noqa: PLC0415  (only this command needs a TOML writer)

    root = Path(__file__).resolve().parent.parent

    with open(root / "pyproject.toml", "r+", encoding="utf-8") as handle:
        data = toml.load(handle)

        # Renames the published package, but not the import name.
        data["project"] = {"name": "pyquil-grpc-web"}
        data["tool"]["poetry"]["name"] = "pyquil-grpc-web"

        # Same dependency definition, under the grpc-web name.
        dependencies = data["tool"]["poetry"]["dependencies"]
        dependencies["qcs-sdk-python-grpc-web"] = dependencies.pop("qcs-sdk-python")

        handle.seek(0)
        handle.write(toml.dumps(data))
        handle.truncate()

    # `__package__` resolves to pyquil, but this package is pyquil_grpc_web.
    version_file = root / "pyquil" / "_version.py"
    version_file.write_text(
        version_file.read_text(encoding="utf-8").replace("__package__", '"pyquil_grpc_web"'),
        encoding="utf-8",
    )
    print("patched pyproject.toml and pyquil/_version.py for pyquil-grpc-web")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ci.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    for name, handler in (
        ("detect-release", command_detect_release),
        ("dev-version", command_dev_version),
        ("release-notes", command_release_notes),
        ("patch-grpc-web", command_patch_grpc_web),
    ):
        doc = textwrap.dedent(handler.__doc__ or "").strip()
        subparser = commands.add_parser(
            name,
            help=doc.splitlines()[0],
            description=doc,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        subparser.set_defaults(handler=handler)
        if name == "release-notes":
            subparser.add_argument("version", help="version to print notes for, without a leading v")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except CIError as error:
        print(f"::error::{error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
