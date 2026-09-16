"""Tests for scripts/ci.py, the helpers the release workflows depend on.

These run against real git repositories in tmp_path rather than mocks, because
the behaviour under test is largely "what does git say", and a mocked git would
not have caught the cases these exist to pin down.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]

# scripts/ is not a package, so load ci.py by path. It must be registered in
# sys.modules before it is executed, because @dataclass looks its own module up
# there while building the class.
_spec = importlib.util.spec_from_file_location("pyquil_ci", ROOT / "scripts" / "ci.py")
ci = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ci
_spec.loader.exec_module(ci)


def write_version(version: str) -> None:
    Path("pyproject.toml").write_text(f'[tool.poetry]\nname = "pyquil"\nversion = "{version}"\n')


def commit(message: str) -> None:
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-qm", message], check=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """An empty git repository at 4.18.0, with the process chdir'd into it."""
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q", "."], check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "ci"], check=True)
    write_version("4.18.0")
    commit("initial")
    return tmp_path


class TestDetectRelease:
    def test_bump_is_a_release(self, repo):
        write_version("4.19.0")
        commit("release 4.19.0")
        release, _ = ci.detect_release()
        assert release.changed
        assert str(release.version) == "4.19.0"
        assert not release.is_prerelease

    def test_release_candidate_is_flagged_as_a_prerelease(self, repo):
        write_version("4.19.0rc1")
        commit("release 4.19.0rc1")
        release, _ = ci.detect_release()
        assert release.changed
        assert release.is_prerelease

    def test_unrelated_change_to_pyproject_is_not_a_release(self, repo):
        Path("pyproject.toml").write_text(Path("pyproject.toml").read_text() + '\nfoo = "bar"\n')
        commit("bump a dependency")
        release, reason = ci.detect_release()
        assert not release.changed
        assert "unchanged" in reason

    def test_respelling_a_version_is_not_a_release(self, repo):
        """4.18.0-rc.1 and 4.18.0rc1 are the same version; only the text differs."""
        write_version("4.19.0rc1")
        commit("release 4.19.0rc1")
        write_version("4.19.0-rc.1")
        commit("use the canonical spelling")
        release, _ = ci.detect_release()
        assert not release.changed

    def test_an_already_tagged_version_is_not_released_again(self, repo):
        write_version("4.19.0")
        commit("release 4.19.0")
        subprocess.run(["git", "tag", "-a", "v4.19.0", "-m", "v4.19.0"], check=True)
        release, reason = ci.detect_release()
        assert not release.changed
        assert "already tagged" in reason

    def test_a_committed_dev_version_is_refused(self, repo):
        write_version("4.19.0.dev1")
        commit("oops")
        with pytest.raises(ci.CIError, match="must never be committed"):
            ci.detect_release()

    def test_an_unparseable_version_is_refused(self, repo):
        write_version("not-a-version")
        commit("oops")
        with pytest.raises(ci.CIError, match="not a valid PEP 440 version"):
            ci.detect_release()


class TestDevVersion:
    def test_derives_the_next_minor(self, repo):
        assert str(ci.dev_version("42")) == "4.19.0.dev42"

    def test_targets_the_release_being_stabilised_when_mid_rc(self, repo):
        """A dev build during a 4.19.0rc1 cycle is aimed at 4.19.0, not 4.20.0."""
        write_version("4.19.0rc1")
        commit("release candidate")
        assert str(ci.dev_version("42")) == "4.19.0.dev42"

    def test_the_result_is_always_publishable_from_a_branch(self, repo):
        """publish.yml only allows a branch to publish versions of this shape."""
        assert ci.DEV_VERSION.fullmatch(str(ci.dev_version("7")))

    def test_run_numbers_order_as_integers_not_strings(self, repo):
        assert Version(str(ci.dev_version("9"))) < Version(str(ci.dev_version("10")))


class TestReleaseNotes:
    CHANGELOG = """# Changelog

## Unreleased

### Features

- Something not yet released.

## 4.18.0 (2026-08-19)

### Features

- The released thing.

### Fixes

- The fixed thing.

## 4.17.0 (2025-10-08)

- Older.
"""

    @pytest.fixture(autouse=True)
    def changelog(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        Path("CHANGELOG.md").write_text(self.CHANGELOG)

    def test_returns_only_the_requested_section(self):
        notes = ci.release_notes(Version("4.18.0"))
        assert "The released thing." in notes
        assert "The fixed thing." in notes
        assert "Older." not in notes
        assert "Something not yet released." not in notes

    def test_a_prerelease_falls_back_to_unreleased(self):
        assert "Something not yet released." in ci.release_notes(Version("4.19.0rc1"))

    def test_a_respelled_heading_still_matches(self):
        Path("CHANGELOG.md").write_text("## 4.18.0-rc.1 (2026-08-19)\n\n- Entry.\n")
        assert "Entry." in ci.release_notes(Version("4.18.0rc1"))

    def test_a_missing_section_is_an_error(self):
        with pytest.raises(ci.CIError, match="no section for 9.9.9"):
            ci.release_notes(Version("9.9.9"))

    def test_an_empty_section_is_an_error(self):
        Path("CHANGELOG.md").write_text("## 4.19.0 (2026-09-14)\n\n## 4.18.0 (2026-08-19)\n\n- Entry.\n")
        with pytest.raises(ci.CIError, match="is empty"):
            ci.release_notes(Version("4.19.0"))
