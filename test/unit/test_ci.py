"""Tests for scripts/ci.py, the helpers the release workflows depend on."""

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


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A checkout at 4.18.0, with the process chdir'd into it."""
    monkeypatch.chdir(tmp_path)
    write_version("4.18.0")
    return tmp_path


class TestDetectRelease:
    def test_an_unreleased_version_is_released(self, repo):
        write_version("4.19.0")
        release, _ = ci.detect_release(already_released=lambda _: False)
        assert release.is_release
        assert str(release.version) == "4.19.0"
        assert not release.is_prerelease

    def test_a_release_candidate_is_flagged_as_a_prerelease(self, repo):
        write_version("4.19.0rc1")
        release, _ = ci.detect_release(already_released=lambda _: False)
        assert release.is_release
        assert release.is_prerelease

    def test_an_already_released_version_is_not_released_again(self, repo):
        """This is what makes an ordinary push to master a no-op."""
        release, reason = ci.detect_release(already_released=lambda _: True)
        assert not release.is_release
        assert release.version is None
        assert "already been released" in reason

    def test_a_tag_without_a_release_still_releases(self, repo):
        """A tag with no release behind it is a release that died part way.

        The tag alone must not wedge it: the workflow's tag and release steps are
        each skipped if already done, so a later push finishes the job.
        """
        release, _ = ci.detect_release(already_released=lambda _: False)
        assert release.is_release

    def test_the_version_is_normalised(self, repo):
        """The tag and the release-exists probe use the canonical spelling."""
        write_version("4.19.0-rc.1")
        release, _ = ci.detect_release(already_released=lambda _: False)
        assert str(release.version) == "4.19.0rc1"

    def test_a_committed_dev_version_is_refused(self, repo):
        write_version("4.19.0.dev1")
        with pytest.raises(ci.CIError, match="must never be committed"):
            ci.detect_release(already_released=lambda _: False)

    def test_an_unparseable_version_is_refused(self, repo):
        write_version("not-a-version")
        with pytest.raises(ci.CIError, match="not a valid PEP 440 version"):
            ci.detect_release(already_released=lambda _: False)

    def test_a_version_override_is_taken_at_face_value(self, repo):
        """Re-publishing an existing release must not be gated on it being unreleased."""
        args = ci.build_parser().parse_args(["detect-release", "--version", "4.19.0"])
        assert args.version == "4.19.0"

    def test_release_exists_reports_false_without_a_release(self, repo, monkeypatch):
        """The default probe shells out to gh; a non-zero exit means not released."""
        monkeypatch.setattr(
            ci.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(args=a, returncode=1),
        )
        assert not ci.release_exists(Version("4.19.0"))


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

    def test_a_stranded_prerelease_section_is_an_error(self):
        """An rc that renamed its heading would silently truncate the final notes."""
        Path("CHANGELOG.md").write_text(
            "## 4.19.0 (2026-09-24)\n\n- Landed after the rc.\n\n"
            "## 4.19.0rc1 (2026-09-17)\n\n- Feature A.\n\n"
            "## 4.18.0 (2026-08-19)\n\n- Old.\n"
        )
        with pytest.raises(ci.CIError, match="4.19.0rc1"):
            ci.release_notes(Version("4.19.0"))

    def test_a_prerelease_section_of_another_version_is_fine(self):
        Path("CHANGELOG.md").write_text(
            "## 4.19.0 (2026-09-24)\n\n- Entry.\n\n"
            "## 4.18.0rc1 (2026-08-01)\n\n- Belongs to the previous cycle.\n"
        )
        assert "Entry." in ci.release_notes(Version("4.19.0"))

    def test_the_prerelease_itself_is_unaffected_by_the_check(self):
        """The check guards final releases only; an rc may have its own section."""
        Path("CHANGELOG.md").write_text("## 4.19.0rc1 (2026-09-17)\n\n- Feature A.\n")
        assert "Feature A." in ci.release_notes(Version("4.19.0rc1"))

    def test_an_empty_section_is_an_error(self):
        Path("CHANGELOG.md").write_text("## 4.19.0 (2026-09-14)\n\n## 4.18.0 (2026-08-19)\n\n- Entry.\n")
        with pytest.raises(ci.CIError, match="is empty"):
            ci.release_notes(Version("4.19.0"))


class TestPatchGrpcWeb:
    PYPROJECT = """[tool.poetry]
name = "pyquil"
version = "4.18.0"
description = "A Python library."

[tool.poetry.dependencies]
python = ">=3.11, <3.13"
numpy = ">=1.26,<3"
qcs-sdk-python = ">=0.20.1,<0.22"
quil = ">=0.15.3,<0.18"

[tool.poetry.extras]
latex = ["ipython"]

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
"""

    VERSION_PY = "from importlib.metadata import version\n\npyquil_version = version(__package__)\n"

    @pytest.fixture
    def root(self, tmp_path):
        (tmp_path / "pyquil").mkdir()
        (tmp_path / "pyproject.toml").write_text(self.PYPROJECT)
        (tmp_path / "pyquil" / "_version.py").write_text(self.VERSION_PY)
        return tmp_path

    def parsed(self, root):
        import tomllib

        return tomllib.loads((root / "pyproject.toml").read_text())

    def test_renames_the_published_package(self, root):
        ci.patch_grpc_web(root)
        data = self.parsed(root)
        assert data["tool"]["poetry"]["name"] == "pyquil-grpc-web"
        assert data["project"]["name"] == "pyquil-grpc-web"

    def test_swaps_the_dependency_keeping_its_constraint(self, root):
        ci.patch_grpc_web(root)
        dependencies = self.parsed(root)["tool"]["poetry"]["dependencies"]
        assert "qcs-sdk-python" not in dependencies
        assert dependencies["qcs-sdk-python-grpc-web"] == ">=0.20.1,<0.22"

    def test_leaves_everything_else_alone(self, root):
        ci.patch_grpc_web(root)
        data = self.parsed(root)
        dependencies = data["tool"]["poetry"]["dependencies"]
        assert data["tool"]["poetry"]["version"] == "4.18.0"
        assert dependencies["numpy"] == ">=1.26,<3"
        assert dependencies["quil"] == ">=0.15.3,<0.18"
        assert data["tool"]["poetry"]["extras"] == {"latex": ["ipython"]}
        assert data["build-system"]["build-backend"] == "poetry.core.masonry.api"

    def test_rewrites_the_package_name_lookup(self, root):
        ci.patch_grpc_web(root)
        source = (root / "pyquil" / "_version.py").read_text()
        assert 'version("pyquil_grpc_web")' in source
        assert "__package__" not in source

    def test_the_result_is_still_valid_toml(self, root):
        ci.patch_grpc_web(root)
        self.parsed(root)  # raises if not
