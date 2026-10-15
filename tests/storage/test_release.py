from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from benchmax.release import LibraryRelease
from tests import REPOSITORY


def test_release_identity_is_name_and_version_not_build_configuration(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname = "benchmax"\nversion = "0.1.0"\n')
    first = LibraryRelease.from_pyproject(project)
    project.write_text(project.read_text() + '[build-system]\nrequires = ["different-build-tool"]\n')
    assert LibraryRelease.from_pyproject(project) == first
    assert replace(first, version="0.2.0").digest != first.digest


def test_release_metadata_round_trips_without_storage() -> None:
    release = LibraryRelease.from_pyproject(REPOSITORY / "pyproject.toml")
    restored = LibraryRelease.from_document(release.to_document())

    assert restored == release
    assert restored.digest == release.digest


@pytest.mark.parametrize("document", [None, {}, {"schema_version": 2, "kind": "library-release"}])
def test_invalid_release_documents_are_rejected(document: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        LibraryRelease.from_document(document)


@pytest.mark.parametrize("tracked", [True, False], ids=["source-checkout", "installed-wheel"])
def test_provenance_only_uses_git_for_tracked_library_source(tmp_path, monkeypatch, tracked):
    from benchmax import release as module

    (tmp_path / ".git").mkdir()
    source = tmp_path / ("src/benchmax/release.py" if tracked else ".venv/lib/site-packages/benchmax/release.py")
    monkeypatch.setattr(module, "__file__", str(source))
    calls = []

    def git(command, **kwargs):
        calls.append(command)
        if command[1] == "ls-files":
            assert command[-1] == str(source.relative_to(tmp_path))
            if not tracked:
                raise subprocess.CalledProcessError(1, command)
            return b"source"
        if command[1] == "rev-parse":
            return "a" * 40
        assert command[1] == "status"
        return b""

    monkeypatch.setattr(subprocess, "check_output", git)
    provenance = module.capture_producer()["provenance"]
    assert provenance["source_revision"] == ("a" * 40 if tracked else None)
    assert ("working_tree_dirty" in provenance) is tracked
    assert len(calls) == (3 if tracked else 1)
