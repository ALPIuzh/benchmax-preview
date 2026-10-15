"""Producing library version and captured runtime provenance."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmax.utils.hashing import _digest_bytes
from benchmax.utils.json import canonical_json


@dataclass(frozen=True, slots=True)
class LibraryRelease:
    """A library's name and version."""

    name: str
    version: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not isinstance(self.version, str) or not self.version:
            raise ValueError("library name and version must be non-empty strings")

    @classmethod
    def installed(cls) -> LibraryRelease:
        from importlib.metadata import version

        return cls("benchmax", version("benchmax"))

    @classmethod
    def from_pyproject(cls, path: str | Path) -> LibraryRelease:
        """Explicitly identify a legacy artifact's producing project."""
        with Path(path).open("rb") as stream:
            project = tomllib.load(stream)["project"]
        return cls(project["name"], project["version"])

    def to_document(self) -> dict[str, Any]:
        return {"schema_version": 1, "kind": "library-release", "name": self.name, "version": self.version}

    @classmethod
    def from_document(cls, document: object) -> LibraryRelease:
        if (not isinstance(document, Mapping) or set(document) != {"schema_version", "kind", "name", "version"}
                or type(document["schema_version"]) is not int or document["schema_version"] != 1
                or document["kind"] != "library-release"):
            raise ValueError("unsupported library release document")
        return cls(document["name"], document["version"])

    @property
    def digest(self) -> str:
        return _digest_bytes(canonical_json(self.to_document()).encode("utf-8"))


def capture_producer() -> dict[str, Any]:
    """Capture the executing environment before execution, retaining unknown source identity."""
    import platform
    import subprocess
    from importlib.metadata import PackageNotFoundError, requires, version

    from packaging.requirements import Requirement

    release = LibraryRelease.installed()
    packages = {release.name: release.version}
    for value in requires(release.name) or []:
        requirement = Requirement(value)
        if requirement.marker is not None and not any(
            requirement.marker.evaluate({"extra": extra}) for extra in ("", "training")
        ):
            continue
        name = requirement.name
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    provenance = {"source_revision": None, "python_version": platform.python_version(), "packages": packages}
    # Only inspect a checkout containing this library, never the caller's project.
    for root in Path(__file__).resolve().parents:
        if (root / ".git").exists():
            try:
                subprocess.check_output(
                    ["git", "ls-files", "--error-unmatch", "--", str(Path(__file__).resolve().relative_to(root))],
                    cwd=root, stderr=subprocess.DEVNULL,
                )
                provenance["source_revision"] = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
                provenance["working_tree_dirty"] = bool(subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL))
            except (OSError, subprocess.CalledProcessError):
                pass
            break
    return {"release": release.to_document(), "provenance": provenance}
