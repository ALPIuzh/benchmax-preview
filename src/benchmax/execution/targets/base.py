"""Passive execution destinations, independent of jobs and installations."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from benchmax.execution.targets.service import JobService


class Target(ABC):
    """Configuration for where jobs run."""

    @abstractmethod
    def service(self) -> JobService:
        """Build a client for this destination."""

    def _shares_job_store(self, other: Target) -> bool:
        """Whether dependency files can be reused directly (False by default)."""
        return False

    def prepare_output(self, output: str | os.PathLike[str] | None, *, caller_directory: Path) -> Path | None:
        """Interpret an output request in this target's filesystem namespace."""

        if output is None:
            return None

        path = Path(output).expanduser()
        return (caller_directory / path).resolve()

    def to_document(self) -> dict[str, Any]:
        """Return a persistable configuration, when supported by this target."""
        raise TypeError(f"{type(self).__name__} does not support saved profiles")

    @staticmethod
    def load(name: str) -> Target:
        from benchmax.execution.targets.registry import load_target

        return load_target(name)

    def register(self, name: str, *, persist: bool = False, replace: bool = False) -> Self:
        """Name this target in this process, optionally saving it for later ones."""
        from benchmax.execution.targets.registry import register_target

        register_target(name, self, persist=persist, replace=replace)
        return self


@dataclass(frozen=True)
class LocalTarget(Target):
    """Detached jobs on this machine, with an optional explicit run directory."""

    runs: str | None = None

    def __post_init__(self) -> None:
        if self.runs is None:
            from benchmax.execution.targets.local.service import _default_runs

            object.__setattr__(self, "runs", str(_default_runs()))

        if not isinstance(self.runs, (str, os.PathLike)) or not str(self.runs):
            raise TypeError("runs must be a non-empty path or None")

        object.__setattr__(self, "runs", str(Path(self.runs).expanduser().resolve()))

    def service(self) -> JobService:
        from benchmax.execution.targets.local import LocalJobService

        return LocalJobService(runs=self.runs)

    def _shares_job_store(self, other: Target) -> bool:
        return isinstance(other, LocalTarget) and self.runs == other.runs

    def to_document(self) -> dict[str, Any]:
        return {"kind": "local", **asdict(self)}


@dataclass(frozen=True)
class AdapterTarget(Target):
    """A passive URI configuration for an installed execution-target adapter."""

    uri: str

    def __post_init__(self) -> None:
        from urllib.parse import urlsplit

        if not isinstance(self.uri, str) or not urlsplit(self.uri).scheme:
            raise ValueError("adapter target requires a URI with a scheme")

    def service(self) -> JobService:
        from benchmax.execution.targets.service import _resolve_adapter_service

        return _resolve_adapter_service(self.uri)

    def to_document(self) -> dict[str, Any]:
        return {"kind": "adapter", **asdict(self)}
