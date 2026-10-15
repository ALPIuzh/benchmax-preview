"""Slurm target configuration for direct or SSH submission."""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from benchmax.execution.targets.base import Target

if TYPE_CHECKING:
    from benchmax.execution.targets.service import JobService


@dataclass(frozen=True, kw_only=True)
class SlurmTarget(Target):
    """Run complete attempts in Slurm using an already installed BenchMAX.

    Paths belong to the execution host and must be absolute. ``host=None``
    submits directly from a cluster login node. SSH uses existing credentials
    with password prompts disabled.
    """

    python: str
    workspace: str
    runs: str
    artifacts: str | None = None
    host: str | None = None
    user: str | None = None
    port: int = 22
    control_path: str | None = None
    account: str | None = None
    partition: str | None = None
    cpus: int = 1
    memory_mb: int = 1024
    time_limit: str = "00:10:00"
    gpus: int = 0
    poll_seconds: float = 300.0
    command_timeout: float = 60.0
    transfer_limit_mb: int = 64
    strict_source: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.strict_source, bool):
            raise TypeError("strict_source must be a boolean")

        for name in ("python", "workspace", "runs", "artifacts"):
            value = getattr(self, name)
            if value is None and name == "artifacts":
                continue

            if not isinstance(value, str) or not PurePosixPath(value).is_absolute() or "\x00" in value or "\n" in value:
                raise ValueError(f"{name} must be an absolute path on the execution host")

            if ".." in PurePosixPath(value).parts:
                raise ValueError(f"{name} must not contain '..'")

        for name in ("host", "user", "account", "partition"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
            ):
                raise ValueError(f"invalid {name}")

        if self.host is None and (self.user is not None or self.control_path is not None):
            raise ValueError("user and control_path require an SSH host")

        if self.control_path is not None and (not isinstance(self.control_path, str) or not self.control_path):
            raise ValueError("control_path must be a non-empty string")

        for name in ("cpus", "memory_mb", "port", "transfer_limit_mb"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if self.port > 65535:
            raise ValueError("port must be at most 65535")

        if not isinstance(self.gpus, int) or isinstance(self.gpus, bool) or self.gpus < 0:
            raise ValueError("gpus must be a non-negative integer")

        if not isinstance(self.time_limit, str) or not re.fullmatch(r"(?:\d+-)?\d{2,}:\d{2}:\d{2}", self.time_limit):
            raise ValueError("time_limit must have the form [days-]HH:MM:SS")

        clock = self.time_limit.split("-")[-1].split(":")
        if int(clock[1]) >= 60 or int(clock[2]) >= 60 or not any(int(v) for v in re.split("[-:]", self.time_limit)):
            raise ValueError("time_limit must be a positive, valid wall-clock limit")

        for name in ("poll_seconds", "command_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    def service(self) -> JobService:
        from benchmax.execution.targets.slurm.service import SlurmJobService

        return SlurmJobService(self)

    def _shares_job_store(self, other: Target) -> bool:
        return isinstance(other, SlurmTarget) and all(
            getattr(self, key) == getattr(other, key) for key in ("host", "user", "port", "runs")
        )

    def prepare_output(self, output: str | os.PathLike[str] | None, *, caller_directory: Path) -> Path | None:
        # The control endpoint resolves relative paths on the execution host.
        return Path(output) if output is not None else None

    @property
    def transfer_limit_bytes(self) -> int:
        return self.transfer_limit_mb * 1024**2

    def to_document(self) -> dict[str, Any]:
        return {"kind": "slurm", **asdict(self)}
