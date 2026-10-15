"""Target-neutral job contracts, durable files, and execution-target discovery."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit

from benchmax.execution.specs import EvaluationSpec, TrainingSpec, WorkloadSpec
from benchmax.utils.files import _exclusive_file_lock, _fsync_directory
from benchmax.utils.hashing import _digest_bytes, _sha256_file
from benchmax.utils.json import write_document

if TYPE_CHECKING:
    from benchmax.execution.artifacts.results import WorkloadResult
    from benchmax.execution.audit.resources import AuditReport
    from benchmax.execution.targets.base import Target


class JobError(RuntimeError):
    """Base class for durable job failures."""


class JobFailedError(JobError):
    """A job reached a terminal failed state."""


class JobTimeoutError(JobError):
    """Waiting for a job exceeded the caller's timeout."""


class _JobDeadline:
    """Bound polling sleeps. Callers check for a ready result before asking to sleep."""

    def __init__(self, job_id: str, timeout: float | None) -> None:
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and non-negative, or None")

        self._job_id = job_id
        self._expires = None if timeout is None else time.monotonic() + timeout

    def delay(self, poll_seconds: float) -> float:
        if self._expires is None:
            return poll_seconds

        remaining = self._expires - time.monotonic()
        if remaining <= 0:
            raise JobTimeoutError(f"timed out waiting for job {self._job_id}")

        return min(poll_seconds, remaining)


class JobSubmissionUnknownError(JobError):
    """Submission may have been accepted, so reconnect before retrying or cancelling."""

    def __init__(self, message: str, *, job_id: str | None = None) -> None:
        super().__init__(message)
        self.job_id = job_id


def _validate_result(result: WorkloadResult, *, workload: WorkloadSpec) -> None:
    if result.workload_digest != workload.digest:
        raise JobError("attempt result does not match the submitted workload")

    if result.operation != workload.kind:
        article = "an" if result.operation == "evaluation" else "a"
        raise JobError(f"{workload.kind} attempt returned {article} {result.operation} result")

    if isinstance(workload, TrainingSpec):
        if result.seed != workload.seed:
            raise JobError("training result seed does not match the submitted workload")
    elif not isinstance(workload, EvaluationSpec):
        raise JobError("unknown workload operation")


_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
_JOB_ID = re.compile(r"(?:training|evaluation)-[0-9a-f]{16}")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise JobError(f"cannot read durable job file {path}: {error}") from error

    if not isinstance(value, dict):
        raise JobError(f"durable job file {path} is not a JSON object")

    return value


def _scheduler_id(run: Path) -> str | None:
    """Return the persisted scheduler ID of a run, if one was recorded."""
    path = run / "scheduler.json"
    if not path.exists():
        return None

    scheduler_id = _read_json(path).get("id")
    if not isinstance(scheduler_id, str) or not scheduler_id.isascii() or not scheduler_id.isdigit():
        raise JobError("invalid persisted Slurm scheduler ID")

    return scheduler_id


def _reservation_path(runs: Path, output: Path) -> Path:
    key = _digest_bytes(str(output).encode())
    return runs / ".output-reservations" / key


def _claim_reservation(path: Path, job_id: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(job_id + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    _fsync_directory(path.parent)


def _reservation_lock(runs: Path) -> AbstractContextManager[None]:
    runs.mkdir(parents=True, exist_ok=True)
    return _exclusive_file_lock(runs / ".output-reservations.lock")


def _release_output_reservation(runs: Path, job: Mapping[str, Any]) -> None:
    output = job.get("output")
    job_id = job.get("id")
    if not isinstance(output, str) or not isinstance(job_id, str):
        return

    if not job.get("output_reserved"):
        return

    with _reservation_lock(runs):
        reservation = _reservation_path(runs, Path(output))

        try:
            owner = reservation.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return

        if owner == job_id:
            reservation.unlink(missing_ok=True)
            _fsync_directory(reservation.parent)


def _publish_result(result: WorkloadResult, *, canonical: Path, output: Path) -> None:
    if output == canonical:
        return

    try:
        write_document(result, output, replace=False)
    except FileExistsError as error:
        try:
            already_published = _sha256_file(output) == _sha256_file(canonical)
        except OSError:
            already_published = False

        if not already_published:
            raise JobError(f"output destination was created while the job ran: {output}") from error


def _write_state(run: Path, state: str, **fields: Any) -> None:
    write_document(
        {
            "schema_version": 1,
            "state": state,
            "updated_at": _now(),
            **fields,
        },
        run / "state.json",
    )


def _state_lock(run: Path) -> AbstractContextManager[None]:
    """Serialize cancellation requests and attempt lifecycle transitions."""
    return _exclusive_file_lock(run / "state.lock")


def _reserve_output(runs: Path, output: Path, job_id: str, *, reconcile: Callable[[str], None] | None = None) -> None:
    """Claim an output path, optionally letting its backend reconcile a stale owner."""
    if output.exists():
        raise JobError(f"output destination already exists: {output}")

    (runs / ".output-reservations").mkdir(parents=True, exist_ok=True)
    reservation = _reservation_path(runs, output)

    for attempt in range(2):
        with _reservation_lock(runs):
            if output.exists():
                raise JobError(f"output destination already exists: {output}")

            try:
                owner = reservation.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                owner = None

            owner_state = "unknown"
            valid_owner = owner is not None and _JOB_ID.fullmatch(owner) is not None
            if valid_owner:
                try:
                    owner_state = _read_json(runs / owner / "state.json").get("state")
                except JobError:
                    pass

            if not valid_owner or owner_state in _TERMINAL_STATES:
                reservation.unlink(missing_ok=True)
                _claim_reservation(reservation, job_id)
                return

        if attempt == 0 and reconcile is not None:
            try:
                reconcile(owner)
            except JobError:
                pass

            continue

        break

    raise JobError(f"output destination is already reserved by another job: {output}")


class JobService(Protocol):
    """Client-side boundary implemented by each execution target.

    If acceptance cannot be determined, submit() raises
    JobSubmissionUnknownError with the reconnectable ID when available.
    """

    def submit(self, workload: WorkloadSpec, *, output: str | os.PathLike[str] | None = None) -> SubmittedJob: ...

    def open(self, job_id: str) -> SubmittedJob: ...

    def audit(self, workload: WorkloadSpec) -> AuditReport: ...

    def inspect_algorithm(self, name: str) -> dict[str, Any]:
        """Describe algorithm availability in the execution environment."""
        ...


class JobServiceFactory(Protocol):
    """Build the client-side service for one target URI."""

    def __call__(self, target: str) -> JobService: ...


class SubmittedJob(Protocol):
    """Private target handle wrapped by the public generic Job."""

    id: str

    @property
    def state(self) -> str: ...

    @property
    def workload(self) -> WorkloadSpec: ...

    @property
    def output(self) -> Path: ...

    def result(self, timeout: float | None = None) -> dict[str, Any]: ...

    async def result_async(self, timeout: float | None = None) -> dict[str, Any]: ...


def _target_adapters() -> dict[str, EntryPoint]:
    return {entry.name: entry for entry in entry_points(group="benchmax.execution_targets")}


def available_target_adapters() -> tuple[str, ...]:
    """Return built-in and installed target-adapter names without loading them."""

    return ("local", *sorted(_target_adapters()))


def resolve_job_service(target: str | Target) -> JobService:
    """Resolve one passive destination and build its client service."""
    from benchmax.execution.targets.registry import load_target

    return load_target(target).service()


def _resolve_adapter_service(target: str) -> JobService:
    """Build a service for a legacy URI adapter."""
    scheme = urlsplit(target).scheme

    if scheme == "local":
        raise JobError("local URI targets are not supported; use target='local' or LocalTarget(runs=...)")

    providers = _target_adapters()

    try:
        provider = providers[scheme]
    except KeyError as error:
        available = ", ".join(("local", *sorted(providers)))
        raise JobError(f"unknown target scheme {scheme!r}; available schemes: {available}") from error

    try:
        factory: JobServiceFactory = provider.load()
        return factory(target)
    except JobError:
        raise
    except Exception as error:
        raise JobError(f"cannot load target adapter {scheme!r}: {error}") from error
