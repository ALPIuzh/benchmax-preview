"""Deferred jobs with a durable, target-neutral lifecycle."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, cast

from benchmax.execution.artifacts.results import WorkloadResult, result_from_document
from benchmax.execution.jobs import outputs
from benchmax.execution.jobs.reference import Reference
from benchmax.execution.specs import _SCHEMA_VERSION, WorkloadSpec
from benchmax.execution.targets.base import Target
from benchmax.execution.targets.registry import load_target
from benchmax.execution.targets.service import (
    _TERMINAL_STATES,
    JobError,
    JobSubmissionUnknownError,
    SubmittedJob,
    _now,
    resolve_job_service,
)

if TYPE_CHECKING:
    from benchmax.execution.audit.resources import AuditReport
    from benchmax.execution.jobs.progress import JobProgress

ResultT = TypeVar("ResultT")
WorkloadFactory = Callable[[], WorkloadSpec]


class Job(Generic[ResultT]):
    """A deferred, target-neutral execution with one durable lifecycle."""

    _workload: WorkloadSpec | None
    _workload_factory: WorkloadFactory | None
    _output: Path | None
    _requested_output: str | os.PathLike[str] | None
    _caller_directory: Path
    _target: Target | None
    _handle: SubmittedJob | None
    _submission_error: JobSubmissionUnknownError | None
    _cancel_requested: bool
    _prepare_lock: threading.Lock
    _run_lock: threading.Lock

    def __init__(self) -> None:
        raise TypeError("Jobs are created by train(..., submit=True), game.evaluate(..., submit=True), or Job.open()")

    @classmethod
    def _create(cls, workload: WorkloadSpec | WorkloadFactory, *, output: str | os.PathLike[str] | None = None) -> Self:
        job = object.__new__(cls)
        job._workload = workload if not callable(workload) else None
        job._workload_factory = workload if callable(workload) else None
        job._output = Path(output).expanduser().resolve() if output is not None else None
        job._requested_output = output
        job._caller_directory = Path.cwd()
        job._target = None
        job._handle = None
        job._submission_error = None
        job._cancel_requested = False
        job._prepare_lock = threading.Lock()
        job._run_lock = threading.Lock()
        return job

    def _prepare(self) -> WorkloadSpec:
        if self._workload is not None:
            return self._workload

        with self._prepare_lock:
            if self._workload is None:
                assert self._workload_factory is not None
                self._workload = self._workload_factory()
                self._workload_factory = None

            return self._workload

    @property
    def id(self) -> str | None:
        """The target ID, including a recoverable ID after uncertain submission."""

        if self._handle is not None:
            return self._handle.id

        return self._submission_error.job_id if self._submission_error is not None else None

    @property
    def state(self) -> str:
        """Return created, submission_unknown, or the submitted target state."""

        if self._handle is not None:
            return self._handle.state

        return "submission_unknown" if self._submission_error is not None else "created"

    @property
    def output(self) -> Path | None:
        """Primary durable output, canonicalized once the job is run."""

        return self._handle.output if self._handle is not None else self._output

    def audit(self, target: str | Target = "local") -> AuditReport:
        """Inspect this workload for one target without running it."""

        return resolve_job_service(target).audit(self._prepare())

    def progress(self) -> JobProgress:
        """Fetch one progress/dependency snapshot, which Jupyter renders automatically.

        This performs target I/O for a submitted job, without waiting for its
        completion. The returned snapshot is passive, so call again to refresh.
        An unsubmitted job is described locally without querying dependencies.
        """
        from benchmax.execution.jobs.progress import JobProgress, unsubmitted_progress

        if self._submission_error is not None:
            return JobProgress(self.id, self._prepare().kind, "submission_unknown", "unknown", _now(),
                               reason=str(self._submission_error))

        if self._handle is None:
            return unsubmitted_progress(self._prepare())

        fetch = getattr(self._handle, "progress", None)

        if fetch is None:
            state = self._handle.state
            return JobProgress(self.id, self._prepare().kind, state, state, _now(),
                               reason="This target adapter does not expose detailed progress.")

        return JobProgress.from_document(fetch())

    def run(self, target: str | Target = "local") -> Self:
        """Submit to one target and return immediately after acceptance."""

        with self._run_lock:
            if self._submission_error is not None:
                raise self._submission_error

            try:
                destination = load_target(target)
            except JobError:
                if self._handle is not None:
                    raise JobError(f"job already runs on target {self._target!r}; it cannot be rebound to {target!r}") from None

                raise

            if self._handle is not None:
                if destination != self._target:
                    raise JobError(f"job already runs on target {self._target!r}; it cannot be rebound to {target!r}")

                return self

            output = destination.prepare_output(self._requested_output, caller_directory=self._caller_directory)

            try:
                self._handle = resolve_job_service(destination).submit(self._prepare(), output=output)
            except JobSubmissionUnknownError as error:
                self._submission_error = error
                self._target = destination
                raise

            self._target = destination

        return self

    def _result_document(self, timeout: float | None = None) -> WorkloadResult:
        if self._handle is None:
            raise JobError("job has not been run")

        return result_from_document(self._handle.result(timeout))

    def result(self, timeout: float | None = None, *, path: str | os.PathLike[str] | None = None) -> ResultT:
        """Wait and return the natural value, optionally retaining files locally.

        path is a new local directory: training stores its manifest and policy
        files, while evaluation stores result.json. The returned value keeps its type.
        """

        destination = outputs._result_path(path)
        result = self._result_document(timeout)
        return cast(ResultT, outputs.load_result(result, self._prepare(), handle=self._handle, path=destination))

    async def result_async(self, timeout: float | None = None, *, path: str | os.PathLike[str] | None = None) -> ResultT:
        """Wait without blocking the caller's event-loop thread."""

        if self._handle is None:
            raise JobError("job has not been run")

        destination = outputs._result_path(path)
        result = result_from_document(await self._handle.result_async(timeout))
        value = await asyncio.to_thread(outputs.load_result, result, self._prepare(),
                                        handle=self._handle, path=destination)
        return cast(ResultT, value)

    def save(self, path: str | os.PathLike[str], *, timeout: float | None = None) -> Reference[ResultT]:
        """Wait and save files in a new local directory without loading a model."""
        destination = outputs._save_path(path)
        result = self._result_document(timeout)
        saved = outputs.save_result(result, self._prepare(), destination, handle=self._handle)
        return Reference.from_document(saved.to_document())

    @classmethod
    def _from_workload(cls, workload: WorkloadSpec, *, output: str | os.PathLike[str] | None = None) -> Job[Any]:
        """Construct from a private portable payload for CLI and adapters."""

        return cls._create(workload, output=output)

    @classmethod
    def open(cls, job_id: str, *, target: str | Target = "local") -> Job[Any]:
        """Reconnect to a durable job from another Python process."""

        destination = load_target(target)
        handle = resolve_job_service(destination).open(job_id)
        job = cls._create(handle.workload)
        job._handle = handle
        job._target = destination
        return job

    def reference(self) -> Reference[ResultT]:
        """Describe this submitted job's future output without waiting for it."""

        if self._handle is None or self._target is None:
            raise JobError("job has not been run; submit it before creating an output reference")

        workload = self._prepare()
        return Reference({"schema_version": _SCHEMA_VERSION, "kind": "job-output", "job_id": self._handle.id,
                          "target": self._target.to_document(), "output": workload.output_kind,
                          "workload_digest": workload.digest})

    def to_document(self) -> dict[str, Any]:
        """Inspect the workload and explicit references."""
        return self._prepare().to_document()

    def cancel(self) -> None:
        """Cancel this exact job. Unsubmitted and terminal jobs are no-ops.

        Cancellation failures and uncertain submission outcomes remain errors.
        A no-op does not prevent an unsubmitted job from being run later.
        """

        with self._run_lock:
            if self._submission_error is not None:
                raise self._submission_error

            if self._handle is None or self._cancel_requested:
                return

            if self._handle.state in _TERMINAL_STATES:
                return

            cancel = getattr(self._handle, "cancel", None)
            if cancel is None:
                raise JobError("this job's target does not support cancellation")

            cancel()
            self._cancel_requested = True
