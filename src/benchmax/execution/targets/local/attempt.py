"""Run a workload after the local target has allocated and persisted it."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from benchmax.execution.artifacts.results import WorkloadResult, result_from_document
from benchmax.utils.hashing import _sha256_file

from benchmax.execution.targets.service import (  # isort: skip
    _JOB_ID, JobError, JobFailedError, _now, _publish_result, _read_json, _release_output_reservation, _state_lock,
    _validate_result, _write_state
)
from benchmax.execution.specs import WorkloadSpec, _ArtifactMetadata, _ReferenceSpec, load_workload

from benchmax.execution.targets.local.service import (  # isort: skip
    LocalJobService, _default_runs, _finish_cancelled, _process_identity
)
from benchmax.utils.json import write_document


class AttemptExecutor(Protocol):
    """Execute one validated workload inside an allocated environment."""

    def __call__(self, workload: WorkloadSpec, output: Path) -> WorkloadResult: ...


@contextmanager
def _worker_process_group() -> Iterator[None]:
    """Keep nested evaluation workers in this dedicated local attempt's session."""
    key = "BENCHMAX_LOCAL_PROCESS_GROUP"
    previous = os.environ.get(key)

    if os.getsid(0) == os.getpid():
        os.environ[key] = str(os.getpgrp())

    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def run_local_attempt(
    job_id: str, *, runs: str | os.PathLike[str] | None = None, executor: AttemptExecutor | None = None
) -> dict[str, object]:
    """Execute the immutable workload owned by one local target attempt."""

    if _JOB_ID.fullmatch(job_id) is None:
        raise JobError(f"invalid job ID {job_id!r}")

    run_store = Path(runs).expanduser().resolve() if runs is not None else _default_runs()
    run = run_store / job_id
    job = _read_json(run / "job.json")

    with (run / "attempt.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise JobFailedError(f"job {job_id} is already executing") from error

        with _state_lock(run):
            state = _read_json(run / "state.json").get("state")
            if state != "pending":
                raise JobFailedError(f"job {job_id} cannot start from durable state {state!r}")

            if (run / "cancel.json").exists():
                _finish_cancelled(run, run_store, job)
                raise JobFailedError(f"job {job_id} was cancelled before starting")

            _write_state(run, "running", started_at=_now(), pid=os.getpid(),
                         process_identity=_process_identity(os.getpid()))

        try:
            workload = load_workload(run / "spec.json")
            if job.get("workload_digest") != workload.digest:
                raise JobFailedError(f"persisted spec for job {job_id} does not match its job record")

            result_value = job.get("result")
            if not isinstance(result_value, str):
                raise JobFailedError(f"job {job_id} has no valid result location")

            result_path = Path(result_value)

            if executor is None:
                from benchmax.execution.jobs.workload import execute_workload

                executor = execute_workload

            product = job.get("output")
            if not isinstance(product, str):
                raise JobError(f"job {job_id} has no valid output location")

            # Resolve dependencies on the execution host before running the workload.
            from benchmax.execution.jobs.outputs import execute_with_references

            def resolve_dependency(reference: _ReferenceSpec) -> tuple[_ArtifactMetadata, dict[str, Any]]:
                handle = LocalJobService(run_store)._dependency(reference)
                return result_from_document(handle.result()).policy, _read_json(handle.run_directory / "state.json")

            with _worker_process_group():
                result = execute_with_references(workload, Path(product), run=run,
                                                 resolve_dependency=resolve_dependency, executor=executor)

            with _state_lock(run):
                if (run / "cancel.json").exists():
                    raise JobFailedError(f"job {job_id} was cancelled")

                _validate_result(result, workload=workload)
                write_document(result, result_path)
                result_digest = _sha256_file(result_path)
                _publish_result(result, canonical=result_path, output=Path(product))
                _write_state(run, "succeeded", finished_at=_now(), result_sha256=result_digest)
                _release_output_reservation(run_store, job)
        except BaseException as error:
            with _state_lock(run):
                if (run / "cancel.json").exists():
                    _finish_cancelled(run, run_store, job)
                else:
                    _write_state(run, "failed", finished_at=_now(), error=f"{type(error).__name__}: {error}")
                    _release_output_reservation(run_store, job)

            raise

        return result.to_document()
