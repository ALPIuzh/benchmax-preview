"""Durable local target implementation and filesystem-backed job handles."""

from __future__ import annotations

import asyncio
import fcntl
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from benchmax.execution.artifacts.results import result_from_document
from benchmax.utils.hashing import _sha256_file

from benchmax.execution.targets.service import (  # isort: skip
    _JOB_ID, _TERMINAL_STATES, JobError, JobFailedError, _JobDeadline, _now, _publish_result, _read_json,
    _release_output_reservation, _reserve_output, _state_lock, _validate_result, _write_state
)
from benchmax.execution.specs import WorkloadSpec, _ReferenceSpec, load_workload
from benchmax.utils.json import write_document

if TYPE_CHECKING:
    from benchmax.execution.audit.resources import AuditReport

_RUNS_ENVIRONMENT = "BENCHMAX_RUNS"
_PROCESS_EXIT_TIMEOUT_SECONDS = 0.5
_PROCESS_EXIT_POLL_SECONDS = 0.01


class Launcher(Protocol):
    """Start one already-persisted attempt without shell interpretation."""

    def __call__(self, command: list[str], environment: dict[str, str], log: Path) -> int | None: ...


def _finish_cancelled(run: Path, runs: Path, job: Mapping[str, Any]) -> None:
    """Called with the attempt and state locks held, once computation has stopped."""
    _write_state(run, "cancelled", finished_at=_now(), error="job cancelled by user")
    _release_output_reservation(runs, job)


def _process_identity(pid: int) -> str | None:
    """Return an OS birth identity that changes when a PID is reused."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        stat = ""

    if stat:
        try:
            # starttime is field 22; splitting after ')' avoids spaces in comm.
            starttime = stat.rsplit(")", 1)[1].split()[19]
        except (IndexError, ValueError):
            pass
        else:
            return f"proc:{starttime}"

    try:
        completed = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)], check=False, capture_output=True, text=True, timeout=1
        )
    except (OSError, subprocess.SubprocessError):
        return None

    started = completed.stdout.strip()
    return f"ps:{started}" if completed.returncode == 0 and started else None


def _process_alive(pid: int, identity: str | None = None) -> bool:
    if identity is not None:
        current_identity = _process_identity(pid)
        if current_identity is not None and current_identity != identity:
            return False

    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        waited = 0

    if waited == pid:
        return False

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    return True


def _may_signal_launched_process(pid: int, identity: str | None) -> bool:
    current_identity = _process_identity(pid)
    if identity is not None and current_identity is not None:
        return current_identity == identity

    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False

    return waited == 0


def _wait_exit(pid: int, identity: str | None) -> bool:
    deadline = time.monotonic() + _PROCESS_EXIT_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        if not _process_alive(pid, identity):
            return True

        time.sleep(_PROCESS_EXIT_POLL_SECONDS)

    return False


def _terminate_launched_process(pid: int, identity: str | None) -> bool:
    """Stop an unrecorded local child before its output claim is released."""

    if not _process_alive(pid, identity):
        return True

    if not _may_signal_launched_process(pid, identity):
        return False

    try:
        # Stop the private group before its leader can exit and orphan workers.
        if os.getpgid(pid) == pid and os.getsid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    if _wait_exit(pid, identity):
        return True

    if not _may_signal_launched_process(pid, identity):
        return not _process_alive(pid, identity)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    return _wait_exit(pid, identity)


def _default_runs() -> Path:
    configured = os.environ.get(_RUNS_ENVIRONMENT)
    if configured:
        return Path(configured).expanduser().resolve()

    return Path.home().joinpath(".local", "share", "benchmax", "runs")


def _launch_process(command: list[str], environment: dict[str, str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)

    with log.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, env=environment, close_fds=True,
            start_new_session=True
        )

    return process.pid


@dataclass(frozen=True, slots=True)
class JobHandle:
    """A durable identifier that can be reconstructed in another process."""

    id: str
    _runs: Path
    _poll_seconds: float = 0.2

    def __post_init__(self) -> None:
        if _JOB_ID.fullmatch(self.id) is None:
            raise JobError(f"invalid job ID {self.id!r}")

    @property
    def run_directory(self) -> Path:
        return self._runs / self.id

    def progress(self) -> dict[str, Any]:
        from benchmax.execution.jobs.progress_store import read_progress
        from benchmax.execution.targets.base import LocalTarget

        return read_progress(self.id, runs=self._runs, target=LocalTarget(runs=str(self._runs))).to_document()

    def cancel(self) -> None:
        """Stop only this attempt's verified process group and persist cancellation."""
        from benchmax.execution.targets.local.cancellation import cancel_local_job

        cancel_local_job(self)

    @property
    def state(self) -> str:
        document = _read_json(self.run_directory / "state.json")

        if document.get("state") in {"pending", "running"}:
            self._reconcile_dead_process(document)
            document = _read_json(self.run_directory / "state.json")

        state = document.get("state")
        if not isinstance(state, str):
            raise JobError(f"job {self.id} has no valid state")

        return state

    @property
    def workload(self) -> WorkloadSpec:
        """Return the validated portable workload owned by this handle."""

        return load_workload(self.run_directory / "spec.json")

    @property
    def output(self) -> Path:
        output = _read_json(self.run_directory / "job.json").get("output")
        if not isinstance(output, str):
            raise JobError(f"job {self.id} has no valid output location")

        return Path(output)

    @property
    def _result_path(self) -> Path:
        result = _read_json(self.run_directory / "job.json").get("result")
        if not isinstance(result, str):
            raise JobError(f"job {self.id} has no valid canonical result location")

        return Path(result)

    def _read_result(self, state: Mapping[str, Any]) -> dict[str, Any]:
        result_path = self._result_path
        expected_digest = state.get("result_sha256")

        try:
            verified = isinstance(expected_digest, str) and _sha256_file(result_path) == expected_digest
        except OSError:
            verified = False

        if not verified:
            raise JobFailedError(f"job {self.id} canonical result failed verification")

        try:
            result = result_from_document(_read_json(result_path))
        except ValueError as error:
            raise JobFailedError(f"job {self.id} canonical result is invalid: {error}") from error

        job = _read_json(self.run_directory / "job.json")
        workload = load_workload(self.run_directory / "spec.json")
        if job.get("workload_digest") != workload.digest:
            raise JobError(f"job {self.id} has no workload digest")

        _validate_result(result, workload=workload)
        return result.to_document()

    def _reconcile_dead_process(self, state: Mapping[str, Any]) -> None:
        launcher_path = self.run_directory / "launcher.json"
        if not launcher_path.is_file():
            return

        launcher = _read_json(launcher_path)
        pid = state.get("pid", launcher.get("pid"))
        identity = state.get("process_identity", launcher.get("process_identity"))

        if not isinstance(pid, int) and launcher.get("status") == "launching":
            pid = launcher.get("owner_pid")
            identity = launcher.get("owner_process_identity")

        if not isinstance(identity, str):
            identity = None

        if not isinstance(pid, int) or _process_alive(pid, identity):
            return

        with (self.run_directory / "attempt.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # A worker won the launch race and owns the state transition.
                return

            self._reconcile_locked()

    def _reconcile_locked(self) -> None:
        with _state_lock(self.run_directory):
            current = _read_json(self.run_directory / "state.json")
            if current.get("state") not in {"pending", "running"}:
                return

            result_path = self._result_path
            job = _read_json(self.run_directory / "job.json")

            if (self.run_directory / "cancel.json").exists():
                _finish_cancelled(self.run_directory, self._runs, job)
                return

            workload = load_workload(self.run_directory / "spec.json")

            try:
                result = result_from_document(_read_json(result_path))

                if job.get("workload_digest") != workload.digest:
                    raise JobError("job record has no matching workload digest")

                _validate_result(result, workload=workload)
                result_digest = _sha256_file(result_path)
                _publish_result(result, canonical=result_path, output=self.output)
            except (JobError, OSError, ValueError) as error:
                _write_state(
                    self.run_directory, "failed", finished_at=_now(),
                    error=f"attempt process exited without a valid result: {error}"
                )
            else:
                _write_state(
                    self.run_directory, "succeeded", finished_at=_now(), result_sha256=result_digest, reconciled=True
                )

            _release_output_reservation(self._runs, job)

    def _result_if_ready(self) -> dict[str, Any] | None:
        for reconcile in (True, False):
            state = _read_json(self.run_directory / "state.json")
            name = state.get("state")
            if name == "succeeded":
                return self._read_result(state)

            if name in {"failed", "cancelled"}:
                detail = state.get("error") or f"job ended in state {name}"
                raise JobFailedError(f"job {self.id} failed: {detail}")

            if name not in {"pending", "running"}:
                raise JobError(f"job {self.id} has unknown state {name!r}")

            if reconcile:
                self._reconcile_dead_process(state)

        return None

    def result(self, timeout: float | None = None) -> dict[str, Any]:
        """Wait for and return the persisted result document."""

        deadline = _JobDeadline(self.id, timeout)
        poll_seconds = self._poll_seconds

        while True:
            result = self._result_if_ready()
            if result is not None:
                return result

            time.sleep(deadline.delay(poll_seconds))
            poll_seconds = min(poll_seconds * 1.5, 5.0)

    async def result_async(self, timeout: float | None = None) -> dict[str, Any]:
        """Wait without blocking an event-loop thread."""

        deadline = _JobDeadline(self.id, timeout)
        poll_seconds = self._poll_seconds

        while True:
            result = await asyncio.to_thread(self._result_if_ready)
            if result is not None:
                return result

            await asyncio.sleep(deadline.delay(poll_seconds))
            poll_seconds = min(poll_seconds * 1.5, 5.0)


class LocalJobService:
    """Persist work locally and execute it in a detached BenchMAX process."""

    def __init__(self, runs: str | os.PathLike[str] | None = None, *, launcher: Launcher = _launch_process) -> None:
        self._runs = Path(runs) if runs is not None else _default_runs()
        self._runs = self._runs.expanduser().resolve()
        self._launcher = launcher

    def inspect_algorithm(self, name: str) -> dict[str, Any]:
        from benchmax.registry import _inspect_entry

        return _inspect_entry("algorithm", name)

    def _reconcile_output_owner(self, owner: str) -> None:
        JobHandle(owner, self._runs)._reconcile_dead_process(_read_json(self._runs / owner / "state.json"))

    def _dependency(self, reference: _ReferenceSpec) -> JobHandle:
        """Check a local producer without waiting for its result."""
        from benchmax.execution.jobs.outputs import validate_producer
        from benchmax.execution.targets.base import LocalTarget
        from benchmax.execution.targets.registry import dependency_target

        dependency_target(reference.target, LocalTarget(runs=str(self._runs)))
        handle = self.open(reference.job_id)
        validate_producer(reference, handle.workload)
        if handle.state in {"failed", "cancelled"}:
            raise JobFailedError(f"dependency {reference.job_id} {handle.state}")

        return handle

    def submit(self, workload: WorkloadSpec, *, output: str | os.PathLike[str] | None = None) -> JobHandle:
        """Persist then launch one attempt, returning its durable handle."""

        from benchmax.execution.jobs.outputs import dependency

        if reference := dependency(workload):
            self._dependency(reference)

        job_id = f"{workload.kind}-{uuid.uuid4().hex[:16]}"
        requested_output = Path(output).expanduser().resolve() if output is not None else None
        run = self._runs / job_id
        result = (run / "result.json").resolve()
        product = requested_output or result
        job_record = {
            "schema_version": 1,
            "id": job_id,
            "workload_digest": workload.digest,
            "output": str(product),
            "output_reserved": requested_output is not None,
            "result": str(result),
            "created_at": _now(),
        }
        run.mkdir(parents=True, exist_ok=False)
        write_document(workload, run / "spec.json")
        write_document(job_record, run / "job.json")
        _write_state(run, "pending")
        write_document(
            {
                "schema_version": 1,
                "status": "launching",
                "pid": None,
                "owner_pid": os.getpid(),
                "owner_process_identity": _process_identity(os.getpid()),
            },
            run / "launcher.json",
        )

        if requested_output is not None:
            with _state_lock(run):
                if _read_json(run / "state.json")["state"] == "cancelled":
                    return JobHandle(job_id, self._runs)

                try:
                    requested_output.parent.mkdir(parents=True, exist_ok=True)
                    _reserve_output(self._runs, requested_output, job_id, reconcile=self._reconcile_output_owner)
                except BaseException as error:
                    _write_state(run, "failed", finished_at=_now(),
                                 error=f"output reservation {type(error).__name__}: {error}")
                    raise

        command = [
            sys.executable,
            "-m",
            "benchmax.cli",
            "_attempt",
            job_id,
        ]
        environment = dict(os.environ)
        environment[_RUNS_ENVIRONMENT] = str(self._runs)

        environment["PYTHONHASHSEED"] = str(workload.python_hash_seed)

        try:
            pid = self._launcher(command, environment, run / "attempt.log")
        except BaseException as error:
            with _state_lock(run):
                if _read_json(run / "state.json")["state"] not in _TERMINAL_STATES:
                    _write_state(run, "failed", finished_at=_now(), error=f"launcher {type(error).__name__}: {error}")

                _release_output_reservation(self._runs, _read_json(run / "job.json"))

            raise

        process_identity = _process_identity(pid) if pid is not None else None

        if pid is None:
            launcher_record = {"schema_version": 1, "status": "external", "pid": None}
        else:
            launcher_record = {
                "schema_version": 1,
                "status": "launched",
                "pid": pid,
                "process_identity": process_identity,
            }

        try:
            write_document(launcher_record, run / "launcher.json")
        except BaseException as error:
            stopped = pid is None or _terminate_launched_process(pid, process_identity)

            with _state_lock(run):
                state = _read_json(run / "state.json").get("state")

                if state in _TERMINAL_STATES:
                    _release_output_reservation(self._runs, job_record)
                    return JobHandle(job_id, self._runs)

                if stopped:
                    _write_state(run, "failed", finished_at=_now(),
                                 error=f"launch metadata {type(error).__name__}: {error}; untracked attempt terminated")
                    _release_output_reservation(self._runs, job_record)

            raise JobError(f"job {job_id} launched but its process metadata could not be persisted: {error}") from error

        return JobHandle(job_id, self._runs)

    def audit(self, workload: WorkloadSpec) -> AuditReport:
        from benchmax.execution.audit.resources import audit_workload

        return audit_workload(workload, target="local")

    def open(self, job_id: str) -> JobHandle:
        """Reconstruct a handle for one job in this run store."""

        if _JOB_ID.fullmatch(job_id) is None:
            raise JobError(f"invalid job ID {job_id!r}")

        run = self._runs / job_id
        if not (run / "job.json").is_file():
            raise JobError(f"unknown local job {job_id!r} in {self._runs}")

        return JobHandle(job_id, self._runs)
