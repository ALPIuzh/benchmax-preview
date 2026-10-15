"""Cancellation of one durable local attempt, without signalling shared processes."""

from __future__ import annotations

import fcntl
import os
import signal
import time
from pathlib import Path
from typing import Any

from benchmax.execution.targets.local.service import JobHandle, _finish_cancelled, _process_identity
from benchmax.execution.targets.service import _TERMINAL_STATES, JobError, _now, _read_json, _state_lock
from benchmax.utils.files import _exclusive_file_lock
from benchmax.utils.json import write_document

_STOP_TIMEOUT_SECONDS = 5.0


def _process(state: dict[str, Any], run: Path) -> tuple[int, str] | None:
    launcher = _read_json(run / "launcher.json")
    pid = state.get("pid", launcher.get("pid"))
    identity = state.get("process_identity", launcher.get("process_identity"))

    if pid is None and state["state"] == "pending":
        # A late-launching attempt must check the durable state before computing.
        # The submitter's owner_pid is never a cancellation target.
        return None

    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1 or pid == os.getpid():
        raise JobError("cannot safely cancel: job has no separate, recorded attempt process")

    if not isinstance(identity, str) or not identity:
        raise JobError("cannot safely cancel: attempt process identity is unavailable")

    try:
        group, session = os.getpgid(pid), os.getsid(pid)
    except ProcessLookupError:
        return None

    if group != pid or session != pid:
        raise JobError("cannot safely cancel: attempt does not own a dedicated process group and session")

    if _process_identity(pid) != identity:
        raise JobError("cannot safely cancel: recorded attempt process identity no longer matches")

    return pid, identity


def cancel_local_job(handle: JobHandle) -> None:
    run = handle.run_directory

    # Separate cancellers serialize here; attempts only take state.lock briefly.
    with _exclusive_file_lock(run / "cancel.lock"):

        with _state_lock(run):
            state = _read_json(run / "state.json")
            if state["state"] in _TERMINAL_STATES:
                return

            if state["state"] not in {"pending", "running"}:
                raise JobError(f"cannot cancel local job in state {state['state']!r}")

            process = _process(state, run)

            if not (run / "cancel.json").exists():
                write_document({"schema_version": 1, "requested_at": _now()},
                               run / "cancel.json", replace=False)

            if process is not None:
                # Recheck after persisting the request. SIGKILL stops native code
                # and every member of this private group without cleanup handlers
                # racing result publication or leaving a compute child behind.
                current = _process(state, run)
                if current is not None:
                    try:
                        os.killpg(current[0], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except OSError as error:
                        raise JobError(f"could not stop local attempt {current[0]}: {error}") from error

        deadline = time.monotonic() + _STOP_TIMEOUT_SECONDS

        with (run / "attempt.lock").open("a+b") as attempt:
            while True:
                try:
                    fcntl.flock(attempt.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise JobError("cancellation requested but the attempt has not stopped; retry cancellation")

                    time.sleep(0.02)

            with _state_lock(run):
                state = _read_json(run / "state.json")
                if state["state"] not in _TERMINAL_STATES:
                    _finish_cancelled(run, handle._runs, _read_json(run / "job.json"))
