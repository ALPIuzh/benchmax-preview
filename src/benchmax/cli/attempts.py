"""Private entry point executed inside one already allocated local attempt."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

AttemptId = Annotated[str, Parameter(help="Durable attempt ID allocated by the target.")]


def attempt(job_id: AttemptId, /) -> int:
    """Execute one persisted attempt owned by the local run store."""

    from benchmax.execution.targets.local.attempt import run_local_attempt

    run_local_attempt(job_id)
    return 0
