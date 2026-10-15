"""Reconnect to durable jobs without exposing target-service internals."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from benchmax.execution.jobs.job import Job

from ._render import render_value, table

JobId = Annotated[str, Parameter(help="Durable job ID printed by --submit.")]
WaitFlag = Annotated[bool, Parameter(negative=(), help="Wait for completion and print the result.")]
TargetName = Annotated[str, Parameter(help="Execution target that owns the job.")]
JsonFlag = Annotated[bool, Parameter(name="--json", negative=(), help="Print machine-readable JSON.")]


def job(job_id: JobId, /, *, wait: WaitFlag = False, target: TargetName = "local", as_json: JsonFlag = False) -> int:
    """Show a durable job, or wait for its result."""

    opened = Job.open(job_id, target=target)

    if wait:
        render_value(opened._result_document().to_document(), as_json=as_json)
        return 0

    state = opened.state
    output = opened.output
    status = {
        "id": opened.id,
        "state": state,
        "target": target,
        "output": str(output) if output is not None else None,
    }

    if as_json:
        render_value(status, as_json=True)
    else:
        print("\n".join(table(
            ["job", "state", "target", "output"], [[job_id, state, target, str(output or "")]], indent=""
        )))

    return 0
