"""Read a bounded dependency tree from durable metadata, without loading results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmax.execution.jobs.outputs import dependency, validate_producer
from benchmax.execution.jobs.progress import JobProgress
from benchmax.execution.specs import TrainingSpec, load_workload
from benchmax.execution.targets.base import Target
from benchmax.execution.targets.registry import dependency_target
from benchmax.execution.targets.service import _JOB_ID, _TERMINAL_STATES, JobError, _now, _read_json, _scheduler_id


def _elapsed(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None

    try:
        return max(0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
    except (ValueError, TypeError):
        return None


def read_progress(job_id: str, *, runs: Path, target: Target,
                  query: Callable[[list[str]], dict[str, dict[str, Any]]] | None = None) -> JobProgress:
    """Fetch only this job and its explicit ancestors, batching scheduler reads."""
    checked = _now()
    records: dict[str, tuple[Any, dict[str, Any], dict[str, Any], str | None]] = {}

    def read(current: str, seen: frozenset[str] = frozenset()) -> JobProgress:
        if not isinstance(current, str) or _JOB_ID.fullmatch(current) is None:
            raise JobError("invalid progress job ID")

        if current in seen or len(seen) >= 16:
            raise JobError("invalid or excessively deep progress dependency tree")

        run = runs / current
        record, state = _read_json(run / "job.json"), _read_json(run / "state.json")
        workload = load_workload(run / "spec.json")
        if record.get("workload_digest") != workload.digest:
            raise JobError("persisted workload digest mismatch")

        scheduler_id = _scheduler_id(run)
        children = ()
        reference = dependency(workload)
        if reference is not None:
            try:
                dependency_target(reference.target, target)

                if _JOB_ID.fullmatch(reference.job_id) is None:
                    raise JobError("invalid producer job ID")

                producer = load_workload(runs / reference.job_id / "spec.json")
                validate_producer(reference, producer)
                children = (read(reference.job_id, seen | {current}),)
            except (JobError, OSError, ValueError) as error:
                children = (JobProgress(reference.job_id, "training", "unknown", "unknown", checked,
                                        reason=f"Dependency status unavailable: {error}"),)

        records[current] = (workload, record, state, scheduler_id)
        return JobProgress(current, workload.kind, state["state"], state["state"], checked, dependencies=children)

    tree = read(job_id)
    ids = sorted({sid for _, _, state, sid in records.values()
                  if sid is not None and state["state"] not in _TERMINAL_STATES})
    scheduler = query(ids) if query and ids else {}

    def enrich(node: JobProgress) -> JobProgress:
        if node.job_id not in records:
            return node

        workload, record, saved, sid = records[node.job_id]
        live = scheduler.get(sid, {})
        children = tuple(enrich(child) for child in node.dependencies)
        state = saved["state"]
        reason = saved.get("error")

        if state not in _TERMINAL_STATES:
            state = live.get("state", state)
            reason = live.get("reason") or reason

        phase = {"pending": "queued", "submission_unknown": "unknown"}.get(state, state)

        if state not in _TERMINAL_STATES and live.get("scheduler_state") == "COMPLETED":
            phase = "awaiting_result"
        elif state not in _TERMINAL_STATES and live.get("scheduler_state") == "SUSPENDED":
            phase = "suspended"

        if state in {"pending", "running", "submission_unknown"}:
            failed = [child for child in children if child.state in {"failed", "cancelled"}]
            waiting = [child for child in children if child.state != "succeeded"]

            if failed:
                phase, reason = "dependency_failed", f"Policy producer {failed[0].job_id} {failed[0].state}."
            elif waiting:
                phase, reason = "waiting_for_policy", f"Waiting for the policy from {waiting[0].job_id}."
        elif state in {"failed", "cancelled"} and any(c.state in {"failed", "cancelled"} for c in children):
            reason = (reason + " " if reason else "") + "A policy dependency failed or was cancelled."

        started = saved.get("started_at") or live.get("started_at")
        finished = saved.get("finished_at") or live.get("finished_at")
        total = workload.total_frames if isinstance(workload, TrainingSpec) else None
        completed = total if state == "succeeded" else (0 if state == "pending" else None)
        return replace(node, state=state, phase=phase, reason=reason, dependencies=children, scheduler_id=sid,
                       scheduler_state=live.get("scheduler_state"), submitted_at=record.get("created_at"),
                       started_at=started, finished_at=finished, observed_at=saved.get("updated_at"),
                       elapsed_seconds=live.get("elapsed_seconds", _elapsed(started, finished or checked)),
                       expected_start=live.get("expected_start"), completed=completed if total is not None else None,
                       total=total, unit="frames" if total is not None else None,
                       warnings=tuple(live.get("warnings", ())))

    return enrich(tree)
