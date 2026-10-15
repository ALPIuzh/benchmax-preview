"""Scoped scheduler observations for a progress tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmax.execution.jobs.progress_store import read_progress
from benchmax.execution.targets.service import JobError
from benchmax.execution.targets.slurm.target import SlurmTarget

_FAILED = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED"}


def _scheduler_state(value: str) -> dict[str, str]:
    name = value.split()[0].rstrip("+") if value.split() else "UNKNOWN"
    state = "cancelled" if name == "CANCELLED" else "failed" if name in _FAILED else None

    if name in {"RUNNING", "COMPLETING"}:
        state = "running"
    elif name in {"PENDING", "CONFIGURING", "SUSPENDED"}:
        state = "pending"

    # Accounting completion alone cannot confirm publication of a BenchMAX result.
    result = {"scheduler_state": name}

    if state:
        result["state"] = state

    if name == "COMPLETED":
        result["reason"] = "Slurm completed; BenchMAX result publication has not yet been confirmed."

    return result


def slurm_progress(target: SlurmTarget, job_id: str) -> dict[str, Any]:
    from benchmax.execution.targets.slurm.bridge import _run

    def query(ids: list[str]) -> dict[str, dict[str, Any]]:
        values: dict[str, dict[str, Any]] = {sid: {} for sid in ids}
        warnings = []

        try:
            command = ["sacct", "-n", "-P", "-j", ",".join(ids),
                       "--format=JobIDRaw,State%30,ExitCode,Start,End,ElapsedRaw"]
            accounting = _run(command, timeout=target.command_timeout)

            for line in accounting.splitlines():
                fields = line.split("|")
                if len(fields) < 6 or fields[0] not in values:
                    continue

                item = values[fields[0]]
                item.update(_scheduler_state(fields[1]))

                if item.get("state") in {"failed", "cancelled"}:
                    item["reason"] = f"Slurm {fields[1]} (exit {fields[2]})."

                for key, value in (("started_at", fields[3]), ("finished_at", fields[4])):
                    if value and value not in {"Unknown", "N/A", "None"}:
                        item[key] = value

                if fields[5].isdigit():
                    item["elapsed_seconds"] = float(fields[5])
        except JobError as error:
            warnings.append(f"Accounting unavailable: {error}")

        queue_ids = [sid for sid, item in values.items() if item.get("state") not in {"failed", "cancelled"}
                     and item.get("scheduler_state") != "COMPLETED"]

        try:
            queue = (_run(["squeue", "--noheader", f"--jobs={','.join(queue_ids)}", "--format=%i|%T|%r|%S"],
                          timeout=target.command_timeout) if queue_ids else "")

            for line in queue.splitlines():
                fields = [field.strip() for field in line.split("|")]
                if len(fields) < 4 or fields[0] not in values:
                    continue

                item = values[fields[0]]
                item.update(_scheduler_state(fields[1]))
                item["reason"] = fields[2] if fields[2] not in {"None", "(null)", ""} else None
                if item.get("state") == "pending" and fields[3] not in {"N/A", "Unknown", "None", ""}:
                    item["expected_start"] = fields[3]
        except JobError as error:
            warnings.append(f"Queue status unavailable: {error}")

        for item in values.values():
            if not item:
                item["warnings"] = ["No current scheduler record; showing the last saved BenchMAX state.", *warnings]
            elif warnings:
                item["warnings"] = warnings

        return values

    return read_progress(job_id, runs=Path(target.runs), target=target, query=query).to_document()
