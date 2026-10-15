"""Small JSON control endpoint in the target's installed Python environment.

Only the ``attempt`` entry point runs the workload, inside an allocation.
The login-node endpoint handles durable files and single scheduler requests.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmax.execution.artifacts.results import result_from_document
from benchmax.utils.hashing import _sha256_file

from benchmax.execution.targets.service import (  # isort: skip
    _JOB_ID, _TERMINAL_STATES, JobError, JobFailedError, JobSubmissionUnknownError, _now, _publish_result, _read_json,
    _release_output_reservation, _reserve_output, _scheduler_id, _state_lock, _validate_result, _write_state
)
from benchmax.execution.specs import (  # isort: skip
    TrainingSpec, _ArtifactMetadata, _ReferenceSpec, load_workload, workload_from_document
)
from benchmax.execution.targets.slurm.progress import _scheduler_state
from benchmax.execution.targets.slurm.target import SlurmTarget
from benchmax.utils.json import write_document


def package_fingerprint() -> str:
    """Identify installed Python/config source without hashing platform binaries."""
    root = Path(__file__).parents[3]
    digest = hashlib.sha256()

    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml"}):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())

    return digest.hexdigest()


def _run(command: list[str], *, timeout: float, input: str | None = None) -> str:
    try:
        result = subprocess.run(command, input=input, text=True, capture_output=True, timeout=timeout, check=True)
    except subprocess.TimeoutExpired as error:
        raise JobError(f"command timed out: {command[0]}; inspect the attempt before retrying submission") from error
    except (OSError, subprocess.CalledProcessError) as error:
        detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise JobError(f"{command[0]} failed: {detail}") from error

    return result.stdout.strip()


def _target(document: dict[str, Any]) -> SlurmTarget:
    settings = dict(document)
    settings.pop("kind")
    return SlurmTarget(**settings)


def _directory(target: SlurmTarget, job_id: str) -> Path:
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise JobError("invalid Slurm attempt ID")

    return Path(target.runs) / job_id


def _dependency_snapshot(reference, target: SlurmTarget) -> dict[str, Any]:
    from benchmax.execution.jobs.outputs import validate_producer
    from benchmax.execution.targets.registry import dependency_target

    producer = dependency_target(reference.target, target)
    snapshot = _snapshot(producer, _directory(producer, reference.job_id), query_scheduler=False)
    validate_producer(reference, workload_from_document(snapshot["workload"]))

    if snapshot["state"] not in {"pending", "running", "succeeded"}:
        raise JobError(f"dependency {reference.job_id} is {snapshot['state']}")

    if snapshot["state"] != "succeeded" and not snapshot["scheduler_id"]:
        raise JobError(f"dependency {reference.job_id} has an unknown submission outcome")

    return snapshot


def _submission(target: SlurmTarget, request: dict[str, Any], *, source_digest: str) -> dict[str, Any]:
    from benchmax.execution.jobs.outputs import dependency
    from benchmax.execution.specs import validate_workload_resolution

    workload = workload_from_document(request["workload"])
    validate_workload_resolution(workload)
    reference = dependency(workload)
    upstream = _dependency_snapshot(reference, target) if reference is not None else None
    run = _directory(target, request["id"])
    run.mkdir(parents=True, exist_ok=False)
    write_document(workload, run / "spec.json")
    output = request.get("output")

    if output is None:
        product = Path(target.artifacts or target.runs) / run.name / "result.json"
    else:
        path = Path(output)
        product = path if path.is_absolute() else Path(target.artifacts or target.workspace) / path

    product = product.resolve()
    if product.exists():
        raise JobError(f"output already exists: {product}")

    # The selected environment/settings are a snapshot, independent of profile edits.
    record = {"id": run.name, "target": target.to_document(), "output": str(product), "output_reserved": True,
              "workload_digest": workload.digest, "source_digest": request["source_digest"],
              "target_source_digest": source_digest, "created_at": _now()}
    write_document(record, run / "job.json")
    _write_state(run, "pending")

    try:
        _reserve_output(Path(target.runs), product, run.name)
    except (OSError, JobError) as error:
        _write_state(run, "failed", error=str(error))
        raise

    script = "\n".join([
        "#!/bin/sh", "set -eu", "umask 0007", f"export PYTHONHASHSEED={workload.python_hash_seed}",
        "export PYTHONNOUSERSITE=1", "unset PYTHONHOME PYTHONPATH",
        f"cd {shlex.quote(target.workspace)}",
        f"export OMP_NUM_THREADS={target.cpus} OPENBLAS_NUM_THREADS={target.cpus} MKL_NUM_THREADS={target.cpus}",
        "exec " + shlex.join([target.python, "-m", "benchmax.execution.targets.slurm.bridge", "attempt", str(run)]), "",
    ])
    command = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", f"--cpus-per-task={target.cpus}",
               f"--mem={target.memory_mb}M", f"--time={target.time_limit}", "--no-requeue",
               f"--job-name=benchmax-{run.name}", f"--output={run / 'attempt.log'}", f"--chdir={target.workspace}"]

    for key in ("account", "partition"):
        if value := getattr(target, key):
            command.append(f"--{key}={value}")

    if target.gpus:
        command.append(f"--gpus={target.gpus}")

    if upstream is not None and upstream["state"] != "succeeded":
        command.extend([f"--dependency=afterok:{upstream['scheduler_id']}", "--kill-on-invalid-dep=yes"])

    # Never inherit SBATCH_* resource overrides from a user's login shell.
    environment = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}

    try:
        (run / "attempt.sh").write_text(script, encoding="utf-8")
        result = subprocess.run([*command, str(run / "attempt.sh")], text=True, capture_output=True,
                                timeout=target.command_timeout, env=environment, check=True)
    except subprocess.TimeoutExpired as error:
        _submission_unknown(run, "sbatch timed out; inspect Slurm before retrying")

        raise JobSubmissionUnknownError(f"submission outcome unknown; reconnect to {run.name} before resubmitting",
                                        job_id=run.name) from error
    except (OSError, subprocess.CalledProcessError) as error:
        detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        with _state_lock(run):
            if _read_json(run / "state.json")["state"] == "pending":
                _write_state(run, "failed", error=f"submission setup or sbatch failed: {detail}")
                _release_output_reservation(Path(target.runs), record)
        raise JobError(f"sbatch failed for {run.name}: {detail}") from error

    scheduler_id = result.stdout.strip().split(";")[0]

    if not scheduler_id.isdigit():
        _submission_unknown(run, "sbatch did not return a recognizable ID")

        raise JobSubmissionUnknownError(f"unrecognized sbatch response for {run.name}; submission may have succeeded",
                                        job_id=run.name)

    try:
        write_document({"id": scheduler_id}, run / "scheduler.json")
        return _snapshot(target, run, query_scheduler=False)
    except (OSError, JobError, ValueError) as error:
        raise JobSubmissionUnknownError(f"submission outcome unknown; reconnect to {run.name}: {error}",
                                        job_id=run.name) from error


def _submission_unknown(run: Path, message: str) -> None:
    with contextlib.suppress(OSError, JobError, ValueError), _state_lock(run):
        if _read_json(run / "state.json")["state"] == "pending":
            _write_state(run, "submission_unknown", error=message)


def _snapshot(target: SlurmTarget, run: Path, *, query_scheduler: bool = True) -> dict[str, Any]:
    record = _read_json(run / "job.json")
    workload = load_workload(run / "spec.json")
    if record["workload_digest"] != workload.digest:
        raise JobError("persisted workload digest mismatch")

    state = _read_json(run / "state.json")
    scheduler_id = _scheduler_id(run)

    if query_scheduler and state["state"] in {"pending", "running"} and scheduler_id:
        # One query per status request, scoped to this exact scheduler ID.
        output = _run(["sacct", "-n", "-P", "-j", scheduler_id, "--format=JobIDRaw,State,ExitCode"],
                      timeout=target.command_timeout)

        for row in output.splitlines():
            fields = row.split("|")
            if len(fields) < 3 or fields[0] != scheduler_id:
                continue

            observed = _scheduler_state(fields[1])
            status = observed["scheduler_state"]
            # The attempt may have published its result while sacct was running.
            with _state_lock(run):
                state = _read_json(run / "state.json")
                if state["state"] not in {"pending", "running"}:
                    break

                if observed.get("state") in {"failed", "cancelled"} or status == "COMPLETED":
                    final = observed.get("state", "failed")
                    message = f"Slurm {status} ({fields[2]}) without a completed result"
                    _write_state(run, final, error=message, finished_at=_now())
                    _release_output_reservation(Path(target.runs), record)
                elif observed.get("state") == "running":
                    state = {**state, "state": "running"}

            break

        # Preserve transient RUNNING status even before the attempt writes it.
        saved = _read_json(run / "state.json")
        if saved["state"] != "pending":
            state = saved

    if query_scheduler and not scheduler_id and state["state"] == "pending":
        state = {**state, "state": "submission_unknown"}

    document = {"id": run.name, "state": state["state"], "error": state.get("error"), "output": record["output"],
                "workload": workload.to_document(), "scheduler_id": scheduler_id}

    if state["state"] == "succeeded":
        path = run / "result.json"
        if _sha256_file(path) != state.get("result_sha256"):
            raise JobError("result digest mismatch")

        result = result_from_document(_read_json(path))
        _validate_result(result, workload=workload)
        document["result"] = result.to_document()

    return document


def run_attempt(run: Path) -> None:
    """Execute only inside the allocation, including isolated evaluation workers."""

    scheduler_id = os.environ.get("SLURM_JOB_ID", "")
    if not scheduler_id.isdigit():
        raise JobError("Slurm attempts must run inside a Slurm allocation")

    import fcntl

    with (run / "attempt.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        with _state_lock(run):
            if _read_json(run / "state.json")["state"] not in {"pending", "submission_unknown"}:
                raise JobError("attempt has already started or terminated")

            write_document({"id": scheduler_id}, run / "scheduler.json")
            _write_state(run, "running", started_at=_now())
        record = _read_json(run / "job.json")

        try:
            workload = load_workload(run / "spec.json")
            if workload.digest != record["workload_digest"]:
                raise JobError("workload has drifted since submission")

            target = _target(record["target"])
            record["execution_source_digest"] = package_fingerprint()
            write_document(record, run / "job.json")

            if target.strict_source and record["execution_source_digest"] != record["source_digest"]:
                raise JobError("installed BenchMAX source has drifted since submission (strict_source=True)")

            from benchmax.execution.jobs.workload import execute_workload
            from benchmax.execution.specs import validate_workload_resolution

            validate_workload_resolution(workload)

            # Hash seed must be set before Python starts; the script supplies it.
            if isinstance(workload, TrainingSpec) and os.environ.get("PYTHONHASHSEED") != str(workload.python_hash_seed):
                raise JobError("attempt Python hash seed does not match its workload")

            from benchmax.execution.jobs.outputs import execute_with_references

            def resolve_dependency(reference: _ReferenceSpec) -> tuple[_ArtifactMetadata, dict[str, Any]]:
                snapshot = _dependency_snapshot(reference, target)
                if snapshot["state"] != "succeeded":
                    raise JobFailedError(f"dependency {reference.job_id} has no successful result")

                state = _read_json(_directory(target, reference.job_id) / "state.json")
                return result_from_document(snapshot["result"]).policy, state

            result = execute_with_references(workload, Path(record["output"]), run=run,
                                             resolve_dependency=resolve_dependency, executor=execute_workload)
            _validate_result(result, workload=workload)
            write_document(result, run / "result.json")

            with _state_lock(run):
                _publish_result(result, canonical=run / "result.json", output=Path(record["output"]))
                _write_state(run, "succeeded", finished_at=_now(), result_sha256=_sha256_file(run / "result.json"))
        except BaseException as error:
            with _state_lock(run):
                _write_state(run, "failed", finished_at=_now(), error=f"{type(error).__name__}: {error}")
            raise
        finally:
            _release_output_reservation(run.parent, record)


def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("schema_version") != 1:
        raise JobError("unsupported Slurm control protocol")

    target = _target(request["target"])
    operation = request["operation"]

    if operation == "inspect":
        from benchmax.registry import _inspect_entry

        report = _inspect_entry("algorithm", request["name"])
        report["scope"] = "login environment; compute-node GPU/runtime availability is not tested"
        return report

    if operation in {"submit", "audit"}:
        source_digest = package_fingerprint()
        if target.strict_source and request["source_digest"] != source_digest:
            raise JobError("installed BenchMAX source differs from the caller (strict_source=True); "
                           "install the same snapshot on the target")

    if operation == "submit":
        return _submission(target, request, source_digest=source_digest)

    if operation == "audit":
        from benchmax.execution.audit.resources import audit_workload
        from benchmax.execution.specs import validate_workload_resolution
        from benchmax.providers import list_providers

        workload = workload_from_document(request["workload"])
        validate_workload_resolution(workload)
        document = audit_workload(workload).to_document()
        report = document["evidence"]
        allocation = {k: getattr(target, k) for k in ("cpus", "memory_mb", "time_limit", "gpus")}
        report["machine"] = {"scope": "login environment; compute-node capacity is not measured",
                             "python": sys.executable, "providers": list_providers(), "allocation": allocation,
                             "source_digest": source_digest, "caller_source_digest": request["source_digest"]}
        report["target"] = "slurm"
        report["estimates"]["storage"] = {k: getattr(target, k) for k in ("workspace", "runs", "artifacts")}
        report["findings"].append("BenchMAX and required providers must already be installed on the target.")
        report["estimates"]["transfers"] = {"source_bytes": 0, "policy_bytes": 0, "workload": "JSON control message"}
        return document

    if operation in {"policy-cache", "upload"}:
        from benchmax.execution.targets.slurm.transfer import cached_policy

        return cached_policy(Path(target.artifacts or target.runs) / "policy-cache", request["digest"],
                             request["manifest"], limit=target.transfer_limit_bytes,
                             bundle=request["bundle"] if operation == "upload" else None)

    run = _directory(target, request["id"])

    if operation == "status":
        return _snapshot(target, run)

    if operation == "progress":
        from benchmax.execution.targets.slurm.progress import slurm_progress

        return slurm_progress(target, run.name)

    if operation == "cancel":
        state = _read_json(run / "state.json")["state"]

        if state not in _TERMINAL_STATES:
            scheduler_id = _scheduler_id(run)
            if scheduler_id is None:
                raise JobError(f"cannot cancel {run.name}: no persisted Slurm scheduler ID")

            _run(["scancel", scheduler_id], timeout=target.command_timeout)

        return {"id": run.name}

    if operation == "download":
        from benchmax.execution.targets.slurm.transfer import pack_policy
        snapshot = _snapshot(target, run, query_scheduler=False)
        if snapshot["state"] != "succeeded":
            raise JobError("policy download requires a succeeded training job")

        result = result_from_document(snapshot["result"])
        return pack_policy(result.policy, limit=target.transfer_limit_bytes)

    raise JobError(f"unknown Slurm operation {operation!r}")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "attempt":
        run_attempt(Path(sys.argv[2]))
        return

    try:
        request = json.load(sys.stdin)

        with contextlib.redirect_stdout(sys.stderr):
            response = dispatch(request)

        print(json.dumps({"ok": True, "value": response}, allow_nan=False))
    except JobSubmissionUnknownError as error:
        print(json.dumps({"ok": False, "error": str(error), "submission_unknown": True}))
        sys.exit(1)
    except Exception as error:  # noqa: BLE001 - serialize failures at the process boundary
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
