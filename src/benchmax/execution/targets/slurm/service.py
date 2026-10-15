"""Client-side Slurm submission, reconnection, and explicit result transfer."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from benchmax.execution.artifacts.results import WorkloadResult, result_from_document
from benchmax.execution.audit.resources import AuditReport
from benchmax.execution.specs import EvaluationSpec, WorkloadSpec, _ArtifactMetadata, workload_from_document
from benchmax.execution.targets.service import (
    _TERMINAL_STATES,
    JobError,
    JobFailedError,
    JobSubmissionUnknownError,
    _JobDeadline,
    _validate_result,
)
from benchmax.execution.targets.slurm.target import SlurmTarget
from benchmax.utils.hashing import _digest_bytes

_BRIDGE = "benchmax.execution.targets.slurm.bridge"


class SlurmJobService:
    def __init__(self, target: SlurmTarget) -> None:
        self.target = target

    def inspect_algorithm(self, name: str) -> dict[str, Any]:
        report = self._request("inspect", name=name)
        if (report.get("name") != name or report.get("group") != "algorithm"
                or not isinstance(report.get("available"), bool) or not isinstance(report.get("definition"), dict)
                or (report.get("available") or report.get("config") is not None)
                and not isinstance(report.get("config"), dict)
                or any(not isinstance(report.get(key), str)
                       for key in ("provider", "error", "python", "workspace", "scope"))):
            raise JobError("invalid algorithm inspection response")

        return report

    @property
    def _endpoint(self) -> str:
        address = [self.target.host, self.target.user, self.target.port]
        return _digest_bytes(json.dumps(address).encode())[:24]

    def _reference(self, path: str, digest: str | None) -> _ArtifactMetadata:
        return _ArtifactMetadata(f"slurm-policy://{self._endpoint}{quote(path, safe='/')}", digest)

    def _workload(self, workload: WorkloadSpec, *, stage: bool = True,
                  transfers: dict[str, Any] | None = None) -> WorkloadSpec:
        if not isinstance(workload, EvaluationSpec):
            return workload

        from benchmax.execution.jobs.outputs import dependency
        from benchmax.execution.targets.registry import dependency_target

        if reference := dependency(workload):
            dependency_target(reference.target, self.target)
            return workload

        reference = workload.policy
        if not isinstance(reference, _ArtifactMetadata):
            return workload
        uri = urlsplit(reference.uri)

        if uri.scheme == "slurm-policy":
            if uri.netloc != self._endpoint:
                raise JobError("policy belongs to another target endpoint; transfer it explicitly")

            path = unquote(uri.path)
            if not path.startswith("/"):
                raise JobError("target policy path must be absolute")

            return replace(workload, policy=_ArtifactMetadata(path, reference.digest))

        if self.target.host is not None:
            from benchmax.execution.artifacts.policies import ArtifactError, validate_policy_artifact
            from benchmax.execution.targets.slurm.transfer import pack_policy

            limit = self.target.transfer_limit_bytes
            path, manifest, _ = validate_policy_artifact(reference, max_tensor_bytes=limit)
            if _ArtifactMetadata.from_path(path).digest != reference.digest:
                raise JobError("pass Reference.from_path(training_result) to transfer its canonical policy artifact")

            size = path.stat().st_size + (path.parent / manifest["tensor_file"]).stat().st_size
            if size > limit:
                raise ArtifactError("policy exceeds transfer_limit_mb")

            fields = {"digest": reference.digest, "manifest": path.name}
            cached = self._request("policy-cache", **fields)
            if transfers is not None:
                transfers.update(policy_bytes=0 if cached["cached"] else size, policy_cached=cached["cached"])

            if stage and not cached["cached"]:
                cached = self._request("upload", bundle=pack_policy(reference, limit=limit), **fields)

            staged = _ArtifactMetadata.from_document(cached["policy"])
            if staged.digest != reference.digest:
                raise JobError("staged policy digest differs from the requested artifact")

            return replace(workload, policy=staged)

        return workload

    def _command(self, *, inspection: bool = False) -> list[str]:
        command = [self.target.python, "-m", _BRIDGE]

        if inspection:
            # Match the Python import environment and thread limits in attempt.sh.
            command = [
                "env", "-u", "PYTHONHOME", "-u", "PYTHONPATH", "PYTHONNOUSERSITE=1", "PYTHONHASHSEED=0",
                f"OMP_NUM_THREADS={self.target.cpus}", f"OPENBLAS_NUM_THREADS={self.target.cpus}",
                f"MKL_NUM_THREADS={self.target.cpus}", *command
            ]

        if self.target.host is None:
            return command

        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
               "-o", "ClearAllForwardings=yes", "-p", str(self.target.port)]

        if self.target.control_path is not None:
            ssh.extend(["-S", self.target.control_path])

        if self.target.user is not None:
            ssh.extend(["-l", self.target.user])

        remote = shlex.join(command)
        if inspection:
            remote = f"cd {shlex.quote(self.target.workspace)} && exec {remote}"

        return [*ssh, self.target.host, remote]

    def _request(self, operation: str, **fields: Any) -> dict[str, Any]:
        request = {"schema_version": 1, "operation": operation, "target": self.target.to_document(), **fields}

        if operation in {"submit", "audit"}:
            from benchmax.execution.targets.slurm.bridge import package_fingerprint
            request["source_digest"] = package_fingerprint()

        command = self._command(inspection=True) if operation == "inspect" else self._command()
        options = {"cwd": self.target.workspace} if operation == "inspect" and self.target.host is None else {}
        try:
            process = subprocess.run(command, input=json.dumps(request, allow_nan=False), text=True,
                                     capture_output=True, timeout=self.target.command_timeout + 10, check=False, **options)
        except (OSError, subprocess.TimeoutExpired) as error:
            detail = ""

            if operation == "submit":
                detail = f"; submission outcome unknown, reconnect with Job.open({fields['id']!r}, target=...)"

            message = f"Slurm {operation} connection failed{detail}: {error}"

            if operation == "submit":
                raise JobSubmissionUnknownError(message, job_id=fields["id"]) from error

            raise JobError(message) from error

        try:
            response = json.loads(process.stdout)
        except ValueError as error:
            detail = f"; submission outcome unknown for {fields['id']}" if operation == "submit" else ""
            message = f"invalid Slurm response{detail}: {process.stderr[-2000:]}"

            if operation == "submit":
                raise JobSubmissionUnknownError(message, job_id=fields["id"]) from error

            raise JobError(message) from error

        if not isinstance(response, dict) or not response.get("ok") or process.returncode:
            error = response.get("error", process.stderr) if isinstance(response, dict) else "invalid response"

            if operation == "submit" and (
                not isinstance(response, dict) or response.get("ok") is not False
                or response.get("submission_unknown", False)
            ):
                raise JobSubmissionUnknownError(f"submission outcome unknown for {fields['id']}: {error}",
                                                job_id=fields["id"])

            raise JobError(f"Slurm {operation} failed: {error}")

        if not isinstance(response.get("value"), dict):
            if operation == "submit":
                message = f"submission outcome unknown for {fields['id']}: invalid response value"
                raise JobSubmissionUnknownError(message, job_id=fields["id"])

            raise JobError("Slurm response has no object value")

        return response["value"]

    def submit(self, workload: WorkloadSpec, *, output: str | os.PathLike[str] | None = None) -> SlurmJobHandle:
        prepared = self._workload(workload)
        job_id = f"{workload.kind}-{uuid.uuid4().hex[:16]}"
        snapshot = self._request("submit", id=job_id, workload=prepared.to_document(),
                                 output=os.fspath(output) if output is not None else None)

        try:
            return SlurmJobHandle(self, snapshot)
        except (KeyError, TypeError, ValueError) as error:
            raise JobSubmissionUnknownError(f"submission outcome unknown; reconnect to {job_id}: {error}",
                                            job_id=job_id) from error

    def open(self, job_id: str) -> SlurmJobHandle:
        return SlurmJobHandle(self, self._request("status", id=job_id))

    def audit(self, workload: WorkloadSpec) -> AuditReport:
        transfers = {}
        prepared = self._workload(workload, stage=False, transfers=transfers)
        document = self._request("audit", workload=prepared.to_document())
        if document.get("schema_version") != 1 or document.get("kind") != "audit":
            raise JobError("invalid Slurm audit document")

        audited = workload_from_document(document["workload"])
        evidence = dict(document["evidence"])
        if evidence.pop("workload_digest") != audited.digest or audited.digest != prepared.digest:
            raise JobError("Slurm audit does not match the requested workload")

        if transfers:
            evidence["estimates"]["transfers"].update(transfers)
            evidence["findings"].append("Local policy files are staged automatically on submission; audit uploads nothing.")

        for key in ("findings", "probes", "evidence"):
            evidence[key] = tuple(evidence[key])

        # Keep client policy locations in a saved audit so submitting it still stages the files.
        return AuditReport(workload=workload if transfers else audited, **evidence)


class SlurmJobHandle:
    def __init__(self, service: SlurmJobService, snapshot: dict[str, Any]) -> None:
        self._service = service
        self.id = snapshot["id"]
        self._snapshot = snapshot
        self._checked_at = time.monotonic()
        self.workload = workload_from_document(snapshot["workload"])
        self.output = Path(snapshot["output"])
        self._downloaded: _ArtifactMetadata | None = None
        self._refresh_lock = threading.Lock()

    def _refresh(self) -> dict[str, Any]:
        with self._refresh_lock:
            if (self._snapshot["state"] not in _TERMINAL_STATES
                    and time.monotonic() - self._checked_at >= self._service.target.poll_seconds):
                self._snapshot = self._service._request("status", id=self.id)
                self._checked_at = time.monotonic()

            return self._snapshot

    @property
    def state(self) -> str:
        return self._refresh()["state"]

    def progress(self) -> dict[str, Any]:
        return self._service._request("progress", id=self.id)

    def _ready(self) -> dict[str, Any] | None:
        value = self._refresh()
        if value["state"] in {"failed", "cancelled"}:
            raise JobFailedError(f"job {self.id}: {value.get('error') or value['state']}")

        if value["state"] == "submission_unknown":
            raise JobError(f"submission outcome unknown for {self.id}; inspect Slurm by this job name before retrying")

        if value["state"] != "succeeded":
            return None

        result = result_from_document(value["result"])
        _validate_result(result, workload=self.workload)

        if result.operation == "training":
            result = WorkloadResult.training(result.workload_digest, result.seed,
                                             self._service._reference(result.policy.uri, result.policy.digest))

        return result.to_document()

    def result(self, timeout: float | None = None) -> dict[str, Any]:
        deadline = _JobDeadline(self.id, timeout)

        while True:
            if (result := self._ready()) is not None:
                return result

            time.sleep(deadline.delay(self._service.target.poll_seconds))

    async def result_async(self, timeout: float | None = None) -> dict[str, Any]:
        deadline = _JobDeadline(self.id, timeout)

        while True:
            if (result := await asyncio.to_thread(self._ready)) is not None:
                return result

            await asyncio.sleep(deadline.delay(self._service.target.poll_seconds))

    def cancel(self) -> None:
        self._service._request("cancel", id=self.id)
        self._checked_at = float('-inf')

    def _download_policy(self, destination: Path | None = None) -> _ArtifactMetadata:
        from benchmax.execution.artifacts.policies import artifact_directory
        from benchmax.execution.targets.slurm.transfer import unpack_policy

        result = result_from_document(self.result())
        if result.operation != "training":
            raise JobError("only training jobs produce a policy")

        if destination is None and self._downloaded is not None:
            return self._downloaded

        bundle = self._service._request("download", id=self.id)
        path = destination if destination is not None else artifact_directory()
        reference = unpack_policy(bundle, path, digest=result.policy.digest,
                                  limit=self._service.target.transfer_limit_bytes)
        self._downloaded = reference
        return reference

    def save_policy(self, result: WorkloadResult, destination: Path) -> _ArtifactMetadata:
        reference = self._download_policy(destination)
        if reference.digest != result.policy.digest:
            raise JobError("downloaded policy differs from the completed result")

        return reference

    def load_policy(self, result: WorkloadResult, *, expected_game=None):
        from benchmax.execution.artifacts.policies import load_policy

        reference = self._download_policy()
        if reference.digest != result.policy.digest:
            raise JobError("downloaded policy differs from the completed result")

        return load_policy(reference, expected_game=expected_game or self.workload.game)
