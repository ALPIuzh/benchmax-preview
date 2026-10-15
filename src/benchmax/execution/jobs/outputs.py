"""Shared output loading, saving, and dependency resolution."""

from __future__ import annotations

import math
import os
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmax.execution.artifacts import policies
from benchmax.execution.artifacts.results import WorkloadResult, result_from_document
from benchmax.execution.specs import EvaluationSpec, TrainingSpec, WorkloadSpec, _ArtifactMetadata, _ReferenceSpec
from benchmax.execution.targets.service import JobError, _validate_result
from benchmax.utils.json import write_document

if TYPE_CHECKING:
    from benchmax.execution.targets.service import SubmittedJob
    from benchmax.policy import PolicyProfile


def dependency(workload: WorkloadSpec) -> _ReferenceSpec | None:
    if (isinstance(workload, EvaluationSpec) and isinstance(workload.policy, _ReferenceSpec)
            and workload.policy.kind == "job-output"):
        return workload.policy

    return None


def validate_producer(reference: _ReferenceSpec, producer: WorkloadSpec) -> None:
    if producer.digest != reference.workload_digest:
        raise JobError("referenced producer workload digest mismatch")

    if reference.output != "policy" or not isinstance(producer, TrainingSpec):
        raise JobError("evaluation requires the policy output of a training job")


def execute_with_references(
    workload: WorkloadSpec, output: Path, *, run: Path, executor: Callable[[WorkloadSpec, Path], WorkloadResult],
    resolve_dependency: Callable[[_ReferenceSpec], tuple[_ArtifactMetadata, dict[str, Any]]],
) -> WorkloadResult:
    """Resolve on the executing host and retain both submitted and pinned specs."""
    if not isinstance(workload, EvaluationSpec) or not isinstance(workload.policy, _ReferenceSpec):
        return executor(workload, output)

    reference = workload.policy
    policy_job = None
    if reference.kind != "job-output":
        from benchmax.execution.jobs.reference import Reference

        saved = Reference.from_document(reference.to_document()).save(run / "policy-input")
        artifact = saved._artifact_metadata()
    else:
        artifact, state = resolve_dependency(reference)
        policy_job = {"job_id": reference.job_id,
                      **{key: state.get(key) for key in ("state", "started_at", "finished_at", "progress")}}

    resolved = replace(workload, policy=artifact)
    # Keep the submitted future reference for inspection, and the exact artifact
    # consumed by this attempt for reproducibility.
    write_document(resolved, run / "resolved-spec.json", replace=False)
    product = executor(resolved, output)

    _validate_result(product, workload=resolved)
    provenance = product.provenance
    if policy_job is not None:
        provenance = {**(provenance or {}), "policy_job": policy_job}

    return WorkloadResult.evaluation(workload.digest, product.metrics, workload=workload,
                                     resolved_workload=resolved, provenance=provenance)


def _result_path(path: str | os.PathLike[str] | None) -> Path | None:
    if path is None:
        return None

    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"result path already exists; choose a new local directory: {destination}")

    return destination


def _save_path(path: str | os.PathLike[str] | None) -> Path:
    destination = _result_path(path)
    if destination is None:
        raise TypeError("save requires a local destination directory")

    return destination


def _file_timeout(timeout: float | None) -> None:
    if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
        raise ValueError("timeout must be finite and non-negative, or None")


def pin_file(path: str | os.PathLike[str]) -> _ReferenceSpec:
    """Validate an existing document and pin its file source, without model loading."""
    metadata = _ArtifactMetadata.from_path(path)
    _, document = policies._read_document(metadata, relative_to=None)

    if document.get("kind") == "evaluation-result":
        result_from_document(document)
        return _ReferenceSpec._from_file(metadata, "metrics")

    if document.get("kind") not in {"policy-artifact", "training-result"}:
        raise policies.ArtifactError("expected a policy manifest, training result, or evaluation result")

    manifest, _, _ = policies.resolve_policy_artifact(metadata)
    # Pin the manifest itself so a saved reference survives removal of the job store.
    pinned = metadata if manifest == Path(metadata.uri) else _ArtifactMetadata.from_path(manifest)
    policies.resolve_policy_artifact(pinned)
    return _ReferenceSpec._from_file(pinned, "policy")


def file_result(reference: _ReferenceSpec, *, timeout: float | None = None) -> Any:
    """Resolve a file directly, without constructing or opening a Job."""
    _file_timeout(timeout)
    metadata = reference._artifact_metadata()

    if reference.output == "policy":
        _, _, training = policies.resolve_policy_artifact(metadata)
        return policies.load_policy(metadata, expected_game=training.game)

    _, document = policies._read_document(metadata, relative_to=None)
    result = result_from_document(document)
    if result.operation != "evaluation":
        raise policies.ArtifactError("metrics reference does not contain an evaluation result")

    return result.metrics


def save_file(reference: _ReferenceSpec, *, path: str | os.PathLike[str],
              timeout: float | None = None) -> _ReferenceSpec:
    """Copy a verified file source without a job lookup or model instantiation."""
    _file_timeout(timeout)
    destination = _save_path(path)
    metadata = reference._artifact_metadata()

    if reference.output == "policy":
        copied = policies._copy_policy_files(metadata, destination)
    else:
        source, document = policies._read_document(metadata, relative_to=None)
        if result_from_document(document).operation != "evaluation":
            raise policies.ArtifactError("metrics reference does not contain an evaluation result")

        destination.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source, destination / "result.json")
        copied = _ArtifactMetadata.from_path(destination / "result.json")
        if copied.digest != metadata.digest:
            raise policies.ArtifactError("copied result differs from the referenced file")

    return _ReferenceSpec._from_file(copied, reference.output)


def save_result(result: WorkloadResult, workload: WorkloadSpec, path: Path,
                *, handle: SubmittedJob | None) -> _ReferenceSpec:
    """Save a verified result using its target's file-transfer support when available."""
    _validate_result(result, workload=workload)

    if result.operation == "training":
        saver = getattr(handle, "save_policy", None)
        metadata = saver(result, path) if saver is not None else policies._copy_policy_files(result.policy, path)
        return _ReferenceSpec._from_file(metadata, "policy")

    if result.workload is None:
        # Legacy results have no execution evidence; never substitute the exporting host's environment.
        result = WorkloadResult.evaluation(result.workload_digest, result.metrics, workload=workload)

    path.mkdir(parents=True, exist_ok=False)
    write_document(result, path / "result.json", replace=False)
    return _ReferenceSpec._from_file(_ArtifactMetadata.from_path(path / "result.json"), "metrics")


def load_result(result: WorkloadResult, workload: WorkloadSpec, *, handle: SubmittedJob | None,
                path: Path | None = None) -> PolicyProfile | dict[str, Any]:
    """Load a result, optionally retaining its files in a new local directory."""
    _validate_result(result, workload=workload)

    if path is not None:
        saved = save_result(result, workload, path, handle=handle)

        if result.operation == "training":
            return policies.load_policy(saved._artifact_metadata(), expected_game=workload.game)

        return file_result(saved)

    if result.operation == "training":
        loader = getattr(handle, "load_policy", None)
        return loader(result) if loader is not None else policies.load_policy(result.policy, expected_game=workload.game)

    return result.metrics
