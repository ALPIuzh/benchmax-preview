"""Policy inputs validated at entry and resolved at execution boundaries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from benchmax.execution.artifacts.policies import ArtifactError, artifact_directory, load_policy, write_policy
from benchmax.execution.jobs.job import Job
from benchmax.execution.jobs.reference import Reference
from benchmax.execution.specs import EntrySpec, SpecError, _ArtifactMetadata
from benchmax.execution.targets.service import JobError
from benchmax.policy import BatchedCallable, PolicyProfile, _as_policy_profile, _attach_policy_metadata

if TYPE_CHECKING:
    from benchmax.core import Game
    from benchmax.evaluation import Evaluator

PolicyInput: TypeAlias = PolicyProfile | BatchedCallable | Reference[PolicyProfile] | str | os.PathLike[str]


def validate_evaluation_input(policy: PolicyInput, *, evaluator: Evaluator | str | None, portable: bool) -> str | None:
    """Validate shared evaluation inputs and return a portable evaluator name."""

    evaluator_name = evaluator if isinstance(evaluator, str) else None
    if evaluator_name == "":
        raise ValueError("evaluator name must be non-empty")

    if evaluator is not None and not isinstance(evaluator, str) and portable:
        raise ValueError(
            "an evaluator passed directly can only be used for synchronous "
            "evaluation without a durable output; pass its registered name for portable evaluation"
        )

    validate_policy_input(policy)
    return evaluator_name


def validate_policy_input(policy: PolicyInput) -> None:
    """Check accepted inputs without loading, pinning, or waiting."""

    if isinstance(policy, Job):
        raise TypeError("pass job.reference() explicitly")

    if isinstance(policy, Reference):
        if policy.output != "policy":
            raise TypeError("evaluation requires a policy output")
    elif not isinstance(policy, (PolicyProfile, str, os.PathLike)) and not callable(policy):
        raise TypeError("policy must be a PolicyProfile, callable, policy Reference, or local artifact path")


def resolve_policy(policy: PolicyInput, *, game: Game) -> PolicyProfile:
    """Load a validated input for synchronous evaluation."""

    if isinstance(policy, Reference):
        from benchmax.execution.jobs.submission import game_entry

        return policy._load_policy(expected_game=game_entry(game))

    if isinstance(policy, (str, os.PathLike)):
        from benchmax.execution.jobs.submission import game_entry

        return load_policy(os.fspath(policy), expected_game=game_entry(game))

    return _as_policy_profile(policy)


def pin_policy(
    policy: PolicyInput, *, output: Path | None = None, game: EntrySpec | None = None
) -> _ArtifactMetadata | Reference[PolicyProfile]:
    """Pin an artifact path or portable in-memory policy for a workload."""

    if isinstance(policy, Reference):
        return policy._artifact_metadata() if policy.kind == "file" else policy

    if isinstance(policy, (str, os.PathLike)):
        from benchmax.execution.artifacts.policies import _read_document, resolve_policy_artifact

        reference = _ArtifactMetadata.from_path(policy)
        _, document = _read_document(reference, relative_to=None)
        if document.get("kind") == "training-result":
            manifest, _, _ = resolve_policy_artifact(reference)
            return _ArtifactMetadata.from_path(manifest)

        return reference

    policy = _as_policy_profile(policy)

    with policy._portable_lock:
        reference = policy._portable_reference

        if isinstance(reference, _ArtifactMetadata) and reference.digest is not None:
            try:
                current = _ArtifactMetadata.from_path(reference.uri)
            except SpecError:
                current = None

            if current is not None and current.digest == reference.digest:
                return reference

        source = policy._portable_training
        if source is None:
            from benchmax.execution.artifacts.policies import _PolicySource

            entry = policy._portable_game or game
            if entry is None:
                raise JobError("policy has no game provenance; provide a policy manifest")

            source = _PolicySource(entry)

        try:
            reference = write_policy(policy, source, artifact_directory(output))
        except ArtifactError as error:
            raise JobError(f"{error}; provide a policy manifest or register a compatible codec") from error

        _attach_policy_metadata(policy, reference=reference, game=source.game)
        return reference
