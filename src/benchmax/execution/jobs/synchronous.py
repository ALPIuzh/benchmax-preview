"""Durable outputs for synchronous training and evaluation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmax.execution.artifacts.policies import artifact_directory, write_policy
from benchmax.execution.artifacts.results import WorkloadResult
from benchmax.execution.jobs.policy_inputs import pin_policy
from benchmax.execution.jobs.submission import game_entry
from benchmax.execution.jobs.workload import evaluate_recorded
from benchmax.execution.specs import EvaluationSpec, TrainingSpec
from benchmax.execution.targets.service import JobError
from benchmax.policy import PolicyProfile, _attach_policy_metadata
from benchmax.utils.json import write_document

if TYPE_CHECKING:
    from benchmax.core import Game
    from benchmax.evaluation import Evaluator


def output_path(output: str | os.PathLike[str]) -> Path:
    """Resolve a new synchronous output path owned by the caller."""

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise JobError(f"output destination already exists: {destination}")

    return destination


def persist_training_result(profile: PolicyProfile, workload: TrainingSpec, output: str | os.PathLike[str]) -> None:
    """Publish the durable counterpart of a synchronous training result."""

    destination = output_path(output)
    policy_directory = artifact_directory(destination)
    reference = write_policy(profile, workload, policy_directory)
    result = WorkloadResult.training(workload.digest, workload.seed, reference)

    try:
        write_document(result, destination, replace=False)
    except BaseException:
        shutil.rmtree(policy_directory, ignore_errors=True)
        raise

    _attach_policy_metadata(profile, reference=reference)


def evaluate_and_save(
    game: Game, profile: PolicyProfile, output: str | os.PathLike[str], *, evaluator: Evaluator,
    evaluator_name: str | None = None
) -> dict[str, Any]:
    """Pin the input, evaluate it, and publish a result with execution provenance."""

    destination = output_path(output)
    workload = EvaluationSpec(
        game=game_entry(game, evaluator_sensitive=True, evaluator=evaluator_name),
        policy=pin_policy(profile, output=destination, game=game_entry(game)),
    )
    result = evaluate_recorded(workload, lambda: evaluator.evaluate(game, profile))
    write_document(result, destination, replace=False)
    return result.metrics
