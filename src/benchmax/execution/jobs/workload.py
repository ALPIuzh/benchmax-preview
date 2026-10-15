"""Execute workloads and capture their durable results."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmax.execution.artifacts.policies import artifact_directory, load_policy, resolve_policy_artifact, write_policy
from benchmax.execution.artifacts.results import WorkloadResult
from benchmax.execution.specs import (
    EvaluationSpec,
    TrainingSpec,
    WorkloadSpec,
    validate_workload_resolution,
)
from benchmax.execution.targets.service import _now
from benchmax.registry import load_game
from benchmax.release import capture_producer


def evaluate_recorded(workload: EvaluationSpec, evaluate: Callable[[], dict[str, Any]]) -> WorkloadResult:
    """Retain requested settings separately from recorded execution evidence."""
    _, manifest, _ = resolve_policy_artifact(workload.policy)
    provenance = {
        "producer": capture_producer(),
        "policy": {key: manifest.get(key) for key in ("training", "producer", "progress")},
        "started_at": _now(),
    }
    started = perf_counter()
    metrics = evaluate()
    provenance.update(finished_at=_now(), duration_seconds=perf_counter() - started)
    return WorkloadResult.evaluation(workload.digest, metrics, workload=workload, provenance=provenance)


def _execute_training(spec: TrainingSpec, output: Path) -> WorkloadResult:
    from benchmax.training.algorithm import load_algorithm
    from benchmax.training.execution import execute_training

    game = load_game(spec.game.name, **spec.game.overrides)
    training_function = load_algorithm(
        spec.algorithm.name, total_frames=spec.total_frames, device=spec.device, verbose=False,
        **spec.algorithm.overrides
    )
    profile = execute_training(game, training_function, seed=spec.seed)
    policy = write_policy(profile, spec, artifact_directory(output))
    return WorkloadResult.training(spec.digest, spec.seed, policy)


def _execute_evaluation(spec: EvaluationSpec) -> WorkloadResult:
    game = load_game(spec.game.name, **spec.game.overrides)
    policy = load_policy(spec.policy, expected_game=spec.game)
    return evaluate_recorded(spec, lambda: game.evaluate(policy))


def execute_workload(workload: WorkloadSpec, output: Path) -> WorkloadResult:
    """Run one allocated workload and return its durable result document."""

    output.parent.mkdir(parents=True, exist_ok=True)
    validate_workload_resolution(workload)

    if isinstance(workload, TrainingSpec):
        return _execute_training(workload, output)

    return _execute_evaluation(workload)
