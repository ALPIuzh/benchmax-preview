"""Prepare portable workloads and jobs from public games, algorithms, and policies."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmax.config import _EntryOrigin
from benchmax.execution.jobs.job import Job
from benchmax.execution.jobs.policy_inputs import PolicyInput, pin_policy, validate_evaluation_input
from benchmax.execution.specs import EntrySpec, EvaluationSpec, TrainingSpec
from benchmax.execution.targets.service import JobError

if TYPE_CHECKING:
    from benchmax.core import Game
    from benchmax.evaluation import Evaluator
    from benchmax.training.algorithm import Algorithm


def game_entry(game: Game, *, evaluator_sensitive: bool = False, evaluator: str | None = None) -> EntrySpec:
    """Return the portable registration behind one loaded game."""

    origin = getattr(game, "_benchmax_origin", None)
    if not isinstance(origin, _EntryOrigin) or origin.resolved.group != "game":
        raise JobError("portable workloads require a game created by benchmax.load_game()")

    if (evaluator_sensitive and evaluator is None
            and getattr(game, "_benchmax_configured_evaluator", None) is not game.evaluator):
        raise JobError("load the game with evaluator=... or evaluate synchronously without a durable output")

    overrides = origin.overrides
    resolved = origin.resolved.configuration

    if evaluator is not None:
        from benchmax.registry import _EVALUATORS

        overrides["evaluator"] = evaluator
        resolved["evaluator"] = _EVALUATORS.resolve(evaluator).configuration

    return EntrySpec(origin.resolved.name, overrides, resolved)


def training_spec(game: Game, algorithm: Algorithm, *, seed: int) -> TrainingSpec:
    """Describe a loaded algorithm without losing its configuration snapshot."""
    origin = algorithm._origin
    return TrainingSpec.create(
        game=game_entry(game), algorithm=EntrySpec(algorithm.name, origin.overrides, origin.resolved.configuration),
        total_frames=algorithm.total_frames, seed=seed, device=algorithm.device
    )


def portable_training_spec(game: Game, algorithm: Algorithm, *, seed: int) -> TrainingSpec | None:
    """Describe training when its game has a portable registration."""
    try:
        return training_spec(game, algorithm, seed=seed)
    except JobError:
        return None


def evaluation_job(
    game: Game, policy: PolicyInput, *, output: str | os.PathLike[str] | None, evaluator: Evaluator | str | None = None,
) -> Job[dict[str, Any]]:
    """Build an unstarted evaluation job from a safe policy reference."""

    evaluator_name = validate_evaluation_input(policy, evaluator=evaluator, portable=True)
    entry = game_entry(game, evaluator_sensitive=True, evaluator=evaluator_name)
    destination = Path(output).expanduser().resolve() if output is not None else None
    if isinstance(policy, (str, os.PathLike)):
        policy = Path(policy).expanduser().resolve()

    def prepare() -> EvaluationSpec:
        return EvaluationSpec(game=entry, policy=pin_policy(policy, output=destination, game=entry))

    return Job._create(prepare, output=output)
