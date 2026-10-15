"""Public synchronous/deferred training adapter."""

from __future__ import annotations

import os
import warnings
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Literal, overload

from benchmax.config import ConfigValue
from benchmax.core import Game
from benchmax.execution.jobs.job import Job
from benchmax.execution.jobs.submission import game_entry, portable_training_spec, training_spec
from benchmax.execution.operation import executable
from benchmax.execution.specs import TrainingSpec
from benchmax.policy import PolicyProfile, _attach_policy_metadata
from benchmax.seeding import resolve_seed, validate_seed
from benchmax.training.algorithm import Algorithm, TrainingFunction, _validate_training_controls, load_algorithm
from benchmax.training.execution import execute_training

if TYPE_CHECKING:
    from benchmax.execution.targets.base import Target


def _prepare_training(
    game: object, *, algorithm: object, total_frames: object, seed: int | None, device: object, verbose: object
) -> int:
    """Validate shared training inputs and choose the seed before executing or deferring."""

    if not isinstance(game, Game):
        raise TypeError("game must be a BenchMAX Game")

    if isinstance(algorithm, str):
        if not algorithm:
            raise TypeError("algorithm must be a non-empty string")
    elif not callable(algorithm):
        raise TypeError("algorithm must be a registered name or training function")

    _validate_training_controls(total_frames=total_frames, device=device, verbose=verbose)
    if seed is not None:
        validate_seed(seed)

    if game.evaluator is None:
        warnings.warn(
            f"{game.name!r}: no evaluator is configured; "
            "training will proceed, but evaluation requires a compatible evaluator "
            "to be configured or passed to game.evaluate()",
            stacklevel=4,
        )

    return resolve_seed(seed, purpose="training")


def training_job(
    game: Game, algorithm: str | TrainingFunction, *, total_frames: int | None, seed: int | None, device: str,
    verbose: bool, output: str | os.PathLike[str] | None, **overrides: ConfigValue
) -> Job[PolicyProfile]:
    """Capture a portable training request without importing the training provider."""

    seed = _prepare_training(
        game, algorithm=algorithm, total_frames=total_frames, seed=seed, device=device, verbose=verbose
    )
    if isinstance(algorithm, Algorithm):
        _require_bound_settings(total_frames, device, verbose, overrides)
        prepare = training_spec(game, algorithm, seed=seed)
    elif isinstance(algorithm, str):
        prepare = partial(TrainingSpec.create, game=game_entry(game), algorithm=algorithm, total_frames=total_frames,
                          seed=seed, device=device, algorithm_overrides=deepcopy(overrides))
    else:
        raise ValueError(  # noqa: TRY004 -- callables are valid for synchronous training
            "submitted training requires a registered algorithm; custom training functions run directly"
        )

    return Job._create(prepare, output=output)


@overload
def train(
    game: Game, algorithm: str | TrainingFunction = "mappo", *, total_frames: int | None = None,
    seed: int | None = None, device: str = "auto", verbose: bool = True, submit: Literal[False] = False,
    target: None = None, output: str | os.PathLike[str] | None = None, **overrides: ConfigValue
) -> PolicyProfile: ...


@overload
def train(
    game: Game, algorithm: str | Algorithm = "mappo", *, total_frames: int | None = None, seed: int | None = None,
    device: str = "auto", verbose: bool = True, submit: Literal[True], target: str | Target | None = None,
    output: str | os.PathLike[str] | None = None, **overrides: ConfigValue
) -> Job[PolicyProfile]: ...


@overload
def train(
    game: Game, algorithm: str | TrainingFunction = "mappo", *, total_frames: int | None = None,
    seed: int | None = None, device: str = "auto", verbose: bool = True, submit: bool,
    target: str | Target | None = None, output: str | os.PathLike[str] | None = None, **overrides: ConfigValue
) -> PolicyProfile | Job[PolicyProfile]: ...


@executable(job=training_job)
def train(
    game: Game, algorithm: str | TrainingFunction = "mappo", *, total_frames: int | None = None,
    seed: int | None = None, device: str = "auto", verbose: bool = True,
    output: str | os.PathLike[str] | None = None, **overrides: ConfigValue
) -> PolicyProfile:
    """Train directly, or return a job with explicit submission controls.

    ``submit=False`` trains synchronously and returns a policy. ``submit=True``
    captures the operation in a :class:`benchmax.execution.Job`. An explicit
    target submits it immediately, otherwise its ``run()`` method selects the
    target later.

    Args:
        game: Registered or custom BenchMAX game.
        algorithm: Registered name, a recipe from ``load_algorithm()``, or a
            custom training function for direct in-process execution.
        total_frames: Positive environment-frame budget.
        seed: Reproducible seed, generated once for a job when omitted.
        device: ``auto`` or an explicit training device.
        verbose: Whether synchronous training writes framework output.
        submit: True creates a job, and False runs synchronously.
        target: Execution target to submit to immediately in job mode.
        output: Optional durable training result. In job mode this is the
            target's primary output path.
        overrides: Registered algorithm configuration overrides.

    Returns:
        A trained policy, or a ``Job[PolicyProfile]``.
    """

    training_seed = _prepare_training(
        game, algorithm=algorithm, total_frames=total_frames, seed=seed, device=device, verbose=verbose
    )

    if isinstance(algorithm, str):
        algorithm = load_algorithm(algorithm, total_frames=total_frames, device=device, verbose=verbose, **overrides)
    else:
        _require_bound_settings(total_frames, device, verbose, overrides)

    if not isinstance(algorithm, Algorithm):
        if output is not None:
            raise ValueError(
                "durable output requires a registered algorithm; custom "
                "training functions return their policy directly"
            )

        return execute_training(game, algorithm, seed=training_seed)

    from benchmax.execution.jobs.synchronous import output_path, persist_training_result

    synchronous_spec = None
    if output is not None:
        output_path(output)
        synchronous_spec = training_spec(game, algorithm, seed=training_seed)

    try:
        profile = execute_training(game, algorithm, seed=training_seed)
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"direct training requires the selected training provider: {error}. "
            "Alternatively, construct a job with submit=True and run it on a compatible target"
        ) from error

    training = synchronous_spec or portable_training_spec(game, algorithm, seed=training_seed)
    if training is not None:
        _attach_policy_metadata(profile, training=training)
        if output is not None:
            persist_training_result(profile, training, output)

    return profile


def _require_bound_settings(
    total_frames: int | None, device: str, verbose: bool, overrides: dict[str, ConfigValue]
) -> None:
    if total_frames is not None or device != "auto" or not verbose or overrides:
        raise ValueError("configure a custom training function directly or bind settings with load_algorithm()")
