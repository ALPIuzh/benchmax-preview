"""Execute training and annotate its resulting policy."""

from __future__ import annotations

from benchmax.core import Game
from benchmax.policy import PolicyProfile
from benchmax.training.algorithm import TrainingFunction


def execute_training(game: Game, training_function: TrainingFunction, *, seed: int) -> PolicyProfile:
    """Execute the same in-memory training operation for every transport."""

    from benchmax.release import capture_producer

    producer = capture_producer()
    profile = training_function(game, seed=seed)
    if not isinstance(profile, PolicyProfile):
        raise TypeError(f"training function must return a PolicyProfile; got {type(profile).__name__}")

    profile._artifact_annotations = {"producer": producer}
    return profile
