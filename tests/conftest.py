"""Only fixtures genuinely shared across subsystems belong here."""

import pytest

from tests.training_helpers import tiny_training_params


@pytest.fixture(scope="session")
def tiny_trained_policy():
    pytest.importorskip("benchmarl")
    from benchmax import load_game
    from benchmax.execution.specs import TrainingSpec
    from benchmax.training import train

    game = load_game("rock_paper_scissors")
    params = tiny_training_params()
    policy = train(game, total_frames=8, seed=7, device="cpu", verbose=False, params=params)
    spec = TrainingSpec.create(
        game="rock_paper_scissors", seed=7, total_frames=8, algorithm_overrides={"params": params}
    )
    return game, policy, spec
