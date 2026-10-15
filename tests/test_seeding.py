from __future__ import annotations

import concurrent.futures

import numpy as np
import pytest

from benchmax.seeding import (
    SEED_SCHEME,
    EntropyDisabledError,
    derive_seed,
    forbid_entropy,
    numpy_generator,
    python_generator,
    resolve_seed,
)


def _stochastic_tabular_game():
    from benchmax.games.tabular import TabularMarkovGame

    transitions = np.full((2, 1, 2), 0.5)
    return TabularMarkovGame(
        name="coin", transitions=transitions, rewards={0: np.zeros((2, 1))},
        initial_state_distribution=np.array([0.5, 0.5]), discount=0.9
    )


def test_explicit_seeds_are_allowed_when_entropy_is_forbidden() -> None:
    with forbid_entropy():
        assert resolve_seed(7, purpose="test") == 7


def test_entropy_can_be_forbidden_across_the_process() -> None:
    with forbid_entropy(), concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        request = pool.submit(resolve_seed, None, purpose="background test")

        with pytest.raises(EntropyDisabledError, match="background test requested fresh entropy"):
            request.result()


def test_a_training_workload_cannot_resolve_an_omitted_seed_in_strict_mode() -> None:
    from benchmax.execution.specs import TrainingSpec

    with forbid_entropy(), pytest.raises(EntropyDisabledError, match="training workload requested fresh entropy"):
        TrainingSpec.create(game="rock_paper_scissors")

    with forbid_entropy():
        assert TrainingSpec.create(game="rock_paper_scissors", seed=7).seed == 7


def test_named_seeds_do_not_depend_on_the_order_in_which_they_are_requested() -> None:
    expected = {name: derive_seed(7, "training", name) for name in ("experiment", "collector", "evaluation")}
    reordered = {name: derive_seed(7, "training", name) for name in ("evaluation", "experiment", "collector")}

    assert expected == reordered
    assert len(set(expected.values())) == len(expected)


def test_seed_derivation_is_a_versioned_contract() -> None:
    assert SEED_SCHEME == "benchmax-sha256-v1"
    assert derive_seed(7, "training", "collector") == 1_198_237_238


def test_generators_have_reproducible_pinned_streams() -> None:
    np.testing.assert_array_equal(numpy_generator(7).random(10), numpy_generator(7).random(10))
    first_python, second_python = python_generator(7), python_generator(7)
    assert [first_python.random() for _ in range(10)] == [second_python.random() for _ in range(10)]


def test_scalar_tabular_environments_reproduce_the_same_sequence() -> None:
    game = _stochastic_tabular_game()
    first = game.new_env()
    second = game.new_env()
    actions = {0: 0}

    with forbid_entropy():
        first_step = first.reset(seed=7)
        second_step = second.reset(seed=7)

        for _ in range(20):
            np.testing.assert_array_equal(first_step.observations[0], second_step.observations[0])
            first_step = first.step(actions)
            second_step = second.step(actions)

    np.testing.assert_array_equal(first_step.observations[0], second_step.observations[0])


def test_a_tabular_environment_rejects_an_unseeded_first_reset() -> None:
    with pytest.raises(RuntimeError, match="first reset requires an explicit seed"):
        _stochastic_tabular_game().new_env().reset()
