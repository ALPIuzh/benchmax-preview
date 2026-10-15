import numpy as np
import pytest

from benchmax import Evaluator, Game, GameInstance, PolicyProfile


class DummyEvaluator:
    def evaluate(self, game: "DummyGame", policy_profile: PolicyProfile) -> dict[str, object]:
        mask = np.ones((1, 2), dtype=bool)
        probabilities = policy_profile(0, np.zeros((1, 1)), (0, 1), mask)
        assert game.name == "dummy"
        return {"probability_of_action_1": float(probabilities[0, 1])}


_DUMMY_EVALUATOR = DummyEvaluator()


class DummyGame(Game):
    def __init__(
        self, evaluator: Evaluator["DummyGame"] | None = _DUMMY_EVALUATOR, discount: float | None = None
    ) -> None:
        super().__init__(
            name="dummy", players=(0, 1), observation_spaces={0: (1,), 1: (1,)}, action_spaces={0: (0, 1), 1: (0, 1)},
            is_fully_observable=True, is_simultaneous=False, evaluator=evaluator, discount=discount
        )

    def new_env(self) -> GameInstance:
        raise NotImplementedError


def test_abstract_interfaces_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Game(  # type: ignore[abstract]
            name="abstract", players=(), observation_spaces={}, action_spaces={},
            is_fully_observable=True, is_simultaneous=False,
        )

    with pytest.raises(TypeError):
        GameInstance(DummyGame())  # type: ignore[abstract]


@pytest.mark.parametrize("discount", (-0.1, 1.0))
def test_discount_must_lie_in_the_unit_interval(discount: float) -> None:
    with pytest.raises(ValueError, match=r"discount must lie in \[0, 1\)"):
        DummyGame(discount=discount)


def test_evaluate_accepts_a_bare_batched_callable() -> None:
    def uniform_policy_profile(
        player: int, observations: np.ndarray, action_space: tuple[int, ...], legal_action_mask: np.ndarray
    ) -> np.ndarray:
        del player, observations, action_space
        probabilities = legal_action_mask.astype(float)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    assert DummyGame().evaluate(uniform_policy_profile) == {"probability_of_action_1": 0.5}
    assert DummyGame().evaluate(PolicyProfile(uniform_policy_profile)) == {"probability_of_action_1": 0.5}


def test_evaluate_normalizes_before_delegating_to_the_game() -> None:
    received: list[object] = []

    def uniform_policy_profile(
        player: int, observations: np.ndarray, action_space: tuple[int, ...], legal_action_mask: np.ndarray
    ) -> np.ndarray:
        del player, observations, action_space
        probabilities = legal_action_mask.astype(float)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    class RecordingEvaluator:
        def evaluate(self, game: DummyGame, policy_profile: PolicyProfile) -> dict[str, object]:
            assert isinstance(game, DummyGame)
            received.append(policy_profile)
            return {}

    game = DummyGame(evaluator=RecordingEvaluator())
    game.evaluate(uniform_policy_profile)
    game.evaluate(PolicyProfile(uniform_policy_profile))

    assert len(received) == 2
    assert all(isinstance(profile, PolicyProfile) for profile in received)


def test_a_game_needs_only_dynamics_and_may_omit_an_evaluator() -> None:
    class GameWithoutEvaluate(Game):
        def __init__(self) -> None:
            super().__init__(
                name="incomplete", players=(0,), observation_spaces={0: (1,)}, action_spaces={0: (0, 1)},
                is_fully_observable=True, is_simultaneous=False,
            )

        def new_env(self) -> GameInstance:
            raise NotImplementedError

    game = GameWithoutEvaluate()

    with pytest.raises(NotImplementedError, match="no evaluator"):
        game.evaluate(lambda p, o, a, m: np.ones_like(m, dtype=float) / m.shape[1])
