from collections.abc import Sequence

import numpy as np
import pytest

from benchmax import InvalidPolicyError, PolicyProfile
from benchmax.policy import _as_policy_profile


def _uniform_batched(
    player: int, observations: np.ndarray, action_space: tuple[int, ...], legal_action_mask: np.ndarray
) -> np.ndarray:
    del player, observations, action_space
    probabilities = legal_action_mask.astype(float)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def test_batched_callable_is_called_and_its_output_validated() -> None:
    calls: list[tuple[int, tuple[int, ...]]] = []
    output = np.array([[1, 0]])

    def batched(
        player: int, observations: np.ndarray, action_space: tuple[int, ...], legal_action_mask: np.ndarray
    ) -> np.ndarray:
        calls.append((player, action_space))
        return output

    profile = PolicyProfile(batched)
    mask = np.array([[True, False]])
    probabilities = profile(5, np.zeros((1, 1)), (10, 20), mask)

    assert calls == [(5, (10, 20))]
    np.testing.assert_array_equal(probabilities, output)
    assert probabilities.dtype == np.dtype(float)

    output[0] = [0, 1]

    with pytest.raises(InvalidPolicyError, match="player 5.*illegal action 20"):
        profile(5, np.zeros((1, 1)), (10, 20), mask)


def test_constructing_a_policy_profile_requires_a_callable() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        PolicyProfile(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be callable"):
        PolicyProfile.from_scalar_callable(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("fallback", ["error", "uniform"])
def test_table_lookup_preserves_large_integer_action_ids(fallback) -> None:
    actions = (2**63 + 1, -1, 2**63)
    policy = PolicyProfile.from_table({(0, (0.0,), actions): {actions[0]: 1.0, -1: 0.0, actions[2]: 0.0}},
                                      fallback=fallback)

    probabilities = policy(0, np.zeros((1, 1)), actions, np.ones((1, 3), dtype=bool))

    np.testing.assert_array_equal(probabilities, [[1.0, 0.0, 0.0]])


def test_as_policy_profile_wraps_callables_and_passes_profiles_through() -> None:
    profile = PolicyProfile(_uniform_batched)
    assert _as_policy_profile(profile) is profile

    wrapped = _as_policy_profile(_uniform_batched)
    assert isinstance(wrapped, PolicyProfile)
    np.testing.assert_array_equal(
        wrapped(0, np.zeros((1, 1)), (0, 1), np.ones((1, 2), dtype=bool)), np.array([[0.5, 0.5]])
    )


def test_scalar_adapter_flattens_observations_and_maps_legal_probabilities() -> None:
    calls: list[tuple[int, tuple[float, ...], tuple[int, ...]]] = []

    def scalar_policy(player: int, observation: tuple[float, ...], legal_actions: Sequence[int]) -> dict[int, float]:
        legal = tuple(legal_actions)
        calls.append((player, observation, legal))

        if legal == (10, 90):
            return {10: 0.25, 90: 0.75}

        assert legal == (30, 120)
        return {30: 0.6, 120: 0.4}

    policy = PolicyProfile.from_scalar_callable(scalar_policy)
    observations = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    action_space = (10, 30, 90, 120)
    legal_action_mask = np.array([[True, False, True, False], [False, True, False, True]])

    probabilities = policy(7, observations, action_space, legal_action_mask)

    assert calls == [(7, (1.0, 2.0, 3.0, 4.0), (10, 90)), (7, (5.0, 6.0, 7.0, 8.0), (30, 120))]
    assert all(type(value) is float for _, observation, _ in calls for value in observation)
    assert probabilities.shape == legal_action_mask.shape
    np.testing.assert_array_equal(probabilities, np.array([[0.25, 0.0, 0.75, 0.0], [0.0, 0.6, 0.0, 0.4]]))
    assert np.all(probabilities[~legal_action_mask] == 0.0)


def test_scalar_adapter_rejects_invalid_scalar_distribution() -> None:
    def invalid_scalar_policy(
        player: int, observation: tuple[float, ...], legal_actions: Sequence[int]
    ) -> dict[int, float]:
        del player, observation
        return {action: 0.4 for action in legal_actions}

    policy = PolicyProfile.from_scalar_callable(invalid_scalar_policy)

    with pytest.raises(InvalidPolicyError, match=r"player 3, batch row 0.+must sum to one"):
        policy(3, np.zeros((1, 2, 2)), (10, 30), np.ones((1, 2), dtype=bool))


def test_tabulate_queries_each_player_once_and_answers_per_state() -> None:
    calls: dict[int, list[int]] = {}

    def batched(
        player: int, observations: np.ndarray, action_space: tuple[int, ...], legal_action_mask: np.ndarray
    ) -> np.ndarray:
        calls.setdefault(player, []).append(observations.shape[0])
        probabilities = legal_action_mask.astype(float)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    queries = {0: (((0.0,), (10, 20)), ((1.0,), (20,))), 1: (((2.0,), (10, 20, 30)),)}
    action_spaces = {0: (10, 20, 30), 1: (10, 20, 30)}

    scalar = PolicyProfile(batched).tabulate(queries, action_spaces)

    assert calls == {0: [2], 1: [1]}
    assert scalar(0, (0.0,), (10, 20)) == pytest.approx({10: 0.5, 20: 0.5})
    assert scalar(0, (1.0,), (20,)) == pytest.approx({20: 1.0})
    assert scalar(1, (2.0,), (10, 20, 30)) == pytest.approx({10: 1 / 3, 20: 1 / 3, 30: 1 / 3})


def test_tabulate_rejects_states_it_did_not_materialize() -> None:
    scalar = PolicyProfile(_uniform_batched).tabulate({0: (((0.0,), (10, 20)),)}, {0: (10, 20)})

    with pytest.raises(KeyError, match="not materialized"):
        scalar(0, (9.0,), (10, 20))


def test_tabulate_validates_the_batched_distributions() -> None:
    def invalid(
        player: int, observations: np.ndarray, action_space: tuple[int, ...], legal_action_mask: np.ndarray
    ) -> np.ndarray:
        del player, observations, action_space
        return legal_action_mask.astype(float) * 0.3

    with pytest.raises(InvalidPolicyError, match="must sum to one"):
        PolicyProfile(invalid).tabulate({0: (((0.0,), (10, 20)),)}, {0: (10, 20)})


def test_arity_error_hints_at_scalar_conversion() -> None:
    def scalar_passed_as_batched(
        player: int, observation: tuple[float, ...], legal_actions: Sequence[int]
    ) -> dict[int, float]:
        del player, observation, legal_actions
        return {}

    profile = PolicyProfile(scalar_passed_as_batched)  # type: ignore[arg-type]

    with pytest.raises(TypeError) as error:
        profile(0, np.zeros((1, 1)), (0, 1), np.ones((1, 2), dtype=bool))

    notes = "".join(getattr(error.value, "__notes__", []))
    assert "PolicyProfile.from_scalar_callable()" in notes


def test_type_errors_raised_inside_a_policy_are_not_annotated() -> None:
    def batched(
        player: int, observations: np.ndarray, action_space: tuple[int, ...], legal_action_mask: np.ndarray
    ) -> np.ndarray:
        raise TypeError("something inside the policy went wrong")

    profile = PolicyProfile(batched)

    with pytest.raises(TypeError, match="inside the policy") as error:
        profile(0, np.zeros((1, 1)), (0, 1), np.ones((1, 2), dtype=bool))

    assert not getattr(error.value, "__notes__", [])
