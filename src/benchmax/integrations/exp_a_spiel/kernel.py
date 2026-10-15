"""Worker-owned exp-a-spiel exact-evaluation kernel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from benchmax.execution.worker import (
    PolicyBatch,
    PolicyOracle,
    evaluation_kernel,
    prepared_resource,
)
from benchmax.types import ActionSpace, Player

_POLICY_BATCH_SIZE = 4096


@evaluation_kernel
def describe_traverser_kernel(traverser_name: str) -> dict[str, Any]:
    """Return native compatibility metadata without constructing the traverser."""

    from benchmax.integrations.exp_a_spiel import _exp_a_spiel

    traverser_type = getattr(_exp_a_spiel, traverser_name, None)
    compatible_game = getattr(traverser_type, "compatible_game", None)
    information_state_size = getattr(traverser_type, "information_state_size", None)
    if not isinstance(compatible_game, dict) or not isinstance(information_state_size, int):
        return {"error": f"unknown exp-a-spiel traverser {traverser_name!r}"}
    return {
        "compatible_game": compatible_game,
        "information_state_size": information_state_size,
        "num_actions": traverser_type.num_actions,
    }


def evaluate_exact_with_oracle(
    exact_evaluator: Any, policy: PolicyOracle, action_spaces: Mapping[Player, ActionSpace]
) -> dict[str, Any]:
    """Materialize both strategies and run the native exact evaluator."""

    strategies = []

    for player in (0, 1):
        rows = exact_evaluator.num_information_states(player)
        action_space = action_spaces[player]
        # Liar's dice stores only legal actions: a dense 21-column table
        # would use over 2.6 GB for the two players, versus about 252 MB.
        packed = hasattr(exact_evaluator, "pack_policy_batch")
        strategy = (
            exact_evaluator.empty_strategy(player)
            if packed
            else np.empty((rows, len(action_space)), dtype=np.float64, order="C")
        )

        for start in range(0, rows, _POLICY_BATCH_SIZE):
            stop = min(start + _POLICY_BATCH_SIZE, rows)
            observations, legal_action_mask = exact_evaluator.policy_input_batch(player, start, stop)
            probabilities = (
                np.empty((stop - start, len(action_space)), dtype=np.float64) if packed else strategy[start:stop]
            )
            policy.evaluate(
                PolicyBatch(
                    player=player,
                    observations=observations,
                    action_space=action_space,
                    legal_action_mask=legal_action_mask,
                    probabilities=probabilities,
                )
            )
            if packed:
                exact_evaluator.pack_policy_batch(player, start, probabilities, strategy)
            del observations, legal_action_mask

        strategies.append(strategy)

    ev0, expl0, expl1 = (float(value) for value in exact_evaluator.evaluate(strategies[0], strategies[1]))
    # expl0 is player 1's improvement against player 0's policy;
    # expl1 is player 0's improvement against player 1's policy.
    return {
        "expected_returns": {0: ev0, 1: -ev0},
        "player_improvements": {0: expl1, 1: expl0},
    }


def _traverser_type(traverser_name: str) -> Any:
    """Import the native binding and return one traverser class."""

    from benchmax.integrations.exp_a_spiel import _exp_a_spiel

    try:
        return getattr(_exp_a_spiel, traverser_name)
    except AttributeError as error:
        raise ValueError(f"unknown exp-a-spiel traverser {traverser_name!r}") from error


@prepared_resource
def _prepared_traverser(traverser_name: str) -> Any:
    """Construct one native traverser in the worker that retains it."""

    return _traverser_type(traverser_name)()


@evaluation_kernel
def evaluate_exact_kernel(
    traverser_name: str, policy: PolicyOracle, action_spaces: Mapping[Player, ActionSpace]
) -> dict[str, Any]:
    """Evaluate through one reusable worker-owned native traverser."""

    return evaluate_exact_with_oracle(_prepared_traverser(traverser_name), policy, action_spaces)
