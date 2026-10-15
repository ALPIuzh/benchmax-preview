"""Exact model-based value computations for zero-sum tabular Markov games.

The fixed-profile value uses a linear solve, and best responses use value
iteration. Both runtime and memory depend on the model size, and value
iteration also depends on the discount and requested tolerance.
"""

from __future__ import annotations

import numpy as np


def profile_value(
    transitions: np.ndarray, payoffs: np.ndarray, policy_0: np.ndarray, policy_1: np.ndarray,
    initial_state_distribution: np.ndarray, discount: float
) -> float:
    """Return player 0's discounted value under a fixed strategy profile.

    The profile induces a Markov chain, so the value solves a linear system
    exactly rather than by iteration.
    """

    weights = policy_0[:, :, None] * policy_1[:, None, :]
    expected_reward = (payoffs * weights).sum(axis=(1, 2))
    chain = (transitions * weights[:, :, :, None]).sum(axis=(1, 2))
    values = np.linalg.solve(
        np.eye(len(chain)) - discount * chain, expected_reward
    )
    return float(initial_state_distribution @ values)


def best_response_value(
    rewards: np.ndarray, transitions: np.ndarray, initial_state_distribution: np.ndarray, discount: float, *,
    tolerance: float = 1e-6, max_iterations: int = 10_000
) -> float:
    """Return the optimal discounted value of a single-agent MDP.

    Args:
        rewards: Expected reward, shape ``(S, A)``.
        transitions: Transition tensor, shape ``(S, A, S)``.
        initial_state_distribution: Distribution over starting states.
        discount: Discount factor in ``[0, 1)``.
        tolerance: Convergence threshold on the sup-norm change in values.
        max_iterations: Cap on the number of sweeps.

    Returns:
        The optimal value averaged over the initial state distribution.
    """

    values = np.zeros(transitions.shape[0])

    for _ in range(max_iterations):
        updated = np.max(rewards + discount * transitions.dot(values), axis=1)
        if np.max(np.abs(updated - values)) < tolerance:
            values = updated
            break

        values = updated

    return float(initial_state_distribution @ values)
