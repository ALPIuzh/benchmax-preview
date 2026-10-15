"""Batched policy sampling over the ordinary environment interface."""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

import numpy as np

from benchmax.core import Game, TimeStep
from benchmax.policy import PolicyProfile, PolicyProfileLike, _as_policy_profile
from benchmax.seeding import derive_seed, python_generator, validate_seed
from benchmax.types import Action, Player


def sample_actions(
    game: Game,
    profile: PolicyProfileLike,
    steps: Sequence[TimeStep],
    rngs: Sequence[Random],
    *,
    players: Sequence[Player] | None = None,
) -> list[dict[Player, Action]]:
    """Query each acting player's policy once per batch, never at hidden states."""
    profile = _as_policy_profile(profile)
    actions: list[dict[Player, Action]] = [{} for _ in steps]
    for player in game.players if players is None else players:
        rows = [i for i, step in enumerate(steps) if player in step.players_to_act]
        if not rows:
            continue
        space = game.action_spaces[player]
        observations = np.asarray([steps[i].observations[player] for i in rows], dtype=float)
        mask = np.asarray([[a in steps[i].legal_actions[player] for a in space] for i in rows])
        probabilities = profile(player, observations, space, mask)
        for row, i in enumerate(rows):
            actions[i][player] = rngs[i].choices(space, weights=probabilities[row], k=1)[0]
    return actions


def rollout(
    game: Game, policy_profile: PolicyProfileLike, *, episodes: int = 1, seed: int, batch_size: int = 32,
    max_steps: int | Sequence[int] | np.ndarray | None = None, collect_data: bool = False,
    require_complete: bool = False
) -> tuple[np.ndarray, dict[Player, dict[str, np.ndarray]]]:
    """Shared implementation of Game.rollout, collecting decision arrays only on request."""
    seed = validate_seed(seed)
    for name, value, minimum in (("episodes", episodes, 0), ("batch_size", batch_size, 1)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")

        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")

    limits = np.asarray(game.max_episode_steps if max_steps is None else max_steps, dtype=object)
    if limits.ndim > 1 or (limits.ndim == 1 and limits.shape != (episodes,)):
        raise ValueError("max_steps must be a scalar or a one-dimensional array of length episodes")

    for limit in limits.flat:
        if not isinstance(limit, (int, np.integer)) or isinstance(limit, (bool, np.bool_)):
            raise TypeError("max_steps must contain integers")

        if limit < 1:
            raise ValueError("max_steps must be positive")

    limits = np.broadcast_to(limits, (episodes,))

    profile = _as_policy_profile(policy_profile)
    returns = np.zeros((episodes, len(game.players)))
    rows = {
        player: {
            "observations": [np.empty((0, *game.observation_spaces[player]), dtype=float)],
            "legal_action_mask": [np.empty((0, len(game.action_spaces[player])), dtype=bool)],
            "actions": [], "episode_ids": [], "step_ids": [],
        }
        for player in game.players if collect_data
    }
    for start in range(0, episodes, batch_size):
        indices = range(start, min(start + batch_size, episodes))
        envs = [game.new_env() for _ in indices]
        steps = [env.reset(seed=derive_seed(seed, "env", i)) for env, i in zip(envs, indices, strict=True)]
        rngs = [python_generator(derive_seed(seed, "actions", i)) for i in indices]
        horizons = limits[start:start + len(envs)]
        for t in range(max(horizons)):
            active = [i for i, step in enumerate(steps)
                      if t < horizons[i] and not (step.terminated or step.truncated)]
            if not active:
                break

            actions = sample_actions(game, profile, [steps[i] for i in active], [rngs[i] for i in active])
            for player, data in rows.items():
                acting = [(i, joint[player]) for i, joint in zip(active, actions, strict=True) if player in joint]
                if not acting:
                    continue

                # Materialize before env.step: environments may reuse mutable observation buffers.
                data["observations"].append(np.array([steps[i].observations[player] for i, _ in acting], dtype=float))
                data["actions"].extend(action for _, action in acting)
                data["legal_action_mask"].append(np.array(
                    [[a in steps[i].legal_actions[player] for a in game.action_spaces[player]] for i, _ in acting],
                    dtype=bool
                ))
                data["episode_ids"].extend(start + i for i, _ in acting)
                data["step_ids"].extend([t] * len(acting))

            for i, joint in zip(active, actions, strict=True):
                steps[i] = envs[i].step(joint)
                returns[start + i] += [steps[i].rewards[p] for p in game.players]

        if require_complete and any(not step.terminated or step.truncated for step in steps):
            raise ValueError(
                "rollout requires complete episodes to terminate within max_episode_steps; "
                "environment truncated or max_steps reached"
            )

    datasets = {}
    for player, data in rows.items():
        datasets[player] = {
            "observations": np.concatenate(data["observations"]),
            "actions": np.asarray(data["actions"], dtype=int),
            "legal_action_mask": np.concatenate(data["legal_action_mask"]),
            "episode_ids": np.asarray(data["episode_ids"], dtype=int),
            "step_ids": np.asarray(data["step_ids"], dtype=int),
        }
    return returns, datasets


def episode_returns(game: Game, profile: PolicyProfile, *, episodes: int, seed: int, batch_size: int) -> np.ndarray:
    """Sample complete, undiscounted returns without collecting decision arrays."""
    returns, _ = rollout(game, profile, episodes=episodes, seed=seed, batch_size=batch_size, require_complete=True)
    return returns
