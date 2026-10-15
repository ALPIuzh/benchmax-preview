"""Evaluate a policy by playing complete episodes against a reference opponent."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from random import Random
from typing import Any, Protocol, runtime_checkable

import numpy as np

from benchmax._rollout import sample_actions
from benchmax.core import Game, GameInstance
from benchmax.evaluation._base import IncompatibleEvaluatorError, require_undiscounted
from benchmax.policy import PolicyProfile, _as_policy_profile
from benchmax.seeding import SEED_SCHEME, derive_seed, python_generator, validate_seed
from benchmax.types import Action, Player


@runtime_checkable
class OpponentSession(Protocol):
    """An opponent opened for one game; :meth:`close` releases it when the evaluation ends.

    Attributes:
        name: Short identifier reported with the metrics.
    """

    name: str
    is_perfect: bool

    def act(self, envs: Sequence[GameInstance], rngs: Sequence[Random]) -> list[Action]:
        """Choose the move for the player to act in each environment.

        Each environment has exactly one acting player. ``rngs`` belong to the
        environments and break ties between equally good moves reproducibly.
        """

        ...

    def game_value(self) -> Mapping[Player, float] | None:
        """Exact initial game values, or None when the opponent cannot provide them.

        A backend providing values must validate that the game is zero-sum.
        ``is_perfect`` additionally guarantees optimal play under the configured
        search budget. Only both guarantees enable solver-bound metrics.
        """
        ...

    def close(self) -> None: ...


@runtime_checkable
class Opponent(Protocol):
    """Configuration of a reference opponent, opened once per evaluation."""

    def open(self, game: Game) -> OpponentSession:
        """Validate ``game`` and start the opponent for it.

        Raises:
            IncompatibleEvaluatorError: If the opponent cannot play ``game``.
        """

        ...


class OpponentEvaluator:
    """Play each selected policy seat against a fixed opponent.

    Reports sampled undiscounted returns and their standard errors. Win, draw
    and loss rates interpret positive, zero and negative episode returns as
    outcomes, so use games with rewards following that convention. These are
    opponent-specific match results. Perfect opponents with exact game values
    additionally provide sampled exploitability lower bounds, never exact
    exploitability. The backend must validate zero-sum semantics for those bounds.

    Supports sequential two-player games.
    """

    def __init__(
        self, *, opponent: Opponent, seats: Sequence[Player] | None = None, episodes: int = 1_000,
        batch_size: int = 128, seed: int = 0,
    ) -> None:
        if not isinstance(opponent, Opponent):
            raise TypeError(f"opponent must implement the Opponent protocol; got {type(opponent).__name__}")

        for name, value, minimum in (("episodes", episodes, 2), ("batch_size", batch_size, 1)):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

        if seats is not None and (not seats or len(set(seats)) != len(seats)):
            raise ValueError("seats must list distinct players")

        self.opponent, self.episodes, self.batch_size = opponent, episodes, batch_size
        self.seats = None if seats is None else tuple(seats)
        self.seed = validate_seed(seed)

    def evaluate(self, game: Game, policy_profile: PolicyProfile) -> dict[str, Any]:
        require_undiscounted(game, "OpponentEvaluator")
        if len(game.players) != 2 or game.is_simultaneous:
            raise IncompatibleEvaluatorError(
                f"OpponentEvaluator serves sequential two-player games; {game.name!r} is not one"
            )

        seats = game.players if self.seats is None else self.seats
        if not set(seats) <= set(game.players):
            raise ValueError(f"seats {seats} are not players of {game.name!r}")

        with contextlib.closing(self.opponent.open(game)) as session:
            profile = _as_policy_profile(policy_profile)
            value = session.game_value() if session.is_perfect else None
            if value is not None:
                if set(value) != set(game.players) or not all(np.isfinite(v) for v in value.values()):
                    raise ValueError("exact game values must contain one finite value per player")

                if not np.isclose(sum(value.values()), 0.0, rtol=0, atol=1e-12):
                    raise ValueError("solver bounds require zero-sum game values")

            returns = {seat: self._play(game, profile, session, seat) for seat in seats}

        metrics: dict[str, Any] = {
            "opponent": session.name,
            "seats": list(seats),
            "is_perfect": session.is_perfect,
            "game_value": None if value is None else dict(value),
            "expected_returns": {seat: float(r.mean()) for seat, r in returns.items()},
            "win_rates": {seat: float((r > 0).mean()) for seat, r in returns.items()},
            "draw_rates": {seat: float((r == 0).mean()) for seat, r in returns.items()},
            "loss_rates": {seat: float((r < 0).mean()) for seat, r in returns.items()},
            "return_standard_errors": {
                seat: float(r.std(ddof=1) / np.sqrt(self.episodes)) for seat, r in returns.items()
            },
        }
        if value is not None:
            # These are minimax shortfalls, not individual unilateral-deviation
            # improvements. Their sum lower-bounds two-player zero-sum NashConv.
            gaps = {seat: float(value[seat] - r.mean()) for seat, r in returns.items()}
            bound = sum(gaps.values())
            error = float(np.sqrt(sum(r.var(ddof=1) for r in returns.values()) / self.episodes))
            metrics.update({
                "solver_player_gaps": gaps, "solver_nash_conv": bound, "solver_exploitability": bound / 2,
                "solver_exploitability_standard_error": error / 2,
            })

        return {**metrics, "episodes_per_seat": self.episodes, "seed": self.seed, "seed_scheme": SEED_SCHEME}

    def _play(self, game: Game, profile: PolicyProfile, session: OpponentSession, seat: Player) -> np.ndarray:
        """Return the policy's undiscounted return in each episode played from ``seat``."""
        returns = np.zeros(self.episodes)
        for start in range(0, self.episodes, self.batch_size):
            indices = range(start, min(start + self.batch_size, self.episodes))
            envs = [game.new_env() for _ in indices]
            steps = [env.reset(seed=derive_seed(self.seed, "match", seat, "env", i)) for env, i in
                     zip(envs, indices, strict=True)]
            policy_rngs = [python_generator(derive_seed(self.seed, "match", seat, "policy", i)) for i in indices]
            opponent_rngs = [python_generator(derive_seed(self.seed, "match", seat, "opponent", i)) for i in indices]

            for _ in range(game.max_episode_steps):
                active = [i for i, step in enumerate(steps) if not step.terminated]
                if not active:
                    break

                if any(steps[i].truncated for i in active):
                    raise ValueError("OpponentEvaluator requires complete episodes; environment truncated")

                policy_rows = [i for i in active if steps[i].players_to_act == (seat,)]
                opponent_rows = [i for i in active if steps[i].players_to_act != (seat,)]

                actions: dict[int, Mapping[Player, Action]] = dict(zip(policy_rows, sample_actions(
                    game, profile, [steps[i] for i in policy_rows], [policy_rngs[i] for i in policy_rows],
                    players=(seat,),
                ), strict=True))
                moves = session.act(
                    [envs[i] for i in opponent_rows], [opponent_rngs[i] for i in opponent_rows]
                ) if opponent_rows else []
                for i, move in zip(opponent_rows, moves, strict=True):
                    actions[i] = {steps[i].players_to_act[0]: move}

                for i in active:
                    steps[i] = envs[i].step(actions[i])
                    returns[start + i] += steps[i].rewards[seat]

            if any(not step.terminated for step in steps):
                raise ValueError("OpponentEvaluator requires episodes to terminate within max_episode_steps")
        return returns
