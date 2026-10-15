"""Budgeted best-response search with sampled or model-based return evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from benchmax._rollout import episode_returns
from benchmax.core import Game
from benchmax.evaluation._base import IncompatibleEvaluatorError, require_undiscounted
from benchmax.games.matrix import TwoPlayerZeroSumMatrixGame
from benchmax.games.tabular.game import TabularZeroSumMarkovGame
from benchmax.games.tabular.solver import profile_value
from benchmax.policy import PolicyProfile, PolicyProfileLike, _as_policy_profile
from benchmax.seeding import SEED_SCHEME, derive_seed, validate_seed
from benchmax.types import Player


def _deviation(reference: PolicyProfile, response: PolicyProfile, player: Player) -> PolicyProfile:
    return PolicyProfile(lambda p, obs, actions, mask: (response if p == player else reference)(p, obs, actions, mask))


def _greedy(profile: PolicyProfile) -> PolicyProfile:
    def policy(player, observations, actions, mask):
        probabilities = profile(player, observations, actions, mask)
        result = np.zeros_like(probabilities)
        result[np.arange(len(result)), probabilities.argmax(axis=1)] = 1.0
        return result

    return PolicyProfile(policy)


class ApproximateExploitabilityEvaluator:
    """Approximate exploitability for two-player zero-sum games.

    Each player's best response is trained with PPO against the frozen
    opponent, or supplied through ``response_profiles``, which needs no
    training dependencies. Small validation batches select among training
    checkpoints, their greedy variants, and the reference policy. The result
    averages the two selected responses' returns.

    Expected response values lower-bound exact exploitability, but finite
    sample estimates can overshoot. Reported sampling errors do not account for
    how far a response is from optimal.

    The registered evaluator uses a compact PPO preset from its YAML
    configuration. Direct construction forwards ``params`` to training, with
    omitted settings taken from the training entry's defaults. After a direct
    evaluation, ``responses`` holds the selected policies.

    Episodes must terminate within ``game.max_episode_steps``. Discounted
    tabular games are valued exactly by their model's solver instead, so only
    response optimality is approximate.
    """

    def __init__(
        self,
        *,
        algorithm: str = "ippo",
        total_frames: int = 100_000,
        restarts: int = 1,
        checkpoints: int = 4,
        greedy: bool = True,
        per_player_metrics: bool = False,
        episodes: int = 1_000,
        validation_episodes: int = 256,
        batch_size: int = 128,
        seed: int = 0,
        device: str = "auto",
        cuda_graphs: bool = True,
        torch_threads: int | None = 1,
        matmul_precision: str | None = None,
        params: Mapping[str, Any] | None = None,
        response_profiles: Mapping[Player, PolicyProfileLike] | None = None,
    ) -> None:
        for name, value in (
            ("total_frames", total_frames),
            ("restarts", restarts),
            ("checkpoints", checkpoints),
            ("episodes", episodes),
            ("validation_episodes", validation_episodes),
            ("batch_size", batch_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < (2 if name == "episodes" else 1):
                raise ValueError(f"{name} must be an integer >= {2 if name == 'episodes' else 1}")
        for name, value in (("greedy", greedy), ("per_player_metrics", per_player_metrics)):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        self.seed = validate_seed(seed)
        self.algorithm, self.total_frames, self.restarts = algorithm, total_frames, restarts
        self.checkpoints, self.greedy, self.per_player_metrics = checkpoints, greedy, per_player_metrics
        self.episodes, self.validation_episodes, self.batch_size = episodes, validation_episodes, batch_size
        self.device = device
        self.params = params
        self.cuda_graphs, self.torch_threads, self.matmul_precision = cuda_graphs, torch_threads, matmul_precision
        self._provided = response_profiles
        self.responses: dict[Player, PolicyProfile] = {}

    def _returns(self, game: Game, profile: PolicyProfile, episodes: int, *stream: str | int) -> np.ndarray:
        """Return sampled episodes, or one exact expectation for a tabular model."""
        if isinstance(game, TabularZeroSumMarkovGame):
            policies = game._policy_tables(profile)
            value = profile_value(
                game.transitions,
                game.payoffs,
                policies[0],
                policies[1],
                game.initial_state_distribution,
                game.discount,
            )
            return np.asarray([[value, -value]])
        return episode_returns(
            game,
            profile,
            episodes=episodes,
            batch_size=self.batch_size,
            seed=derive_seed(self.seed, "approximate-exploitability", *stream),
        )

    def evaluate(self, game: Game, policy_profile: PolicyProfile) -> dict[str, Any]:
        exact_returns = isinstance(game, TabularZeroSumMarkovGame)
        if not exact_returns:
            require_undiscounted(game, "approximate_exploitability")
        spiel = getattr(game, "spiel_game", None)
        if len(game.players) != 2 or not (
            isinstance(game, (TwoPlayerZeroSumMatrixGame, TabularZeroSumMarkovGame))
            or spiel is not None
            and spiel.get_type().utility.name == "ZERO_SUM"
        ):
            raise IncompatibleEvaluatorError(
                "approximate_exploitability requires a two-player zero-sum matrix, tabular Markov or OpenSpiel game"
            )
        if self._provided is not None and set(self._provided) != set(game.players):
            raise ValueError("response_profiles must contain exactly the game's players")

        trainer = None
        if self._provided is None:
            from benchmax.integrations.benchmarl.training import BenchMARLTrainingFunction
            from benchmax.training.algorithm import load_training_function

            trainer = load_training_function(
                self.algorithm,
                total_frames=self.total_frames,
                device=self.device,
                verbose=False,
                cuda_graphs=self.cuda_graphs,
                torch_threads=self.torch_threads,
                matmul_precision=self.matmul_precision,
                **({"params": self.params} if self.params is not None else {}),
            )
            if not isinstance(trainer, BenchMARLTrainingFunction):
                raise TypeError("response training requires a BenchMARL algorithm; otherwise supply response_profiles")

        reference = _as_policy_profile(policy_profile)
        validation = self._returns(game, reference, self.validation_episodes, "validation", "reference").mean(axis=0)
        selected, selected_seeds, sources, modes, selected_frames = {}, {}, {}, {}, {}
        checkpoint_counts = dict.fromkeys(game.players, 0)
        actual_training_frames = 0
        # Geometric spacing spends most of the budget learning, while retaining
        # useful early responses. The final completed update is always included.
        checkpoint_frames = tuple(sorted({
            max(1, (self.total_frames + (1 << i) - 1) // (1 << i)) for i in range(self.checkpoints)
        }))
        for column, player in enumerate(game.players):
            selected[player], selected_seeds[player], sources[player] = reference, None, "reference"
            modes[player], selected_frames[player] = "reference", None
            best = validation[column]
            for restart in range(self.restarts if trainer is not None else 1):
                training_seed = derive_seed(self.seed, "approximate-exploitability", "training", player, restart)
                seen_frames = 0

                def select(
                    response: PolicyProfile, frames: int, *, player=player, column=column,
                    restart=restart, training_seed=training_seed,
                ) -> None:
                    nonlocal best, seen_frames
                    seen_frames = frames
                    checkpoint_counts[player] += 1
                    candidates = {"stochastic": response}
                    if self.greedy:
                        candidates["greedy"] = _greedy(response)
                    for mode, candidate in candidates.items():
                        score = self._returns(
                            game, _deviation(reference, candidate, player), self.validation_episodes,
                            "validation", player, restart, frames, mode,
                        )[:, column].mean()
                        if score > best:
                            best = score
                            selected[player] = candidate
                            selected_seeds[player] = training_seed if trainer is not None else None
                            sources[player] = "trained" if trainer is not None else "provided"
                            modes[player] = mode
                            selected_frames[player] = frames if trainer is not None else None

                if trainer is not None:
                    trainer.best_response(
                        game, reference, player, seed=training_seed,
                        checkpoint_frames=checkpoint_frames, on_checkpoint=select,
                    )
                    actual_training_frames += seen_frames
                else:
                    select(_as_policy_profile(self._provided[player]), 0)

        response_samples, baseline = [], None
        for column, player in enumerate(game.players):
            if sources[player] == "reference" and baseline is not None:
                returns = baseline
            else:
                returns = self._returns(
                    game, _deviation(reference, selected[player], player), self.episodes, "test", player,
                )
                if sources[player] == "reference":
                    baseline = returns
            response_samples.append(returns[:, column])
        response_returns = np.column_stack(response_samples)
        means = response_returns.mean(axis=0)
        raw_exploitability = float(means.sum() / 2)
        reference_only = all(source == "reference" for source in sources.values())
        standard_error = (
            0.0 if exact_returns or reference_only
            else float(np.sqrt(response_returns.var(axis=0, ddof=1).sum() / self.episodes) / 2)
        )
        metrics = {
            "approximate_exploitability": max(0.0, raw_exploitability),
            "approximate_nash_conv": max(0.0, 2 * raw_exploitability),
            "raw_approximate_exploitability": raw_exploitability,
        }
        if self.per_player_metrics:
            if baseline is None:
                baseline = self._returns(game, reference, self.episodes, "test", "reference")
            gains = response_returns - baseline
            raw = gains.mean(axis=0)
            errors = np.zeros(2) if exact_returns else gains.std(axis=0, ddof=1) / np.sqrt(self.episodes)
            metrics.update({
                "expected_returns": dict(zip(game.players, baseline.mean(axis=0).tolist(), strict=True)),
                "player_improvements": dict(zip(game.players, np.maximum(0, raw).tolist(), strict=True)),
                "raw_player_improvements": dict(zip(game.players, raw.tolist(), strict=True)),
                "player_improvement_standard_errors": dict(zip(game.players, errors.tolist(), strict=True)),
                "approximate_nash_gap": float(max(0, raw.max())),
            })
        self.responses = selected
        return {
            **metrics,
            "response_returns": dict(zip(game.players, means.tolist(), strict=True)),
            "raw_exploitability_standard_error": standard_error,
            "response_policy_sources": sources,
            "response_action_modes": modes,
            "selected_response_seeds": selected_seeds,
            "selected_response_frames": selected_frames,
            "evaluated_checkpoints_per_player": checkpoint_counts,
            "return_estimation": "exact_model" if exact_returns else "monte_carlo",
            "discount": game.discount,
            "episodes_per_profile": 0 if exact_returns else self.episodes,
            "validation_episodes_per_candidate": 0 if exact_returns else self.validation_episodes,
            "training_frames_per_restart": self.total_frames if trainer is not None else 0,
            "actual_training_frames": actual_training_frames,
            "restarts_per_player": self.restarts if trainer is not None else 0,
            "response_algorithm": self.algorithm if trainer is not None else None,
            "seed": self.seed,
            "seed_scheme": SEED_SCHEME,
        }
