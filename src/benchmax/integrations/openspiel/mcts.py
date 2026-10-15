"""OpenSpiel Monte Carlo tree search as an evaluation opponent."""

from __future__ import annotations

import contextlib
import io
import math
from collections.abc import Sequence
from random import Random

with contextlib.redirect_stdout(io.StringIO()):
    import pyspiel

from benchmax.core import Game, GameInstance
from benchmax.evaluation import IncompatibleEvaluatorError
from benchmax.evaluation.opponent import Opponent, OpponentSession
from benchmax.integrations.openspiel.game import OpenSpielGame
from benchmax.types import Action


class OpenSpielMCTSOpponent(Opponent):
    """A configurable, in-process MCTS opponent for sequential perfect-information games.

    Each move starts a fresh search using its episode's RNG. ``solve=True``
    propagates outcomes proved during search, but a finite search budget still
    does not guarantee optimal play or certify exploitability.
    """

    name = "mcts"

    def __init__(self, *, max_simulations: int = 1_000, uct_c: float = 2.0, rollouts: int = 1,
                 solve: bool = True) -> None:
        for setting, value in (("max_simulations", max_simulations), ("rollouts", rollouts)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{setting} must be an integer >= 1")

        if not math.isfinite(uct_c) or uct_c <= 0:
            raise ValueError("uct_c must be finite and positive")

        if not isinstance(solve, bool):
            raise TypeError("solve must be a boolean")

        self.max_simulations, self.uct_c, self.rollouts, self.solve = max_simulations, float(uct_c), rollouts, solve

    def open(self, game: Game) -> _MCTSSession:
        if not isinstance(game, OpenSpielGame):
            raise IncompatibleEvaluatorError(
                f"{self.name} requires an OpenSpiel game; got {game.name!r} ({type(game).__name__})"
            )

        game_type = game.spiel_game.get_type()
        if game_type.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
            raise IncompatibleEvaluatorError(f"{self.name} requires a perfect-information game; {game.name!r} is not")

        if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise IncompatibleEvaluatorError(f"{self.name} requires a sequential game; {game.name!r} is not")

        if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
            raise IncompatibleEvaluatorError(f"{self.name} requires terminal rewards; {game.name!r} has intermediate rewards")

        return _MCTSSession(self, game.spiel_game)


class _MCTSSession(OpponentSession):
    name = OpenSpielMCTSOpponent.name
    is_perfect = False

    def __init__(self, opponent: OpenSpielMCTSOpponent, spiel_game: pyspiel.Game) -> None:
        self._opponent, self._spiel_game = opponent, spiel_game

    def act(self, envs: Sequence[GameInstance], rngs: Sequence[Random]) -> list[Action]:
        actions = []
        for env, rng in zip(envs, rngs, strict=True):
            seed = rng.getrandbits(31)  # OpenSpiel takes signed 32-bit seeds.
            evaluator = pyspiel.RandomRolloutEvaluator(self._opponent.rollouts, seed)
            bot = pyspiel.MCTSBot(
                self._spiel_game, evaluator, self._opponent.uct_c, self._opponent.max_simulations,
                1024, self._opponent.solve, seed, False
            )
            actions.append(int(bot.step(env.spiel_state())))

        return actions

    def game_value(self) -> None:
        return None

    def close(self) -> None:
        pass
