"""Finite-budget Stockfish opponent for standard OpenSpiel chess."""

from __future__ import annotations

import asyncio
import math
import shutil
from collections.abc import Sequence
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

from benchmax.core import Game, GameInstance
from benchmax.evaluation import IncompatibleEvaluatorError, Opponent, OpponentSession
from benchmax.integrations.openspiel.game import OpenSpielGame
from benchmax.types import Action

if TYPE_CHECKING:
    import chess.engine


class StockfishOpponent(Opponent):
    """Play with one UCI process, a fixed node budget, and a bounded response time.

    Install the ``stockfish`` extra for python-chess and run ``benchmax setup
    stockfish`` separately, or supply an existing executable. The engine hash
    is reset for each position. ``elo`` enables Stockfish's approximate target
    strength, while None leaves strength unrestricted within the node budget. The
    supported Elo range comes from the installed engine. Strength limiting
    uses engine-internal randomness that BenchMAX's seed does not control.
    """

    def __init__(self, *, executable: str | Path | None = None, nodes: int = 10_000,
                 hash_mb: int = 16, timeout: float = 30.0, elo: int | None = None) -> None:
        if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 1:
            raise ValueError("nodes must be a positive integer")

        if isinstance(hash_mb, bool) or not isinstance(hash_mb, int) or hash_mb < 1:
            raise ValueError("hash_mb must be a positive integer")

        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")

        if elo is not None and (isinstance(elo, bool) or not isinstance(elo, int) or elo < 1):
            raise ValueError("elo must be a positive integer or None")

        self.elo = elo
        self.executable = executable
        self.nodes, self.hash_mb, self.timeout = nodes, hash_mb, timeout

    def open(self, game: Game) -> _StockfishSession:
        if not isinstance(game, OpenSpielGame) or game.spiel_name != "chess":
            raise IncompatibleEvaluatorError("Stockfish requires standard OpenSpiel chess")

        if game.params.get("chess960", False):
            raise IncompatibleEvaluatorError("StockfishOpponent currently supports standard chess only")

        try:
            import chess.engine
        except ImportError as exc:
            raise ImportError("Stockfish requires the optional dependency: pip install 'benchmax[stockfish]'") from exc

        from benchmax.integrations.stockfish.install import default_executable

        executable = self.executable or default_executable()
        path = shutil.which(str(executable))
        if path is None:
            raise FileNotFoundError(f"Stockfish executable not found: {executable}. Run benchmax setup stockfish "
                                    "or set opponent.executable to an installed engine.")

        engine = chess.engine.SimpleEngine.popen_uci(path, timeout=self.timeout)
        try:
            options = {"Threads": 1, "Hash": self.hash_mb}
            if self.elo is not None:
                if "UCI_Elo" not in engine.options or "UCI_LimitStrength" not in engine.options:
                    raise ValueError("the installed engine does not support Elo strength limiting")

                engine.options["UCI_Elo"].parse(self.elo)
                options.update({"UCI_LimitStrength": True, "UCI_Elo": self.elo})
            engine.configure(options)
        except BaseException:
            engine.close()
            raise

        return _StockfishSession(engine, self.nodes, self.timeout, self.elo)


class _StockfishSession(OpponentSession):
    is_perfect = False

    def __init__(
        self, engine: chess.engine.SimpleEngine, nodes: int, timeout: float = 30.0, elo: int | None = None,
    ) -> None:
        self._engine, self._nodes, self._timeout = engine, nodes, timeout
        strength = "" if elo is None else f", elo={elo}"
        self.name = f"{engine.id.get('name', 'Stockfish')} (nodes={nodes}, threads=1{strength})"

    def game_value(self) -> None:
        """Finite search provides no exact game value."""
        return

    def act(self, envs: Sequence[GameInstance], rngs: Sequence[Random]) -> list[Action]:
        import chess

        if len(envs) != len(rngs):
            raise ValueError("one RNG is required per environment")

        actions = []
        for env in envs:
            state = env.spiel_state()
            if state.is_terminal():
                raise ValueError("Stockfish cannot act on a terminal state")

            board = chess.Board(state.start_fen())
            for move in state.moves_history():
                board.push_uci(move.to_lan())

            result = asyncio.run_coroutine_threadsafe(self._play(board), self._engine.protocol.loop).result()
            if result.move is None or result.move not in board.legal_moves:
                raise RuntimeError("Stockfish returned no legal move")

            action = state.parse_move_to_action(result.move.uci())
            if action not in state.legal_actions():
                raise RuntimeError(f"Stockfish move {result.move.uci()} is illegal in OpenSpiel")

            actions.append(action)
        return actions

    async def _play(self, board: chess.Board) -> chess.engine.PlayResult:
        import chess.engine

        # SimpleEngine.play disables its timeout for node-only limits. A fresh
        # game token triggers ucinewgame, clearing Stockfish's search hash.
        return await asyncio.wait_for(
            self._engine.protocol.play(board, chess.engine.Limit(nodes=self._nodes), game=object()), self._timeout
        )

    def close(self) -> None:
        # close() terminates the transport even if the engine stopped responding.
        self._engine.close()
