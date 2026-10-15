"""Reference opponent backed by an explicitly installed Pons executable."""

from __future__ import annotations

import contextlib
import math
import os
import selectors
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random

from benchmax.core import Game, GameInstance
from benchmax.evaluation import IncompatibleEvaluatorError, Opponent, OpponentSession
from benchmax.integrations.openspiel.game import OpenSpielGame, pyspiel
from benchmax.integrations.pons.install import built_for, default_directory, validate_board
from benchmax.types import Action, Player

_INVALID_MOVE = -1000


class PonsError(RuntimeError):
    """The installed Pons engine failed or returned an invalid response."""


class PonsOpponent(Opponent):
    """Perfect Connect Four play using a persistent external engine.

    ``executable`` must be built for the game's board dimensions. The opening
    book, if any, lives next to it. No installation occurs during evaluation.
    ``timeout`` bounds each position query, including initial game-value search.
    """

    def __init__(self, *, executable: str | Path | None = None, timeout: float = 60) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.executable = None if executable is None else Path(executable).expanduser().resolve()
        self.timeout = timeout

    def open(self, game: Game) -> _PonsSession:
        if not isinstance(game, OpenSpielGame) or game.spiel_name != "connect_four":
            raise IncompatibleEvaluatorError("Pons requires OpenSpiel connect_four")

        params = game.spiel_game.get_parameters()
        rows, columns = params["rows"], params["columns"]
        if params["x_in_row"] != 4 or game.spiel_game.get_type().utility != pyspiel.GameType.Utility.ZERO_SUM:
            raise IncompatibleEvaluatorError("Pons requires zero-sum Connect Four with four in a row")

        try:
            validate_board(rows, columns)
        except ValueError as error:
            raise IncompatibleEvaluatorError(str(error)) from error

        executable = self.executable or default_directory(rows, columns) / "c4solver"
        if not executable.is_file():
            raise FileNotFoundError(
                f"Pons executable missing: {executable}. Run `benchmax setup pons --rows {rows} --columns {columns}` "
                "or configure opponent.executable with a matching build."
            )

        if built_for(executable.parent, rows, columns) is False:
            raise IncompatibleEvaluatorError(f"Pons installation at {executable} was built for a different board")
        return _PonsSession(executable, columns, self.timeout)


class _PonsSession(OpponentSession):
    name = "pons"
    is_perfect = True

    def __init__(self, executable: Path, columns: int, timeout: float) -> None:
        self._columns, self._timeout = columns, timeout
        self._scores: dict[str, tuple[int | None, ...]] = {}
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115 - session owns diagnostics
        try:
            self._process = subprocess.Popen(
                [str(executable), "-a"], cwd=executable.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, bufsize=0,
            )
        except OSError:
            self._stderr.close()
            raise

    def act(self, envs: Sequence[GameInstance], rngs: Sequence[Random]) -> list[Action]:
        moves = []
        for env, rng in zip(envs, rngs, strict=True):
            position = "".join(str(column + 1) for column in env.spiel_state().history())
            scores = self._solve(position)
            best = max(score for score in scores if score is not None)
            moves.append(rng.choice([column for column, score in enumerate(scores) if score == best]))
        return moves

    def game_value(self) -> Mapping[Player, float]:
        best = max(score for score in self._solve("") if score is not None)
        first = float((best > 0) - (best < 0))
        return {0: first, 1: -first}

    def close(self) -> None:
        with contextlib.suppress(BrokenPipeError):
            self._process.stdin.close()
        try:
            self._process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process.stdout.close()
        self._stderr.close()

    def _solve(self, position: str) -> tuple[int | None, ...]:
        if position in self._scores:
            return self._scores[position]

        try:
            self._process.stdin.write(f"{position}\n".encode())
        except BrokenPipeError as error:
            raise self._failure("stopped accepting positions") from error

        # One small request at a time prevents input/output pipe deadlock. Raw
        # reads avoid buffered readline hiding bytes from the readiness check.
        deadline, response = time.monotonic() + self._timeout, bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            while not response.endswith(b"\n"):
                if not selector.select(max(0, deadline - time.monotonic())):
                    raise self._failure(f"timed out after {self._timeout:g}s at position {position!r}")
                chunk = os.read(self._process.stdout.fileno(), 4096)
                if not chunk:
                    raise self._failure(f"exited before answering position {position!r}")
                response.extend(chunk)
                if len(response) > 4096 or time.monotonic() > deadline:
                    raise self._failure("invalid or overdue response")

        fields = response.split()
        echoed, values = fields[:-self._columns], fields[-self._columns:]
        if echoed != ([position.encode()] if position else []) or len(values) != self._columns:
            raise self._failure(f"invalid response for position {position!r}")

        try:
            scores = tuple(None if int(v) == _INVALID_MOVE else int(v) for v in values)
        except ValueError as error:
            raise self._failure("non-integer move score") from error

        if all(score is None for score in scores):
            raise self._failure("no playable move in a nonterminal position")
        self._scores[position] = scores
        return scores

    def _failure(self, message: str) -> PonsError:
        self._stderr.seek(0)
        diagnostics = self._stderr.read().decode(errors="replace").strip()
        return PonsError(f"Pons {message}" + (f": {diagnostics}" if diagnostics else ""))
