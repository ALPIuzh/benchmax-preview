"""Stockfish adapter checks without downloading or requiring an engine."""

from __future__ import annotations

import asyncio
import os
import threading
from random import Random
from types import SimpleNamespace

import pytest

pytest.importorskip("chess")
import chess.engine

from benchmax.evaluation import IncompatibleEvaluatorError
from benchmax.integrations.openspiel.game import OpenSpielGame
from benchmax.integrations.stockfish.opponent import StockfishOpponent, _StockfishSession


class FakeEngine:
    def __init__(self, move="e2e4"):
        self.id = {"name": "Stockfish test"}
        self.move, self.calls, self.closed = move, [], False
        self.protocol = self
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    async def play(self, board, limit, *, game):
        self.calls.append((board, limit, game))
        return SimpleNamespace(move=chess.Move.from_uci(self.move) if self.move else None)

    def close(self):
        self.closed = True
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()


def environment(moves=()):
    env = OpenSpielGame(spiel_name="chess").new_env()
    env.reset(seed=3)
    for move in moves:
        state = env.spiel_state()
        env.step({state.current_player(): state.parse_move_to_action(move)})
    return env


@pytest.mark.parametrize(
    "kwargs", [{"nodes": 0}, {"nodes": True}, {"hash_mb": 0}, {"timeout": 0}, {"timeout": float("inf")}],
)
def test_invalid_budget(kwargs):
    with pytest.raises(ValueError):
        StockfishOpponent(**kwargs)


def test_rejects_other_games():
    with pytest.raises(IncompatibleEvaluatorError):
        StockfishOpponent().open(OpenSpielGame(spiel_name="tic_tac_toe"))


def test_explicit_missing_executable(tmp_path):
    with pytest.raises(FileNotFoundError, match="setup stockfish"):
        StockfishOpponent(executable=tmp_path / "missing").open(OpenSpielGame(spiel_name="chess"))


def test_history_budget_and_independent_searches():
    env = environment(("g1f3", "g8f6", "f3g1", "f6g8"))
    engine = FakeEngine()
    session = _StockfishSession(engine, 123)
    actions = session.act([env, env], [Random(1), Random(2)])
    assert actions == [env.spiel_state().parse_move_to_action("e2e4")] * 2
    assert len(engine.calls[0][0].move_stack) == 4
    assert engine.calls[0][0].is_repetition(2)
    assert engine.calls[0][1].nodes == 123
    assert engine.calls[0][2] is not engine.calls[1][2]
    assert session.game_value() is None
    assert not session.is_perfect
    session.close()
    assert engine.closed


@pytest.mark.parametrize("move", [None, "e2e5"])
def test_invalid_engine_move(move):
    session = _StockfishSession(FakeEngine(move), 10)
    try:
        with pytest.raises(RuntimeError, match="no legal move"):
            session.act([environment()], [Random(0)])
    finally:
        session.close()


def test_engine_configuration_cleanup(monkeypatch):
    engine = FakeEngine()

    def fail(options):
        assert options == {"Threads": 1, "Hash": 16}
        raise RuntimeError("configuration failed")

    engine.configure = fail
    monkeypatch.setattr("benchmax.integrations.stockfish.opponent.shutil.which", lambda path: "/engine")
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda path, timeout: engine)
    with pytest.raises(RuntimeError, match="configuration failed"):
        StockfishOpponent().open(OpenSpielGame(spiel_name="chess"))
    assert engine.closed


def test_real_engine_reproducible_and_legal():
    executable = os.environ.get("BENCHMAX_STOCKFISH_EXECUTABLE")
    if not executable:
        pytest.skip("set BENCHMAX_STOCKFISH_EXECUTABLE for a real engine smoke test")

    game = OpenSpielGame(spiel_name="chess")
    session = StockfishOpponent(executable=executable, nodes=100).open(game)
    try:
        env = environment()
        first, second = session.act([env, env], [Random(0), Random(1)])
        assert first == second
        assert first in env.spiel_state().legal_actions()
    finally:
        session.close()


def test_node_only_search_has_response_timeout():
    engine = FakeEngine()

    async def stall(*args, **kwargs):
        await asyncio.sleep(10)

    engine.play = stall
    session = _StockfishSession(engine, 10, timeout=0.01)
    try:
        with pytest.raises(TimeoutError):
            session.act([environment()], [Random(0)])
    finally:
        session.close()


def test_castling_conversion():
    env = environment(("e2e4", "e7e5", "g1f3", "b8c6", "f1e2", "g8f6"))
    session = _StockfishSession(FakeEngine("e1g1"), 10)
    try:
        action, = session.act([env], [Random(0)])
        assert action == env.spiel_state().parse_move_to_action("e1g1")
    finally:
        session.close()


@pytest.mark.parametrize("elo", [True, 0, -1, 1800.5, "1800"])
def test_invalid_elo(elo):
    with pytest.raises(ValueError, match="elo"):
        StockfishOpponent(elo=elo)


@pytest.mark.parametrize("elo", [None, 1800])
def test_configured_elo_reaches_engine_and_result_name(monkeypatch, elo):
    from benchmax import load_game

    engine = FakeEngine()
    engine.options = {
        "UCI_Elo": chess.engine.Option("UCI_Elo", "spin", 1320, 1320, 3190, None),
        "UCI_LimitStrength": chess.engine.Option("UCI_LimitStrength", "check", False, None, None, None),
    }
    configured = {}
    engine.configure = configured.update
    monkeypatch.setattr("benchmax.integrations.stockfish.opponent.shutil.which", lambda path: "/engine")
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda path, timeout: engine)
    game = load_game("openspiel.chess", evaluator={"name": "stockfish", "opponent": {"elo": elo}})
    session = game.evaluator.opponent.open(game)
    try:
        expected = {"Threads": 1, "Hash": 16}
        if elo is not None:
            expected.update({"UCI_LimitStrength": True, "UCI_Elo": elo})
        assert configured == expected
        assert ("elo=1800" in session.name) is (elo is not None)
        assert session.is_perfect is False
        assert session.game_value() is None
    finally:
        session.close()


@pytest.mark.parametrize("supported,elo", [(False, 1800), (True, 1319), (True, 3191)])
def test_unsupported_elo_closes_engine(monkeypatch, supported, elo):
    engine = FakeEngine()
    engine.options = {
        "UCI_Elo": chess.engine.Option("UCI_Elo", "spin", 1320, 1320, 3190, None),
        "UCI_LimitStrength": chess.engine.Option("UCI_LimitStrength", "check", False, None, None, None),
    } if supported else {}
    engine.configure = lambda options: pytest.fail("invalid Elo must fail before configuring engine")
    monkeypatch.setattr("benchmax.integrations.stockfish.opponent.shutil.which", lambda path: "/engine")
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda path, timeout: engine)
    with pytest.raises((ValueError, chess.engine.EngineError)):
        StockfishOpponent(elo=elo).open(OpenSpielGame(spiel_name="chess"))
    assert engine.closed


def test_real_engine_accepts_elo():
    executable = os.environ.get("BENCHMAX_STOCKFISH_EXECUTABLE")
    if not executable:
        pytest.skip("set BENCHMAX_STOCKFISH_EXECUTABLE for a real engine smoke test")

    session = StockfishOpponent(executable=executable, nodes=100, elo=1800).open(OpenSpielGame(spiel_name="chess"))
    try:
        env = environment()
        action, = session.act([env], [Random(0)])
        assert action in env.spiel_state().legal_actions()
        assert "elo=1800" in session.name
    finally:
        session.close()
