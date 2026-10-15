import warnings
from random import Random

import pytest

from benchmax import load_game
from benchmax.evaluation import IncompatibleEvaluatorError
from benchmax.integrations.openspiel.mcts import OpenSpielMCTSOpponent


def _uniform(player, observations, actions, mask):
    return mask / mask.sum(axis=1, keepdims=True)


def _game(name="openspiel.tic_tac_toe", episodes=6, **opponent):
    return load_game(name, evaluator={"name": "mcts", "episodes": episodes, "opponent": opponent})


def test_a_strong_search_never_loses_tic_tac_toe_to_random_play() -> None:
    results = _game(episodes=10, max_simulations=1_000).evaluate(_uniform)

    assert results["opponent"] == "mcts"
    assert results["seats"] == [0, 1]
    assert results["win_rates"] == {0: 0.0, 1: 0.0}
    assert results["game_value"] is None
    assert "solver_exploitability" not in results


def test_search_is_seeded_per_episode() -> None:
    def run(seed, batch_size):
        game = load_game("openspiel.tic_tac_toe", evaluator={
            "name": "mcts", "episodes": 6, "seed": seed, "batch_size": batch_size,
            "opponent": {"max_simulations": 20},
        })
        return game.evaluate(_uniform)

    assert run(1, 128) == run(1, 2)
    assert run(1, 128) != run(2, 128)


def test_search_leaves_environment_unchanged_and_returns_legal_moves() -> None:
    game = load_game("openspiel.connect_four")
    env = game.new_env()
    step = env.reset(seed=0)
    before = env.spiel_state().serialize()
    session = OpenSpielMCTSOpponent(max_simulations=20).open(game)

    action, = session.act([env], [Random(0)])

    assert action in step.legal_actions[0]
    assert env.spiel_state().serialize() == before
    assert session.is_perfect is False
    assert session.game_value() is None
    session.close()


@pytest.mark.parametrize("name", ["kuhn_poker", "phantom_tic_tac_toe"])
def test_games_with_hidden_information_are_rejected(name) -> None:
    with pytest.raises(IncompatibleEvaluatorError, match="perfect-information"):
        OpenSpielMCTSOpponent().open(load_game(name))


def test_non_openspiel_games_are_rejected() -> None:
    with pytest.raises(IncompatibleEvaluatorError, match="OpenSpiel game"):
        OpenSpielMCTSOpponent().open(load_game("rock_paper_scissors"))


@pytest.mark.parametrize("settings", [
    {"max_simulations": 0}, {"max_simulations": True}, {"rollouts": 0}, {"rollouts": 1.5},
    {"uct_c": 0.0}, {"uct_c": float("inf")}, {"uct_c": float("nan")}, {"solve": "yes"},
])
def test_invalid_settings_are_rejected(settings) -> None:
    with pytest.raises((TypeError, ValueError)):
        OpenSpielMCTSOpponent(**settings)


def test_open_does_not_traverse_game_tree_or_warn(monkeypatch) -> None:
    game = load_game("openspiel.tic_tac_toe")

    def unexpected_tree():
        pytest.fail("opening MCTS must not traverse the game tree")

    monkeypatch.setattr(game, "new_tree", unexpected_tree)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        OpenSpielMCTSOpponent().open(game).close()
