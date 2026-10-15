import pytest

from benchmax import list_available_games, load_game

BARE_NAMES = [
    "abrupt_dark_hex_3x3",
    "abrupt_phantom_tic_tac_toe",
    "connect_four",
    "dark_hex_3x3",
    "gridworld_3x3",
    "kuhn_poker",
    "liars_dice_2d5f",
    "phantom_tic_tac_toe",
    "rock_paper_scissors",
]

#: Raw provider entries that ship a configuration; also listed.
PROVIDER_ENTRIES = [
    "openspiel.connect_four",
    "openspiel.dark_hex",
    "openspiel.kuhn_poker",
    "openspiel.phantom_ttt",
]

def test_games_declare_which_library_implements_them() -> None:
    assert load_game("kuhn_poker").provenance == "openspiel"
    assert load_game("rock_paper_scissors").provenance == "benchmax"


def test_overrides_replace_configuration_keys() -> None:
    game = load_game("kuhn_poker", evaluator=None)

    assert game.evaluator is None
    assert load_game("kuhn_poker").evaluator is not None


def test_list_available_games_does_not_instantiate_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        pytest.fail("list_available_games() instantiated a game")

    monkeypatch.setattr("hydra.utils.instantiate", fail_if_called)

    assert list_available_games() == sorted(BARE_NAMES + PROVIDER_ENTRIES)


def test_rejects_unknown_game() -> None:
    with pytest.raises(ValueError, match="unknown game 'chess'"):
        load_game("chess")


def test_rejects_non_string_name() -> None:
    with pytest.raises(TypeError, match="game name must be a string"):
        load_game(1)  # type: ignore[arg-type]


def test_evaluators_are_referenced_by_name_not_import_path() -> None:
    from benchmax.integrations.exp_a_spiel import ExpASpielEvaluator
    from benchmax.integrations.openspiel.evaluation import OpenSpielNashConvEvaluator

    assert isinstance(load_game("kuhn_poker").evaluator, OpenSpielNashConvEvaluator)
    assert isinstance(load_game("dark_hex_3x3").evaluator, ExpASpielEvaluator)

    swapped = load_game("dark_hex_3x3", evaluator="nash_conv")
    assert isinstance(swapped.evaluator, OpenSpielNashConvEvaluator)


def test_a_referenced_evaluator_can_be_overridden_in_place() -> None:
    game = load_game("dark_hex_3x3", evaluator={"_use_": "exp_a_spiel", "traverser": "AbruptDhEvaluator"})

    assert game.evaluator._traverser_name == "AbruptDhEvaluator"


def test_rejects_unknown_evaluator_reference() -> None:
    with pytest.raises(ValueError, match="unknown evaluator 'nope'"):
        load_game("kuhn_poker", evaluator="nope")


def test_connect_four_board_parameters_are_overridable():
    game = load_game("connect_four", params={"rows": 4, "columns": 5})
    params = game.spiel_game.get_parameters()
    assert (params["rows"], params["columns"], params["x_in_row"]) == (4, 5, 4)
    variant = load_game("connect_four", params={"rows": 3, "columns": 4, "x_in_row": 3}, evaluator=None)
    assert variant.spiel_game.get_parameters()["x_in_row"] == 3
