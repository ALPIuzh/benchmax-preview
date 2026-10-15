"""The configuration rules documented in ``benchmax/conf/README.md``.

Each test names the rule it pins, so a rule cannot be relaxed silently.
"""

from importlib.resources import files

import pytest
from omegaconf import OmegaConf

from benchmax import load_game
from benchmax.config import FixedParameterError, _resolve, list_names


def test_inline_build_applies_overrides_and_preserves_fixed_parameters() -> None:
    from benchmax.registry import _GAMES

    config = _resolve("game", "rock_paper_scissors")
    game = _GAMES.build(config, max_episode_steps=2)
    assert game.max_episode_steps == 2
    assert config.max_episode_steps == 1

    with pytest.raises(FixedParameterError):
        _GAMES.build(config, payoffs=[[0.0]])


def test_resolved_configuration_cannot_be_changed_through_a_returned_dictionary() -> None:
    from benchmax.registry import _GAMES

    resolved = _GAMES.resolve("openspiel.kuhn_poker", params={"players": 2})
    exposed = resolved.configuration
    exposed["params"]["players"] = 3

    assert resolved.configuration["params"]["players"] == 2


def test_loaded_game_origin_cannot_be_changed_through_returned_overrides() -> None:
    from benchmax.execution.jobs.submission import game_entry

    game = load_game("openspiel.kuhn_poker", params={"players": 2})
    expected = game_entry(game)
    game._benchmax_origin.overrides["params"]["players"] = 3

    assert game_entry(game) == expected


# --------------------------------------------------------------------------
# Raw provider entries
# --------------------------------------------------------------------------


def test_a_provider_entry_is_the_provider_game_with_its_own_defaults() -> None:
    game = load_game("openspiel.dark_hex")

    assert game.provenance == "openspiel"
    assert game.spiel_name == "dark_hex"
    assert game.params == {}
    assert game.spiel_game.get_parameters()["board_size"] == 3


def test_a_provider_entry_needs_no_configuration_file() -> None:
    entry = files("benchmax.conf").joinpath("game/openspiel/leduc_poker.yaml")
    assert not entry.is_file()

    game = load_game("openspiel.leduc_poker")

    assert game.spiel_name == "leduc_poker"
    assert game.evaluator is None


def test_a_provider_entry_file_adds_to_the_provider_rather_than_replacing_it() -> None:
    config = _resolve("game", "openspiel.kuhn_poker")

    # identity comes from the path; the file only contributes BenchMAX's part
    assert config["_target_"].endswith("OpenSpielGame")
    assert config["spiel_name"] == "kuhn_poker"
    assert config["evaluator"] == "nash_conv"


def test_no_shipped_provider_entry_sets_params() -> None:
    directory = files("benchmax.conf").joinpath("game/openspiel")

    for entry in directory.iterdir():
        if not entry.name.endswith(".yaml"):
            continue

        with entry.open(encoding="utf-8") as stream:
            config = OmegaConf.load(stream)

        assert "_params_" not in config, entry.name


def test_the_loader_refuses_a_provider_entry_that_sets_params(monkeypatch: pytest.MonkeyPatch) -> None:
    # the rule is enforced when loading, not only by inspecting shipped files.
    # The provider's own defaults are parameters, so what is refused is what
    # the configuration file adds to them.
    import benchmax.config as config_module

    monkeypatch.setattr(config_module, "_read", lambda entry, label: OmegaConf.create({"_params_": {"board_size": 4}}))

    with pytest.raises(ValueError, match="may not set '_params_'"):
        _resolve("game", "openspiel.dark_hex")


def test_a_named_configuration_is_not_reachable_through_a_provider() -> None:
    with pytest.raises(ValueError, match="OpenSpiel has no game named"):
        load_game("openspiel.dark_hex_3x3")


# --------------------------------------------------------------------------
# Aliases
# --------------------------------------------------------------------------


def test_an_alias_is_the_provider_entry_under_a_friendlier_name() -> None:
    alias = load_game("dark_hex_3x3")
    raw = load_game("openspiel.dark_hex")

    assert alias.spiel_name == raw.spiel_name
    assert alias.params == raw.params
    assert type(alias.evaluator) is type(raw.evaluator)
    assert alias.name == "dark_hex_3x3"
    assert raw.name == "dark_hex"


def test_an_alias_refuses_overrides_of_the_parameters_its_name_depends_on() -> None:
    with pytest.raises(FixedParameterError, match="board_size"):
        load_game("dark_hex_3x3", params={"board_size": 4})


def test_an_alias_cannot_be_pointed_at_another_game() -> None:
    # `_fixed_` covers the game's own arguments, and every OpenSpiel entry
    # inherits `spiel_name` from the provider's base configuration, so no name
    # BenchMAX ships can be made to load a different game.
    with pytest.raises(FixedParameterError, match="spiel_name"):
        load_game("dark_hex_3x3", spiel_name="kuhn_poker")


def test_frozen_names_accumulate_as_configurations_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmax.config as config_module
    from benchmax.config import _resolve as resolve

    monkeypatch.setattr(
        config_module,
        "_aliases",
        lambda group: OmegaConf.create(
            {"derived": {"_extends_": "openspiel.dark_hex", "_fixed_params_": ["board_size"]}}
        ),
    )

    config = resolve("game", "derived")

    # what the entry freezes is added to what it inherits, not swapped for it
    assert list(config["_fixed_"]) == ["spiel_name"]
    assert list(config["_fixed_params_"]) == ["board_size"]


def test_an_override_the_game_does_not_take_is_refused_by_the_game() -> None:
    # Parameters reach OpenSpiel through `params` and nowhere else, so a
    # parameter spelled as a keyword argument is not a way around `_fixed_`:
    # it is simply not an argument OpenSpielGame has.
    with pytest.raises(Exception, match="unexpected keyword argument"):
        load_game("dark_hex_3x3", board_size=4)


def test_the_same_parameter_is_free_on_the_provider_entry() -> None:
    game = load_game("openspiel.dark_hex", params={"board_size": 4})

    assert len(game.action_spaces[0]) == 16


def test_parameters_a_name_does_not_depend_on_stay_overridable() -> None:
    game = load_game("dark_hex_3x3", params={"obstype": "reveal-numturns"})

    assert game.spiel_game.get_parameters()["obstype"] == "reveal-numturns"


# --------------------------------------------------------------------------
# Named configurations
# --------------------------------------------------------------------------


def test_a_named_configuration_changes_the_provider_defaults() -> None:
    variant = load_game("abrupt_dark_hex_3x3")
    raw = load_game("openspiel.dark_hex")

    assert variant.params["gameversion"] == "adh"
    assert raw.spiel_game.get_parameters()["gameversion"] == "cdh"
    assert variant.name == "abrupt_dark_hex_3x3"


def test_source_resolves_without_a_configuration_for_the_base() -> None:
    entry = files("benchmax.conf").joinpath("game/openspiel/liars_dice.yaml")
    assert not entry.is_file()

    game = load_game("liars_dice_2d5f")

    assert game.spiel_name == "liars_dice"
    assert game.params["numdice"] == 2


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------


def test_extends_requires_the_entry_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmax.config as config_module

    monkeypatch.setattr(
        config_module, "_aliases", lambda group: OmegaConf.create({"broken": {"_extends_": "does.not.exist"}})
    )

    with pytest.raises(ValueError):
        _resolve("game", "broken")


def test_source_and_extends_layer_library_then_benchmax_then_the_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmax.config as config_module

    monkeypatch.setattr(
        config_module,
        "_aliases",
        lambda group: OmegaConf.create(
            {
                "shared": {"max_episode_steps": 7, "description": "shared"},
                "layered": {"_source_": "openspiel.dark_hex", "_extends_": "shared", "description": "mine"},
            }
        ),
    )

    config = _resolve("game", "layered")

    assert config["spiel_name"] == "dark_hex"  # from the provider
    assert config["max_episode_steps"] == 7  # from the extended entry
    assert config["description"] == "mine"  # the entry itself wins


def test_a_configuration_that_refers_to_itself_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmax.config as config_module

    monkeypatch.setattr(config_module, "_aliases", lambda group: OmegaConf.create({"loop": {"_extends_": "loop"}}))

    with pytest.raises(ValueError, match="refers to itself"):
        _resolve("game", "loop")


def test_use_replaces_a_key_with_an_entry_of_another_group() -> None:
    from benchmax.integrations.exp_a_spiel import ExpASpielEvaluator

    game = load_game("abrupt_dark_hex_3x3")

    assert isinstance(game.evaluator, ExpASpielEvaluator)
    assert game.evaluator._traverser_name == "AbruptDhEvaluator"


@pytest.mark.parametrize("selector", ["name", "_use_"])
def test_selecting_an_evaluator_drops_the_previous_backends_arguments(selector):
    from benchmax.evaluation import ApproximateExploitabilityEvaluator

    game = load_game("dark_hex_3x3", evaluator={
        selector: "approximate_exploitability", "seed": 17,
        "params": {"model": {"num_cells": [32]}, "experiment": {"lr": 0.001}},
    })
    assert isinstance(game.evaluator, ApproximateExploitabilityEvaluator)
    assert game.evaluator.seed == 17
    assert game.evaluator.params["model"]["num_cells"] == [32]
    assert game.evaluator.params["experiment"]["lr"] == 0.001
    assert game.evaluator.params["experiment"]["on_policy_collected_frames_per_batch"] == 1024


def test_params_never_reach_benchmax_keys_and_the_reverse() -> None:
    game = load_game("dark_hex_3x3")

    assert "evaluator" not in game.params
    assert "description" not in game.params
    assert "max_episode_steps" not in game.params


def test_provider_parameters_stay_nested_when_names_collide(monkeypatch) -> None:
    from benchmax import config
    from benchmax.registry import _GAMES

    original = config._resolve

    def resolve(group, name, *args, **kwargs):
        document = original(group, name, *args, **kwargs)

        if group == "game" and name == "openspiel.kuhn_poker":
            document["_params_"] = {"description": "provider value"}

        return document

    monkeypatch.setattr(config, "_resolve", resolve)
    resolved = _GAMES.resolve("openspiel.kuhn_poker", description="BenchMAX value").configuration
    assert resolved["description"] == "BenchMAX value"
    assert resolved["params"]["description"] == "provider value"


# --------------------------------------------------------------------------
# Overrides and listing
# --------------------------------------------------------------------------


def test_overrides_merge_rather_than_replace() -> None:
    game = load_game("abrupt_dark_hex_3x3", params={"obstype": "reveal-numturns"})

    assert game.params["obstype"] == "reveal-numturns"
    assert game.params["gameversion"] == "adh"


def test_every_group_lists_without_underscore_entries() -> None:
    for group in ("game", "algorithm", "evaluator"):
        assert not any(name.startswith("_") for name in list_names(group))


# --------------------------------------------------------------------------
# Algorithms follow the same rules as games
# --------------------------------------------------------------------------


def test_a_bare_algorithm_name_is_an_alias_for_the_library_algorithm() -> None:
    pytest.importorskip("benchmarl")
    from benchmax.training.algorithm import load_training_function

    alias = load_training_function("mappo")
    raw = load_training_function("benchmarl.mappo")

    # BenchMAX ships no hyperparameters of its own: settings that suit an
    # algorithm on one game do not suit it on another, so tuned values belong
    # to a (game, algorithm) pair rather than to the algorithm.
    assert alias.experiment.lr == raw.experiment.lr
    assert list(alias.model.num_cells) == list(raw.model.num_cells)
    assert alias.algorithm.entropy_coef == raw.algorithm.entropy_coef


def test_no_algorithm_configuration_ships_hyperparameters() -> None:
    directory = files("benchmax.conf").joinpath("algorithm")

    for entry in directory.iterdir():
        if not entry.name.endswith(".yaml"):
            continue

        text = entry.read_text(encoding="utf-8")
        assert "_params_" not in text, entry.name


def test_provider_algorithms_need_no_configuration_file() -> None:
    directory = files("benchmax.conf").joinpath("algorithm")
    assert not directory.joinpath("benchmarl").is_dir()

    pytest.importorskip("benchmarl")
    from benchmax.integrations.benchmarl.provider import algorithm_config_class
    from benchmax.training.algorithm import load_training_function

    training_function = load_training_function("benchmarl.qmix")

    assert type(training_function.algorithm) is algorithm_config_class("qmix")


def test_an_unknown_provider_algorithm_is_rejected_by_the_provider() -> None:
    pytest.importorskip("benchmarl")
    from benchmax.training.algorithm import load_training_function

    with pytest.raises(ValueError, match="unknown benchmarl algorithm"):
        load_training_function("benchmarl.not_an_algorithm")
