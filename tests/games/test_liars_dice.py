from benchmax import load_game
from benchmax.integrations.exp_a_spiel import ExpASpielEvaluator


def test_reset_exposes_spaces_and_legal_bids_and_is_seeded() -> None:
    game = load_game("liars_dice_2d5f")
    first = game.new_env().reset(seed=17)
    second = game.new_env().reset(seed=17)

    assert game.observation_spaces == {0: (33,), 1: (33,)}
    assert game.action_spaces == {0: tuple(range(21)), 1: tuple(range(21))}
    assert len(first.observations[0]) == 33
    assert first.legal_actions == {0: tuple(range(20))}
    assert first.players_to_act == (0,)
    assert first.observations.keys() == {0}
    assert first.observations == second.observations
    assert first.rewards == {0: 0.0, 1: 0.0}
    assert first.terminated is False
    assert first.truncated is False
    assert first.info == {}


def test_after_a_bid_liar_and_only_higher_bids_are_legal() -> None:
    env = load_game("liars_dice_2d5f").new_env()
    env.reset(seed=3)

    time_step = env.step({0: 6})

    assert time_step.players_to_act == (1,)
    assert time_step.legal_actions == {1: tuple(range(7, 21))}
    assert time_step.rewards == {0: 0.0, 1: 0.0}


def test_calling_liar_terminates_with_unit_zero_sum_rewards() -> None:
    env = load_game("liars_dice_2d5f").new_env()
    env.reset(seed=3)
    env.step({0: 0})

    time_step = env.step({1: 20})

    assert time_step.terminated is True
    assert time_step.truncated is False
    assert time_step.observations == {}
    assert time_step.players_to_act == ()
    assert time_step.legal_actions == {}
    assert sum(time_step.rewards.values()) == 0.0
    assert {abs(reward) for reward in time_step.rewards.values()} == {1.0}


def test_registry_loading_attaches_exact_evaluation() -> None:
    game = load_game("liars_dice_2d5f")

    assert game.name == "liars_dice_2d5f"

    assert isinstance(game.evaluator, ExpASpielEvaluator)
    assert game.evaluator._traverser_name == "Ld2d5fEvaluator"
