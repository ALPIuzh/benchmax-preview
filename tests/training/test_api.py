from importlib.util import find_spec
from unittest.mock import Mock

import pytest

from benchmax import PolicyProfile, load_algorithm, load_game, training
from benchmax.training import list_available_algorithms
from tests.process_helpers import assert_python_succeeds


def test_bundled_algorithms_and_presets_are_listed() -> None:
    algorithms = list_available_algorithms()

    assert [name for name in algorithms if not name.startswith("presets.")] == [
        "cfr", "cfr_plus", "ippo", "iql", "mappo",
    ]
    assert "presets.ippo_kuhn_poker" in algorithms
    assert "presets.iql_gridworld_3x3" in algorithms
    assert not any(name.startswith("_") for name in algorithms)


def test_an_unknown_attribute_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        _ = training.nonexistent


@pytest.mark.parametrize("alias,algorithm", [("train_ippo", "ippo"), ("train_mappo", "mappo"), ("train_iql", "iql")])
def test_training_aliases_select_the_algorithm(monkeypatch, alias, algorithm) -> None:
    from benchmax import Algorithm, Game

    game = Mock(spec=Game, evaluator=object())
    expected = PolicyProfile(lambda *args: None)
    calls = []

    def run(self, received, *, seed):
        calls.append((self.name, self.configuration))
        return expected

    monkeypatch.setattr(Algorithm, "__call__", run)
    assert getattr(training, alias)(game, seed=7, device="cpu", total_frames=8) is expected
    assert calls[0][0] == algorithm
    assert calls[0][1]["total_frames"] == 8


def test_training_rejects_a_game_of_the_wrong_type() -> None:
    pytest.importorskip("benchmarl")
    from benchmax.training import train

    with pytest.raises(TypeError, match="game must be a BenchMAX Game"):
        train("kuhn_poker")  # type: ignore[arg-type]


@pytest.mark.parametrize("algorithm", ["cfr", "cfr_plus"])
def test_cfr_trains_an_evaluable_kuhn_policy(algorithm: str) -> None:
    from benchmax.training import train

    game = load_game("kuhn_poker")
    result = game.evaluate(train(game, algorithm=algorithm, iterations=1000))

    assert result["exploitability"] < 0.01


def test_cfr_handles_simultaneous_decisions() -> None:
    from benchmax.training import train

    game = load_game("rock_paper_scissors")
    result = game.evaluate(train(game, algorithm="cfr", iterations=1))

    assert result["exploitability"] == pytest.approx(0.0)


def test_cfr_rejects_distinct_histories_its_observation_policy_cannot_represent() -> None:
    from benchmax.training import train

    game = load_game("openspiel.connect_four", params={"rows": 2, "columns": 3, "x_in_row": 3}, evaluator="nash_conv")

    with pytest.raises(ValueError, match="observations and legal actions must distinguish information sets"):
        train(game, algorithm="cfr", iterations=1)


def test_cfr_rejects_information_states_that_forget_the_players_own_action() -> None:
    from benchmax.core import GameTreeNode
    from benchmax.training import train

    class ForgetfulNode(GameTreeNode):
        def __init__(self, turn=0):
            self.turn = turn

        def returns(self):
            return {0: 0, 1: 0} if self.turn == 2 else None

        def chance_outcomes(self):
            return ()

        def players_to_act(self):
            return (0,)

        def observation(self, player):
            return (0.0,)

        def legal_actions(self, player):
            return (0, 1, 2)

        def chance_child(self, action):
            raise NotImplementedError

        def child(self, actions):
            return ForgetfulNode(self.turn + 1)

    game = load_game("rock_paper_scissors")
    game.new_tree = ForgetfulNode

    with pytest.raises(ValueError, match="imperfect recall"):
        train(game, algorithm="cfr", iterations=1)


def test_direct_training_accepts_a_configured_training_function() -> None:
    from benchmax import Game, PolicyProfile
    from benchmax.training import train

    game = Mock(spec=Game, evaluator=object())
    expected = PolicyProfile(lambda *args: None)
    calls: list[tuple[object, int]] = []

    def training_function(received_game, *, seed):
        calls.append((received_game, seed))
        return expected

    profile = train(game, training_function, seed=7)

    assert profile is expected
    assert calls == [(game, 7)]


def test_submitted_training_requires_a_registered_training_function() -> None:
    from benchmax import Game, PolicyProfile
    from benchmax.training import train

    def training_function(game, *, seed):
        del game, seed
        return PolicyProfile(lambda *args: None)

    with pytest.raises(ValueError, match="requires a registered algorithm"):
        train(Mock(spec=Game, evaluator=object()), training_function, seed=7, submit=True)


def test_training_function_must_return_a_policy_profile() -> None:
    from benchmax import Game
    from benchmax.training import train

    def training_function(game, *, seed):
        del game, seed
        return object()

    with pytest.raises(TypeError, match="must return a PolicyProfile"):
        train(Mock(spec=Game, evaluator=object()), training_function, seed=7)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"algorithm": ""}, "algorithm must be a non-empty string"),
        ({"seed": -1}, r"seed must lie in \[0, 2\*\*31\)"),
        ({"seed": 2**31}, r"seed must lie in \[0, 2\*\*31\)"),
        ({"device": ""}, "device must be a non-empty string"),
    ],
)
def test_training_validates_portable_fields_at_the_public_boundary(arguments, message) -> None:
    from benchmax import load_game
    from benchmax.training import train

    game = load_game("rock_paper_scissors")

    with pytest.raises((TypeError, ValueError), match=message):
        train(game, submit=True, **arguments)


@pytest.mark.skipif(find_spec("benchmarl") is not None, reason="checks the base-only error")
def test_deferred_training_is_available_without_local_training_dependencies() -> None:
    from benchmax import load_game
    from benchmax.execution import Job
    from benchmax.training import train

    game = load_game("rock_paper_scissors")

    job = train(game, seed=7, submit=True)

    assert type(job) is Job
    assert job._prepare().algorithm.resolved is not None

    with pytest.raises(ModuleNotFoundError, match=r"install benchmax\[training\]"):
        train(game, seed=7)


def test_deferred_training_does_not_import_the_training_runtime() -> None:
    assert_python_succeeds(
        """
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "benchmarl" or name.startswith("benchmarl."):
                raise AssertionError(f"unexpected training-runtime import: {name}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked

        from benchmax import load_algorithm, load_game
        from benchmax.training import train, list_available_algorithms

        assert "mappo" in list_available_algorithms()
        algorithm = load_algorithm("mappo", total_frames=8)
        job = train(load_game("rock_paper_scissors"), algorithm, seed=7, submit=True)
        assert job._prepare().algorithm.resolved["provider"] == "benchmarl"
        """,
        timeout=10,
    )


def test_loaded_configuration_is_detached_from_caller_mutations():
    params = {"experiment": {"lr": 1e-4}}
    algorithm = load_algorithm("mappo", params=params)
    params["experiment"]["lr"] = 0.5
    algorithm.configuration["params"]["experiment"]["lr"] = 0.9
    assert algorithm.configuration["params"]["experiment"]["lr"] == 1e-4


def test_reusing_an_algorithm_starts_with_fresh_learner_state(monkeypatch):
    from benchmax.registry import _ALGORITHMS

    observed = []

    def build(*args, **kwargs):
        updates = []

        def learn(game, *, seed):
            updates.append(seed)
            observed.append(tuple(updates))
            return PolicyProfile(lambda *args: None)

        return learn

    monkeypatch.setattr(_ALGORITHMS, "load", build)
    algorithm = load_algorithm("cfr")
    game = load_game("rock_paper_scissors")
    for seed in (7, 19):
        training.train(game, algorithm, seed=seed)

    assert observed == [(7,), (19,)]


def test_loaded_and_named_algorithms_produce_the_same_job():
    game = load_game("kuhn_poker")
    settings = {"total_frames": 24, "device": "cpu", "params": {"experiment": {"lr": 1e-4}}}
    named = training.train(game, "mappo", seed=7, submit=True, **settings)._prepare()
    loaded = training.train(game, load_algorithm("mappo", **settings), seed=7, submit=True)._prepare()
    assert named.to_document() == loaded.to_document()


@pytest.mark.parametrize("submit", [False, True])
def test_loaded_algorithm_rejects_conflicting_settings(submit):
    with pytest.raises(ValueError, match="bind settings with load_algorithm"):
        training.train(load_game("kuhn_poker"), load_algorithm("cfr"), seed=7, submit=submit, iterations=20)
