from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.training_helpers import tiny_training_params

pytest.importorskip("benchmarl")

import torch

from benchmax import Game, load_game
from benchmax.integrations.benchmarl import training as benchmarl_training
from benchmax.policy import PolicyProfile
from benchmax.training import list_available_algorithms, train_ippo, train_mappo
from benchmax.training.algorithm import load_training_function


def test_self_play_checkpoints_freeze_both_players_without_changing_training():
    game = load_game("kuhn_poker")
    trainer = load_training_function(
        "ippo", total_frames=24, device="cpu", verbose=False, params=tiny_training_params(),
    )
    expected = trainer(game, seed=7)._batched_policy_profile._state_dicts()
    snapshots, original_states = [], []

    def capture(policy, frames):
        snapshots.append((policy, frames))
        original_states.append({
            player: {name: value.clone() for name, value in state.items() if isinstance(value, torch.Tensor)}
            for player, state in policy._batched_policy_profile._state_dicts().items()
        })

    actual = trainer._train(game, seed=7, checkpoint_frames=(8, 24), on_checkpoint=capture)
    assert [frames for _, frames in snapshots] == [8, 24]
    for (snapshot, _), original in zip(snapshots, original_states, strict=True):
        state = snapshot._batched_policy_profile._state_dicts()
        assert set(state) == set(game.players)
        for player in game.players:
            for name, value in original[player].items():
                torch.testing.assert_close(state[player][name], value, rtol=0, atol=0)
    final = actual._batched_policy_profile._state_dicts()
    last_snapshot = snapshots[-1][0]._batched_policy_profile._state_dicts()
    for player in game.players:
        for name, value in expected[player].items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(final[player][name], value, rtol=0, atol=0)
                torch.testing.assert_close(last_snapshot[player][name], value, rtol=0, atol=0)


def test_training_seeds_the_rollout_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmax.seeding import derive_seed

    experiment = SimpleNamespace(collector=SimpleNamespace(set_seed=Mock()), close=Mock())
    experiment.run = experiment.close
    create_experiment = Mock(return_value=experiment)
    monkeypatch.setattr(benchmarl_training, "_create_experiment", create_experiment)
    monkeypatch.setattr(benchmarl_training, "_policy_from_experiment", lambda experiment, game: (experiment, game))
    game = load_game("rock_paper_scissors")

    training_function = load_training_function("mappo", total_frames=1, device="cpu", verbose=False)
    _, trained_game = training_function(game, seed=17)

    create_experiment.assert_called_once()
    assert create_experiment.call_args.kwargs["seed"] == derive_seed(17, "training", "experiment")
    experiment.collector.set_seed.assert_called_once_with(derive_seed(17, "training", "collector"))
    experiment.close.assert_called_once_with()
    assert trained_game is game


@pytest.mark.parametrize("failure_at", ["seed", "run"])
def test_training_closes_the_experiment_once_on_failure(monkeypatch: pytest.MonkeyPatch, failure_at: str) -> None:
    set_seed = Mock(side_effect=RuntimeError("seed failed") if failure_at == "seed" else None)
    experiment = SimpleNamespace(collector=SimpleNamespace(set_seed=set_seed), close=Mock())

    def run():
        experiment.close()
        raise RuntimeError("run failed")

    experiment.run = run
    monkeypatch.setattr(benchmarl_training, "_create_experiment", Mock(return_value=experiment))
    monkeypatch.setattr(benchmarl_training, "_policy_from_experiment", lambda experiment, game: (experiment, game))
    training_function = load_training_function("mappo", total_frames=1, device="cpu", verbose=False)

    with pytest.raises(RuntimeError, match=f"{failure_at} failed"):
        training_function(Mock(spec=Game), seed=17)

    experiment.close.assert_called_once_with()


@pytest.mark.parametrize("training_function", [train_ippo, train_mappo])
@pytest.mark.parametrize(
    "name,settings",
    [("rock_paper_scissors", {}),
     ("openspiel.connect_four", {"params": {"rows": 2, "columns": 3, "x_in_row": 3}, "evaluator": "nash_conv"})],
)
def test_training_returns_an_evaluable_policy_profile(training_function: Callable[..., PolicyProfile], name, settings):
    game = load_game(name, **settings)
    game.name = f"configured_{game.name}"
    game.action_spaces = {player: tuple(reversed(actions)) for player, actions in game.action_spaces.items()}

    profile = training_function(
        game, total_frames=8, seed=7, device="cpu", verbose=False, params=tiny_training_params()
    )

    results = game.evaluate(profile)

    assert set(results) == {"expected_returns", "player_improvements", "nash_gap", "nash_conv", "exploitability"}
    assert 0 <= results["exploitability"] <= 2
    assert sum(results["expected_returns"].values()) == pytest.approx(0.0)


def test_cpu_training_is_reproducible_within_the_numerical_tolerance() -> None:
    from benchmax.seeding import SEED_SCHEME, forbid_entropy

    game = load_game("rock_paper_scissors")
    settings = {"total_frames": 8, "seed": 7, "device": "cpu", "verbose": False, "params": tiny_training_params()}

    with forbid_entropy():
        first = train_mappo(game, **settings)
        second = train_mappo(game, **settings)

    assert first._portable_training == second._portable_training
    assert first._portable_training.seed == 7
    assert first._portable_training.seed_scheme == SEED_SCHEME
    assert first._portable_training.digest == second._portable_training.digest
    first_state = first._batched_policy_profile._state_dicts()
    second_state = second._batched_policy_profile._state_dicts()

    for player in game.players:
        for name, value in first_state[player].items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, second_state[player][name], rtol=0.0, atol=1e-5)

    first_metrics = game.evaluate(first)
    second_metrics = game.evaluate(second)
    assert first_metrics["nash_conv"] == pytest.approx(second_metrics["nash_conv"], abs=1e-5)
    assert first_metrics["exploitability"] == pytest.approx(second_metrics["exploitability"], abs=1e-5)


@pytest.mark.parametrize("device,threads,expected", [("cuda", 1, 1), ("cuda:0", 2, 2), ("cuda", None, 3), ("cpu", 1, 3)])
def test_training_thread_setting_is_scoped_even_on_failure(device, threads, expected):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(3)
        with pytest.raises(RuntimeError, match="training failed"), benchmarl_training._training_threads(device, threads):
            assert torch.get_num_threads() == expected
            raise RuntimeError("training failed")
        assert torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("threads", [0, -1, True, 1.5])
def test_invalid_training_thread_settings_are_rejected(threads):
    with pytest.raises(ValueError, match="torch_threads"):
        load_training_function("ippo", torch_threads=threads)


@pytest.mark.parametrize("device,requested,initial,expected", [
    ("cuda", "tf32", "ieee", "tf32"),
    ("cuda:0", "ieee", "tf32", "ieee"),
    ("cuda", "tf32", "none", "tf32"),
    ("cuda", None, "tf32", "tf32"),
    ("cpu", "tf32", "ieee", "ieee"),
])
def test_matmul_precision_is_scoped_even_on_failure(device, requested, initial, expected):
    previous = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = initial
        with (
            pytest.raises(RuntimeError, match="training failed"),
            benchmarl_training._training_matmul_precision(device, requested),
        ):
            assert torch.backends.cuda.matmul.fp32_precision == expected
            raise RuntimeError("training failed")
        assert torch.backends.cuda.matmul.fp32_precision == initial
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous


@pytest.mark.parametrize("precision", [True, 32, "high", "bf16"])
def test_invalid_matmul_precision_is_rejected(precision):
    with pytest.raises(ValueError, match="matmul_precision"):
        load_training_function("ippo", matmul_precision=precision)


@pytest.mark.parametrize("name", [name for name in list_available_algorithms() if name.startswith("presets.")])
def test_presets_build_from_portable_specs_with_partial_overrides(name):
    from benchmax.execution.specs import TrainingSpec
    from benchmax.integrations.benchmarl.training import load_benchmarl_training_function

    _, game = name.removeprefix("presets.").split("_", 1)
    spec = TrainingSpec.create(
        game=game, algorithm=name, seed=7, device="cpu", total_frames=32,
        algorithm_overrides={"params": {"experiment": {"lr": 0.000123}}},
    )
    restored = TrainingSpec.from_document(spec.to_document())
    assert restored.to_document() == spec.to_document()
    function = load_benchmarl_training_function(restored)
    assert function.total_frames == 32
    assert function.experiment.lr == 0.000123
    assert function.model.num_cells == restored.algorithm.resolved["params"]["model"]["num_cells"]
