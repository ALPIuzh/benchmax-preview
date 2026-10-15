"""Native binding tests run in clean processes to preserve OpenMP isolation."""

from tests.process_helpers import assert_python_succeeds


def test_corner_dark_hex_policy_inputs_and_evaluation() -> None:
    assert_python_succeeds(
        """
        import numpy as np
        import pytest

        from benchmax.integrations.exp_a_spiel import _exp_a_spiel

        def policy_inputs_and_uniform_strategy(evaluator, player):
            rows = evaluator.num_information_states(player)
            split = rows // 2
            batches = (
                evaluator.policy_input_batch(player, 0, split),
                evaluator.policy_input_batch(player, split, rows),
            )
            observations = np.concatenate([batch[0] for batch in batches])
            legal_action_mask = np.concatenate([batch[1] for batch in batches])
            strategy = np.ascontiguousarray(
                legal_action_mask / legal_action_mask.sum(axis=1, keepdims=True)
            )
            return observations, legal_action_mask, strategy

        evaluator = _exp_a_spiel.CornerDhEvaluator()
        observations_0, mask_0, strategy_0 = (
            policy_inputs_and_uniform_strategy(evaluator, 0)
        )
        observations_1, mask_1, strategy_1 = (
            policy_inputs_and_uniform_strategy(evaluator, 1)
        )

        for player, observations, mask in (
            (0, observations_0, mask_0),
            (1, observations_1, mask_1),
        ):
            rows = evaluator.num_information_states(player)
            assert observations.dtype == np.float64
            assert observations.shape == (rows, 162)
            assert observations.flags.c_contiguous
            assert mask.dtype == np.bool_
            assert mask.shape == (rows, 9)
            assert mask.flags.c_contiguous

        assert evaluator.evaluate(strategy_0, strategy_1) == pytest.approx(
            (0.371428571, 1.161904762, 0.628571429)
        )

        with pytest.raises(TypeError, match="dtype float64"):
            evaluator.evaluate(strategy_0.astype(np.float32), strategy_1)
        with pytest.raises(TypeError, match="C-contiguous"):
            evaluator.evaluate(strategy_0[:, ::-1], strategy_1)
        invalid_strategy_0 = strategy_0.copy()
        invalid_strategy_0[0] = 0.0
        with pytest.raises(ValueError, match="not a valid strategy"):
            evaluator.evaluate(invalid_strategy_0, strategy_1)
        """
    )


def test_compact_cfr_operator_trains_corner_dark_hex() -> None:
    assert_python_succeeds(
        """
        from types import SimpleNamespace

        from benchmax import load_game
        from benchmax.integrations.exp_a_spiel import _exp_a_spiel
        from benchmax.integrations.exp_a_spiel.payoff import _ExpASpielPayoffOperator
        from benchmax.training.cfr import train_cfr

        game = load_game("dark_hex_3x3")
        evaluator = _exp_a_spiel.CornerDhEvaluator()
        operator = _ExpASpielPayoffOperator(game, evaluator)
        game.evaluator = SimpleNamespace(payoff_operator=lambda _: operator)
        policy = train_cfr(game, 8, plus=True)
        strategies = policy._exp_a_spiel_strategies(evaluator)

        assert strategies is not None
        _, improvement_0, improvement_1 = evaluator.evaluate(*strategies)
        assert (improvement_0 + improvement_1) / 2 < 0.03
        """
    )


def test_production_evaluator_information_state_sizes() -> None:
    assert_python_succeeds(
        """
        from benchmax.integrations.exp_a_spiel import _exp_a_spiel

        expected = {"PtttEvaluator": 108, "AbruptPtttEvaluator": 108, "DhEvaluator": 162, "AbruptDhEvaluator": 162}
        for name, size in expected.items():
            assert getattr(_exp_a_spiel, name).information_state_size == size, name
        """
    )


def test_native_evaluators_publish_their_exact_openspiel_contracts() -> None:
    assert_python_succeeds(
        """
        from benchmax.integrations.exp_a_spiel import _exp_a_spiel

        assert _exp_a_spiel.PtttEvaluator.compatible_game == {
            "spiel_name": "phantom_ttt",
            "parameters": {
                "gameversion": "classical",
                "obstype": "reveal-nothing",
            },
        }
        assert _exp_a_spiel.AbruptPtttEvaluator.compatible_game == {
            "spiel_name": "phantom_ttt",
            "parameters": {
                "gameversion": "abrupt",
                "obstype": "reveal-nothing",
            },
        }
        assert _exp_a_spiel.DhEvaluator.compatible_game == {
            "spiel_name": "dark_hex",
            "parameters": {
                "board_size": 3,
                "num_rows": 3,
                "num_cols": 3,
                "gameversion": "cdh",
                "obstype": "reveal-nothing",
            },
        }
        assert _exp_a_spiel.AbruptDhEvaluator.compatible_game == {
            "spiel_name": "dark_hex",
            "parameters": {
                "board_size": 3,
                "num_rows": 3,
                "num_cols": 3,
                "gameversion": "adh",
                "obstype": "reveal-nothing",
            },
        }
        assert _exp_a_spiel.CornerDhEvaluator.compatible_game == {
            "spiel_name": None,
            "parameters": {},
            "reason": (
                "the native corner Dark Hex traverser starts with two occupied "
                "corners, which stock OpenSpiel dark_hex cannot represent"
            ),
        }
        """
    )


def test_indexed_policy_artifact_round_trip_on_small_native_layout() -> None:
    # Exercise the real native lookup and treeplex on the bounded corner fixture;
    # constructing a full production phantom-game layout is unsuitable for CI.
    assert_python_succeeds(
        """
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import numpy as np
        import pytest

        from benchmax import Reference, load_game
        from benchmax.execution.artifacts.policies import ArtifactError, validate_policy_artifact
        from benchmax.integrations.exp_a_spiel import _exp_a_spiel
        from benchmax.integrations.exp_a_spiel.evaluation import ExpASpielEvaluator
        from benchmax.integrations.exp_a_spiel.payoff import _ExpASpielPayoffOperator, _ExpASpielPolicy
        from benchmax.training.cfr import train_cfr

        native = _exp_a_spiel.CornerDhEvaluator()

        class DhEvaluator:
            def __getattr__(self, name):
                return getattr(native, name)

        traverser = DhEvaluator()
        game = load_game("dark_hex_3x3")
        operator = _ExpASpielPayoffOperator(game, traverser)
        with patch.object(game.evaluator, "payoff_operator", return_value=operator):
            profile = train_cfr(game, 8, plus=True)

        profile._artifact_annotations = {"producer": {"name": "native-test"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(ExpASpielEvaluator, "_get_traverser", return_value=traverser):
                saved = profile.save(root / "policy", game=game)
                with patch.object(np, "load", side_effect=AssertionError("inspection loads no arrays")):
                    validate_policy_artifact(saved._artifact_metadata())
                    copied = saved.save(root / "copy")
                restored = copied.result()
                assert isinstance(restored, _ExpASpielPolicy)
                assert restored._artifact_annotations == profile._artifact_annotations
                for player in game.players:
                    observations, masks = native.policy_input_batch(player, 0, native.num_information_states(player))
                    np.testing.assert_array_equal(
                        restored(player, observations, game.action_spaces[player], masks),
                        profile(player, observations, game.action_spaces[player], masks),
                    )
                manifest = json.loads(saved.path.read_text())
                assert manifest["format"] == "benchmax.exp-a-spiel-indexed-policy.npz"
                manifest["layouts"]["0"] = "0" * 64
                saved.path.write_text(json.dumps(manifest))
                with pytest.raises(ArtifactError, match="layout has changed"):
                    Reference.from_path(saved.path).result()
        """
    )
