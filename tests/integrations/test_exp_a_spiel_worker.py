from importlib.util import find_spec

import pytest

from tests.process_helpers import assert_python_succeeds

requires_torch = pytest.mark.skipif(find_spec("torch") is None, reason="requires the training extra")


@requires_torch
@pytest.mark.parametrize("torch_first", (True, False), ids=("torch-first", "evaluation-first"))
def test_torch_parent_and_native_evaluation_worker_remain_isolated(torch_first: bool) -> None:
    program = (
        """
        import sys
        import numpy as np

        from benchmax import PolicyProfile
        from benchmax.execution.worker import PolicyProfileOracle, close_default_evaluation_runtime
        from benchmax.execution.worker import LocalWorkerRuntime, get_evaluation_runtime
        from benchmax.integrations.exp_a_spiel.kernel import evaluate_exact_kernel

        torch_first = __TORCH_FIRST__
        if torch_first:
            import torch

        native_module = "benchmax.integrations.exp_a_spiel._exp_a_spiel"
        assert native_module not in sys.modules

        def uniform(player, observations, action_space, legal_action_mask):
            del player, observations, action_space
            probabilities = legal_action_mask.astype(np.float64)
            return probabilities / probabilities.sum(axis=1, keepdims=True)

        action_spaces = {0: tuple(range(9)), 1: tuple(range(9))}
        policy = PolicyProfileOracle(PolicyProfile(uniform))
        result = evaluate_exact_kernel("CornerDhEvaluator", policy, action_spaces)
        runtime = get_evaluation_runtime()
        worker_pid = runtime.worker_pid

        if not torch_first:
            import torch

        repeated = evaluate_exact_kernel("CornerDhEvaluator", policy, action_spaces)

        assert native_module not in sys.modules
        assert isinstance(runtime, LocalWorkerRuntime)
        assert worker_pid is not None
        assert runtime.worker_pid == worker_pid
        assert result["expected_returns"][0] == 0.3714285714285717
        assert sum(result["player_improvements"].values()) == 1.7904761904761903
        assert repeated == result
        assert torch.softmax(torch.randn(2, 3), dim=-1).shape == (2, 3)
        close_default_evaluation_runtime()
        """
    ).replace("__TORCH_FIRST__", repr(torch_first))

    assert_python_succeeds(program)
