"""Structural resource evidence for BenchMARL training workloads."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

from benchmax.execution.audit.evidence import ExecutionEvidence
from benchmax.execution.specs import EvaluationSpec, TrainingSpec, WorkloadSpec

_GIB = 1024**3


def _linear_mlp_parameters(input_features: int, hidden_features: Sequence[int], output_features: int) -> int:
    widths = (input_features, *hidden_features, output_features)
    return sum(
        source * destination + destination
        for source, destination in pairwise(widths)
    )


def _structural_measurements(spec: TrainingSpec) -> dict[str, int | str]:
    """Derive exact default-MLP sizes without allocating model tensors."""

    import torch
    from benchmarl.algorithms import IppoConfig, IqlConfig, MappoConfig

    from benchmax.integrations.benchmarl.training import load_benchmarl_training_function
    from benchmax.registry import load_game

    training_function = load_benchmarl_training_function(spec)
    algorithm = training_function.algorithm
    if type(algorithm) not in (IppoConfig, MappoConfig, IqlConfig):
        raise ValueError("parameter formula supports IPPO, MAPPO, and IQL")
    model = training_function.model
    if model.layer_class is not torch.nn.Linear or model.norm_class is not None:
        raise ValueError("parameter formula only supports linear MLPs without norms")

    activation = model.activation_class(**(model.activation_kwargs or {}))
    if any(True for _ in activation.parameters()):
        raise ValueError("parameter formula requires a stateless activation")

    game = load_game(spec.game.name, **spec.game.overrides)
    hidden = tuple(int(width) for width in model.num_cells)
    actor_parameters = sum(
        _linear_mlp_parameters(math.prod(game.observation_spaces[player]) + 1, hidden, len(game.action_spaces[player]))
        for player in game.players
    )
    centralized = training_function.algorithm.has_centralized_critic()
    critic_parameters = sum(
        _linear_mlp_parameters(
            math.prod(game.state_space)
            if centralized and game.state_space is not None
            else math.prod(game.observation_spaces[player]) + 1,
            hidden,
            1,
        )
        for player in game.players
    ) if algorithm.has_critic() else 0
    parameter_count = actor_parameters + critic_parameters
    parameter_bytes = parameter_count * 4

    frame_bytes = sum(
        (math.prod(game.observation_spaces[player]) + 1) * 4
        + len(game.action_spaces[player])
        + 8
        + 4
        for player in game.players
    )

    if centralized and game.state_space is not None:
        frame_bytes += math.prod(game.state_space) * 4

    frame_bytes += 3
    config = training_function.experiment
    frames_per_batch = int(config.collected_frames_per_batch(algorithm.on_policy()))
    batch_lower_bound = frame_bytes * frames_per_batch
    # Parameters, gradients, and Adam's two moment tensors, plus two batch-sized
    # buffers. Activations and framework memory are intentionally excluded.
    structural_lower_bound = parameter_bytes * 4 + batch_lower_bound * 2
    if not algorithm.on_policy():
        collected_batch_frames = -(-frames_per_batch // config.off_policy_n_envs_per_worker) * config.off_policy_n_envs_per_worker
        replay_frames = min(
            config.off_policy_memory_size,
            -(-training_function.total_frames // collected_batch_frames) * collected_batch_frames,
        )
        replay_frame_bytes = sum(
            2 * ((math.prod(game.observation_spaces[player]) + 1) * 4 + len(game.action_spaces[player]))
            + 8 + 4 + 2 + 8  # Action, reward, done/terminated, transition duration.
            for player in game.players
        )
        structural_lower_bound += replay_frames * replay_frame_bytes
        if algorithm.delay_value:
            structural_lower_bound += actor_parameters * 4

    if structural_lower_bound >= 8 * _GIB:
        host_memory = "high"
    elif structural_lower_bound >= 2 * _GIB:
        host_memory = "moderate"
    else:
        host_memory = "ordinary"

    return {
        "host_memory": host_memory,
        "model_parameter_count": parameter_count,
        "model_parameter_bytes": parameter_bytes,
        "training_state_lower_bound_bytes": structural_lower_bound,
        "collected_batch_payload_lower_bound_bytes": batch_lower_bound,
        "frames_per_batch": frames_per_batch,
        "environments_per_worker": int(config.n_envs_per_worker(algorithm.on_policy())),
    }


def resolve(workload: WorkloadSpec) -> ExecutionEvidence | None:
    """Return cheap structural evidence for BenchMARL-backed training."""

    if isinstance(workload, EvaluationSpec):
        evaluator = (workload.game.resolved or {}).get("evaluator")
        if not isinstance(evaluator, dict) or evaluator.get("_target_") != "benchmax.evaluation.ApproximateExploitabilityEvaluator":
            return None
        device = evaluator.get("device", "auto")
        return ExecutionEvidence(
            operation="benchmarl.best_response",
            requirements={
                "accelerator": "none" if device == "cpu" else "preferred" if device == "auto" else "required",
                "host_memory": "unknown",
            },
            measurements={
                "training_frames_per_restart": evaluator["total_frames"],
                "restarts_per_player": evaluator["restarts"],
            },
            provenance={"status": "configured-budgets"},
            findings=("Approximate evaluation trains responses sequentially using the configured training device.",),
        )

    if not isinstance(workload, TrainingSpec):
        return None

    algorithm = workload.algorithm.resolved
    if algorithm is None or algorithm.get("provider") != "benchmarl":
        return None

    try:
        measurements = _structural_measurements(workload)
    except (ImportError, TypeError, ValueError) as error:
        return ExecutionEvidence(
            operation="benchmarl.training", requirements={"host_memory": "unknown"}, measurements={},
            provenance={"status": "unsupported-structural-probe"},
            findings=(f"BenchMARL resource sizing could not model this architecture: {error}",)
        )

    host_memory = str(measurements.pop("host_memory"))
    lower_bound = int(measurements["training_state_lower_bound_bytes"])
    return ExecutionEvidence(
        operation="benchmarl.training",
        requirements={"host_memory": host_memory},
        measurements=measurements,
        provenance={
            "status": "structural-estimate",
            "dtype": "float32",
            "formula": "linear-mlp-parameters+gradients+adam+2x-batch-payload+off-policy-replay-and-target",
        },
        findings=(
            (
                "BenchMAX derived the configured actor, critic, optimizer-state, "
                "and batch payload sizes without allocating the model."
            ),
            (
                f"The {lower_bound / 1024**2:.1f} MiB structural working-set "
                "lower bound excludes framework memory and activations; completed "
                "attempt telemetry should refine the recommendation."
            ),
        ),
    )
