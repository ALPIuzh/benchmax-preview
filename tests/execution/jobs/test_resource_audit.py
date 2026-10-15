from importlib.util import find_spec

import pytest

from benchmax.execution.audit.resources import audit_workload
from benchmax.execution.specs import EvaluationSpec, TrainingSpec, _ArtifactMetadata
from tests.process_helpers import assert_python_succeeds

requires_training = pytest.mark.skipif(
    find_spec("benchmarl") is None or find_spec("torch") is None, reason="requires the training extra"
)


def test_benchmarl_evidence_is_importable_without_loading_training_stack() -> None:
    program = """
import sys

import benchmax.integrations.benchmarl.evidence

assert "benchmax.integrations.benchmarl.provider" not in sys.modules
assert "benchmarl" not in sys.modules
assert "torch" not in sys.modules
"""
    assert_python_succeeds(program)


def test_training_audit_does_not_require_an_accelerator_for_indexed_cpu() -> None:
    workload = TrainingSpec.create(game="rock_paper_scissors", seed=7, device="cpu:0")
    report = audit_workload(workload, target="local")

    assert report.requirements["accelerator"] == "none"


@requires_training
def test_benchmarl_training_is_sized_from_the_resolved_model_and_batch() -> None:
    report = audit_workload(TrainingSpec.create(game="phantom_tic_tac_toe", algorithm="mappo", seed=7))

    assert report.requirements["host_memory"] == "ordinary"
    assert report.estimates["model_parameter_count"] == 435_732
    assert report.estimates["model_parameter_bytes"] == 1_742_928
    assert report.estimates["frames_per_batch"] == 6_000
    assert report.duration_seconds < 5


@requires_training
def test_iql_resource_evidence_counts_q_network_replay_and_off_policy_batches():
    settings = {
        "model": {"num_cells": [8]},
        "experiment": {
            "off_policy_collected_frames_per_batch": 8,
            "off_policy_n_envs_per_worker": 2,
            "off_policy_memory_size": 32,
        },
    }
    report = audit_workload(TrainingSpec.create(
        game="rock_paper_scissors", algorithm="iql", seed=7, total_frames=16,
        algorithm_overrides={"params": settings},
    ))
    assert report.estimates["model_parameter_count"] == 102
    assert report.estimates["frames_per_batch"] == 8
    assert report.estimates["environments_per_worker"] == 2
    assert report.estimates["training_state_lower_bound_bytes"] > 102 * 4 * 5


@requires_training
def test_structural_training_audit_flags_a_huge_model_without_allocating_it() -> None:
    report = audit_workload(
        TrainingSpec.create(
            game="phantom_tic_tac_toe", algorithm="mappo", seed=7,
            algorithm_overrides={"params": {"model": {"num_cells": [100_000, 100_000]}}}
        )
    )

    assert report.requirements["host_memory"] == "high"
    assert report.estimates["model_parameter_bytes"] > 40_000_000_000


def test_exact_evaluation_does_not_use_bundled_machine_measurements() -> None:
    report = audit_workload(
        EvaluationSpec.create(game="phantom_tic_tac_toe", policy=_ArtifactMetadata("policy.json", "0" * 64)),
        target="local",
    )

    assert report.requirements["accelerator"] == "none"
    assert report.requirements["host_memory"] == "unknown"
    assert "measured_peak_rss_bytes" not in report.estimates
    assert "logical_policy_input_bytes" not in report.estimates
    assert report.evidence[0]["provenance"]["status"] == "structural-unmeasured"
    assert any("does not ship" in finding for finding in report.findings)


def test_liars_dice_exact_evaluation_has_structural_evidence() -> None:
    report = audit_workload(
        EvaluationSpec.create(game="liars_dice_2d5f", policy=_ArtifactMetadata("policy.json", "0" * 64))
    )

    assert report.requirements["host_memory"] == "unknown"
    assert report.requirements["accelerator"] == "none"
    assert report.evidence[0]["provenance"]["traverser_name"] == "Ld2d5fEvaluator"
    assert report.probes[0]["status"] == "resolved"


def test_unmeasured_exact_variants_are_not_given_another_games_numbers() -> None:
    report = audit_workload(
        EvaluationSpec.create(game="abrupt_phantom_tic_tac_toe", policy=_ArtifactMetadata("policy.json", "0" * 64))
    )

    assert report.requirements["host_memory"] == "unknown"
    assert "measured_peak_rss_bytes" not in report.estimates
    assert report.evidence[0]["provenance"]["status"] == "structural-unmeasured"
