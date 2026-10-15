import json
import shutil

import pytest

from benchmax.execution.audit.resources import AuditReport
from benchmax.execution.specs import (
    EntrySpec,
    EvaluationSpec,
    SpecError,
    TrainingSpec,
    _ArtifactMetadata,
    load_workload,
    validate_workload_resolution,
)
from benchmax.seeding import SEED_SCHEME
from benchmax.utils.json import write_document
from tests.process_helpers import assert_python_succeeds


def _audit_report(
    workload: TrainingSpec | EvaluationSpec, *, target: str, memory_bytes: int | None = None
) -> AuditReport:
    return AuditReport(
        workload=workload, target=target, budget_seconds=45.0, duration_seconds=0.1,
        requirements={"host_memory": memory_bytes}, estimates={}, machine={"accelerators": []}, findings=()
    )


def test_training_specs_resolve_the_seed_before_their_identity_is_computed() -> None:
    first = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    same = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    other = TrainingSpec.create(game="rock_paper_scissors", seed=8)

    assert first.seed == 7
    assert first.digest == same.digest
    assert first.digest != other.digest


def test_evaluator_is_not_part_of_a_training_workload() -> None:
    configured = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    unevaluated = TrainingSpec.create(game="rock_paper_scissors", game_overrides={"evaluator": None}, seed=7)

    assert configured.game.to_document() == unevaluated.game.to_document()
    assert configured.game.overrides["evaluator"] is None
    assert configured.game.resolved is not None
    assert configured.game.resolved["evaluator"] is None
    assert configured.digest == unevaluated.digest


def test_evaluator_drift_does_not_block_a_training_workload() -> None:
    document = TrainingSpec.create(game="rock_paper_scissors", seed=7).to_document()
    document["game"]["resolved"]["evaluator"] = {"_target_": "removed.Backend"}
    workload = TrainingSpec.from_document(document)

    validate_workload_resolution(workload)


def test_unresolved_training_specs_still_have_an_auditable_identity() -> None:
    configured = TrainingSpec(game=EntrySpec("optional.game", {"evaluator": "old_backend"}), seed=7)
    unevaluated = TrainingSpec(game=EntrySpec("optional.game", {"evaluator": None}), seed=7)

    assert configured.digest == unevaluated.digest


def test_training_spec_and_drift_check_are_structural_without_the_training_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmax import providers

    available = providers.find_spec
    monkeypatch.setattr(providers, "find_spec", lambda module: None if module == "benchmarl" else available(module))

    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)

    assert spec.algorithm.resolved is not None
    assert spec.algorithm.resolved["provider"] == "benchmarl"
    validate_workload_resolution(spec)


def test_execution_reports_a_missing_training_runtime(tmp_path) -> None:
    assert_python_succeeds(
        f"""
        import builtins
        from pathlib import Path

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "benchmarl" or name.startswith("benchmarl."):
                raise ModuleNotFoundError("No module named 'benchmarl'")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked

        from benchmax.execution.specs import TrainingSpec
        from benchmax.execution.jobs.workload import execute_workload

        spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)
        try:
            execute_workload(spec, Path({str(tmp_path / "result.json")!r}))
        except ModuleNotFoundError as error:
            assert "install benchmax[training]" in str(error), error
        else:
            raise AssertionError("execution succeeded without BenchMARL")
        """,
        timeout=10,
    )


def test_generated_training_seeds_are_structural_values() -> None:
    spec = TrainingSpec.create(game="rock_paper_scissors")

    assert 0 <= spec.seed < 2**31
    assert spec.to_document()["seed"] == spec.seed
    assert spec.to_document()["seed_scheme"] == SEED_SCHEME


def test_training_seed_scheme_survives_a_document_round_trip() -> None:
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)

    assert TrainingSpec.from_document(spec.to_document()) == spec


def test_training_documents_require_an_explicit_seed_scheme() -> None:
    document = TrainingSpec.create(game="rock_paper_scissors", seed=7).to_document()
    del document["seed_scheme"]

    with pytest.raises(SpecError, match="missing training fields: seed_scheme"):
        TrainingSpec.from_document(document)


def test_unknown_seed_schemes_are_rejected() -> None:
    document = TrainingSpec.create(game="rock_paper_scissors", seed=7).to_document()
    document["seed_scheme"] = "future-v99"

    with pytest.raises(SpecError, match="unsupported seed scheme"):
        TrainingSpec.from_document(document)


def test_audit_evidence_does_not_change_the_workload_identity(tmp_path) -> None:
    workload = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    first = _audit_report(workload, target="laptop", memory_bytes=1)
    second = _audit_report(workload, target="cluster", memory_bytes=2)
    path = tmp_path / "audited.json"

    write_document(first, path)

    assert first.workload.digest == second.workload.digest == workload.digest
    assert load_workload(path, expected_kind="training") == workload


def test_input_kind_is_checked_after_an_audit_envelope_is_unwrapped(tmp_path) -> None:
    workload = EvaluationSpec(game=EntrySpec("rock_paper_scissors"), policy=_ArtifactMetadata("policy.json", "0" * 64))
    path = tmp_path / "evaluation.json"
    write_document(_audit_report(workload, target="local"), path)

    with pytest.raises(SpecError, match="contains an evaluation workload"):
        load_workload(path, expected_kind="training")


def test_specs_reject_unknown_fields_instead_of_silently_ignoring_them(tmp_path) -> None:
    path = tmp_path / "training.json"
    document = TrainingSpec.create(game="rock_paper_scissors", seed=7).to_document()
    document["mystery"] = True
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SpecError, match="unknown training fields: mystery"):
        load_workload(path)


def test_training_budget_has_only_one_canonical_location() -> None:
    with pytest.raises(SpecError, match="training workload field"):
        TrainingSpec.create(game="rock_paper_scissors", seed=7, algorithm_overrides={"total_frames": 8})


@pytest.mark.parametrize(
    "overrides",
    [
        {"_target_": "malicious.Code"},
        {"params": {"nested": {"_target_": "malicious.Code"}}},
        {"evaluator": {"_use_": "malicious"}},
    ],
)
def test_workload_overrides_cannot_supply_hydra_loader_controls(overrides) -> None:
    with pytest.raises(SpecError, match="reserved configuration key"):
        EntrySpec("rock_paper_scissors", overrides)


@pytest.mark.parametrize("group", ["game", "algorithm"])
def test_execution_detects_configuration_drift_in_an_input_document(tmp_path, group) -> None:
    document = TrainingSpec.create(game="rock_paper_scissors", seed=7).to_document()
    document[group]["resolved"]["changed"] = True
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    workload = load_workload(path)

    with pytest.raises(SpecError, match=f"resolved {group} .* differs from the workload"):
        validate_workload_resolution(workload)


def test_execution_detects_configuration_file_changes_in_the_same_process(tmp_path, monkeypatch) -> None:
    from benchmax import config

    directory = config._group_directory
    shutil.copytree(directory("game"), tmp_path / "game")
    monkeypatch.setattr(config, "_group_directory",
                        lambda group: tmp_path / "game" if group == "game" else directory(group))
    workload = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    path = tmp_path / "game" / "rock_paper_scissors.yaml"
    path.write_text(path.read_text().replace("max_episode_steps: 1", "max_episode_steps: 2"))

    with pytest.raises(SpecError, match="resolved game .* differs from the workload"):
        validate_workload_resolution(workload)


def test_nested_spec_values_and_returned_documents_are_detached() -> None:
    overrides = {"params": {"model": {"num_cells": [8]}}}
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7, algorithm_overrides=overrides)
    digest = spec.digest

    overrides["params"]["model"]["num_cells"][0] = 1024
    document = spec.to_document()
    document["algorithm"]["overrides"]["params"]["model"]["num_cells"][0] = 2048

    assert spec.algorithm.overrides["params"]["model"]["num_cells"] == [8]
    assert spec.digest == digest


def test_evaluation_creation_pins_the_policy_document(tmp_path) -> None:
    policy = tmp_path / "training-result.json"
    policy.write_text('{"kind":"training-result"}', encoding="utf-8")

    spec = EvaluationSpec.create(game="rock_paper_scissors", policy=str(policy))
    digest = spec.digest
    policy.write_text('{"kind":"changed"}', encoding="utf-8")

    assert spec.policy.digest is not None
    assert spec.digest == digest


def test_policy_location_does_not_change_evaluation_identity() -> None:
    first = EvaluationSpec(
        game=EntrySpec("rock_paper_scissors"), policy=_ArtifactMetadata("/submitter/policy.json", "0" * 64)
    )
    staged = EvaluationSpec(
        game=EntrySpec("rock_paper_scissors"), policy=_ArtifactMetadata("/target/cache/policy.json", "0" * 64)
    )

    assert first.digest == staged.digest
