from __future__ import annotations

import numpy as np
import pytest

from benchmax.execution.artifacts.results import ResultError, WorkloadResult
from benchmax.execution.specs import EvaluationSpec, _ArtifactMetadata

_DIGEST = "0" * 64


def test_result_envelope_requires_an_operation_constructor() -> None:
    with pytest.raises(TypeError, match="WorkloadResult.training"):
        WorkloadResult()


def test_evaluation_result_has_an_explicit_stable_json_contract() -> None:
    metrics = {
        "expected_returns": {0: 1.0, 1: -1.0},
        "player_improvements": {0: 0.25, 1: 0.5},
        "custom_player_metric": {0: (1.0, 2.0)},
        "exploitability": 0.375,
    }

    result = WorkloadResult.evaluation(_DIGEST, metrics)
    metrics["exploitability"] = 100.0

    assert result.to_document()["metrics"] == {
        "expected_returns": {"type": "map", "items": [[0, 1.0], [1, -1.0]]},
        "player_improvements": {"type": "map", "items": [[0, 0.25], [1, 0.5]]},
        "custom_player_metric": {"type": "map", "items": [[0, {"type": "tuple", "items": [1.0, 2.0]}]]},
        "exploitability": 0.375,
    }
    assert result.metrics["expected_returns"] == {0: 1.0, 1: -1.0}
    assert result.metrics["custom_player_metric"] == {0: (1.0, 2.0)}
    assert WorkloadResult.from_document(result.to_document()).metrics == result.metrics


def test_evaluation_result_reports_nonportable_metrics_with_their_location() -> None:
    with pytest.raises(ResultError, match="metrics.value"):
        WorkloadResult.evaluation(_DIGEST, {"value": np.float32(0.5)})


def test_training_result_requires_a_content_pinned_policy() -> None:
    with pytest.raises(ResultError, match="content-pinned"):
        WorkloadResult.training(_DIGEST, 7, _ArtifactMetadata("policy.json"))


@pytest.mark.parametrize("field", ["workload", "resolved_workload", "provenance"])
def test_result_rejects_malformed_context(field):
    document = WorkloadResult.evaluation("0" * 64, {}).to_document()
    document[field] = []
    with pytest.raises(ResultError, match="must be an object"):
        WorkloadResult.from_document(document)


def test_result_rejects_mismatched_or_changed_specification():
    artifact = _ArtifactMetadata("policy.json", "1" * 64)
    workload = EvaluationSpec.create(game="rock_paper_scissors", policy=artifact)
    with pytest.raises(ResultError, match="does not match"):
        WorkloadResult.evaluation("0" * 64, {}, workload=workload)

    changed = EvaluationSpec.create(game="rock_paper_scissors", policy=_ArtifactMetadata("other", "0" * 64))
    with pytest.raises(ResultError, match="already pinned"):
        WorkloadResult.evaluation(workload.digest, {}, workload=workload, resolved_workload=changed)


def test_evaluation_provenance_is_frozen_and_round_trips():
    spec = EvaluationSpec.create(game="rock_paper_scissors", policy=_ArtifactMetadata("policy.json", "1" * 64))
    provenance = {"policy": {"progress": {"iterations": 12}}}
    result = WorkloadResult.evaluation(spec.digest, {}, workload=spec, provenance=provenance)
    provenance["policy"]["progress"]["iterations"] = 99
    assert result.provenance["policy"]["progress"]["iterations"] == 12
    assert WorkloadResult.from_document(result.to_document()) == result


@pytest.mark.parametrize("value", [float("nan"), object()])
def test_evaluation_provenance_rejects_non_json_values(value):
    spec = EvaluationSpec.create(game="rock_paper_scissors", policy=_ArtifactMetadata("policy.json", "1" * 64))
    with pytest.raises(ResultError, match="JSON data"):
        WorkloadResult.evaluation(spec.digest, {}, workload=spec, provenance={"invalid": value})
