"""Validated result documents produced by allocated workloads."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self, cast

from benchmax.execution.specs import EvaluationSpec, SpecError, _ArtifactMetadata
from benchmax.execution.values import value_from_document, value_to_document
from benchmax.utils.hashing import _is_sha256
from benchmax.utils.json import JsonObject, JsonValue


class ResultError(SpecError):
    """A workload result violates its durable JSON contract."""


def _metrics_document(metrics: Mapping[str, object]) -> JsonObject:
    """Encode every metric value without game-specific key conventions."""

    documents: JsonObject = {}

    for name, value in metrics.items():
        if not isinstance(name, str):
            raise ResultError(f"metrics contains non-string metric name {name!r}")

        try:
            document = value_to_document(value, location=f"metrics.{name}")
            assert document is None or isinstance(document, (bool, int, float, str, list, dict))
            documents[name] = cast(JsonValue, document)
        except (TypeError, ValueError) as error:
            raise ResultError(str(error)) from error

    return documents


def _metrics_value(document: Mapping[str, object]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    for name, value in document.items():
        if not isinstance(name, str):
            raise ResultError(f"metrics contains non-string metric name {name!r}")

        try:
            metrics[name] = value_from_document(value)
        except (TypeError, ValueError) as error:
            raise ResultError(f"metrics.{name} is not portable: {error}") from error

    return metrics


@dataclass(frozen=True, slots=True, init=False)
class WorkloadResult:
    """One strict durable result envelope for every operation."""

    workload_digest: str
    operation: Literal["training", "evaluation"]
    _payload_json: str
    schema_version: ClassVar[Literal[1]] = 1

    def __init__(self) -> None:
        raise TypeError("use WorkloadResult.training() or WorkloadResult.evaluation()")

    @classmethod
    def _create(cls, workload_digest: str, operation: Literal["training", "evaluation"], payload: JsonObject) -> Self:
        if not _is_sha256(workload_digest):
            raise ResultError("workload_digest must be a lowercase SHA-256 digest")

        try:
            encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ResultError("result payload must be JSON data") from error

        instance = object.__new__(cls)
        object.__setattr__(instance, "workload_digest", workload_digest)
        object.__setattr__(instance, "operation", operation)
        object.__setattr__(instance, "_payload_json", encoded)
        return instance

    @classmethod
    def training(cls, workload_digest: str, seed: int, policy: _ArtifactMetadata) -> Self:
        """Validate the policy payload of one training operation."""

        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ResultError("training result seed must be an integer")

        if not isinstance(policy, _ArtifactMetadata) or policy.digest is None:
            raise ResultError("training result policy must be content-pinned")

        return cls._create(workload_digest, "training", {"seed": seed, "policy": policy.to_document()})

    @classmethod
    def evaluation(cls, workload_digest: str, metrics: Mapping[str, object], *, workload: EvaluationSpec | None = None,
                   resolved_workload: EvaluationSpec | None = None, provenance: Mapping[str, Any] | None = None) -> Self:
        """Validate the JSON metrics of one evaluation operation."""

        payload = {"metrics": _metrics_document(metrics)}
        if workload is not None:
            if not isinstance(workload, EvaluationSpec) or workload.digest != workload_digest:
                raise ResultError("evaluation specification does not match workload_digest")

            payload["workload"] = workload.to_document()

        if resolved_workload is not None:
            if (workload is None or not isinstance(resolved_workload, EvaluationSpec)
                    or resolved_workload.game != workload.game
                    or not isinstance(resolved_workload.policy, _ArtifactMetadata)):
                raise ResultError("resolved evaluation must retain the game and pin its policy artifact")

            if isinstance(workload.policy, _ArtifactMetadata) and workload.digest != resolved_workload.digest:
                raise ResultError("resolved evaluation changes an already pinned policy")

            payload["resolved_workload"] = resolved_workload.to_document()

        if provenance is not None:
            if workload is None or not isinstance(provenance, Mapping):
                raise ResultError("evaluation provenance requires a specification and an object")

            payload["provenance"] = dict(provenance)

        return cls._create(workload_digest, "evaluation", payload)

    @property
    def workload(self) -> EvaluationSpec | None:
        document = self._payload.get("workload")
        return EvaluationSpec.from_document(document) if document is not None else None

    @property
    def resolved_workload(self) -> EvaluationSpec | None:
        document = self._payload.get("resolved_workload")
        return EvaluationSpec.from_document(document) if document is not None else None

    @property
    def provenance(self) -> dict[str, Any] | None:
        return self._payload.get("provenance")

    @property
    def kind(self) -> str:
        return f"{self.operation}-result"

    @property
    def _payload(self) -> JsonObject:
        value = json.loads(self._payload_json)
        assert isinstance(value, dict)
        return value

    @property
    def seed(self) -> int:
        if self.operation != "training":
            raise ResultError("an evaluation result has no training seed")

        seed = self._payload["seed"]
        assert isinstance(seed, int) and not isinstance(seed, bool)
        return seed

    @property
    def policy(self) -> _ArtifactMetadata:
        if self.operation != "training":
            raise ResultError("an evaluation result has no policy artifact")

        return _ArtifactMetadata.from_document(self._payload["policy"])

    @property
    def metrics(self) -> dict[str, Any]:
        if self.operation != "evaluation":
            raise ResultError("a training result has no evaluation metrics")

        metrics = self._payload["metrics"]
        assert isinstance(metrics, dict)
        return _metrics_value(metrics)

    def to_document(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "workload_digest": self.workload_digest,
            **self._payload,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        kind = document.get("kind")
        if kind not in ("training-result", "evaluation-result"):
            raise ResultError(f"unknown result kind {kind!r}")

        operation = kind.removesuffix("-result")
        required = {"schema_version", "kind", "workload_digest"}
        required |= {"seed", "policy"} if operation == "training" else {"metrics"}
        optional = {"workload", "resolved_workload", "provenance"} if operation == "evaluation" else set()
        if not required <= document.keys() or document.keys() - required - optional:
            raise ResultError(f"{operation} result has missing or unknown fields")

        if document["schema_version"] != 1:
            raise ResultError(f"unsupported {operation} result header")

        if operation == "training":
            return cls.training(document["workload_digest"], document["seed"],
                                _ArtifactMetadata.from_document(document["policy"]))

        for key in {"metrics"} | (optional & document.keys()):
            if not isinstance(document[key], Mapping):
                raise ResultError(f"evaluation result {key} must be an object")

        specifications = {key: EvaluationSpec.from_document(document[key])
                          for key in ("workload", "resolved_workload") if key in document}
        return cls.evaluation(document["workload_digest"], _metrics_value(document["metrics"]),
                              provenance=document.get("provenance"), **specifications)


def result_from_document(document: Mapping[str, Any]) -> WorkloadResult:
    """Validate one persisted workload result."""

    return WorkloadResult.from_document(document)
