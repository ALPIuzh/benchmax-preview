"""Provider-extensible execution evidence for resource audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import entry_points
from typing import Any, Protocol

from benchmax.execution.specs import WorkloadSpec

_ENTRY_POINT_GROUP = "benchmax.execution_evidence"


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    """One integration's characteristics and compatible measurements."""

    operation: str
    requirements: dict[str, str]
    measurements: dict[str, Any]
    provenance: dict[str, Any]
    findings: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        value = asdict(self)
        value["findings"] = list(self.findings)
        return value


class EvidenceResolver(Protocol):
    """Return evidence when an integration recognizes a workload."""

    def __call__(self, workload: WorkloadSpec) -> ExecutionEvidence | None: ...


def execution_evidence(workload: WorkloadSpec) -> tuple[ExecutionEvidence, ...]:
    """Ask installed integration entry points for compatible evidence."""

    matched: list[ExecutionEvidence] = []
    providers = sorted(
        entry_points(group=_ENTRY_POINT_GROUP),
        key=lambda entry: entry.name,
    )

    for provider in providers:
        resolver: EvidenceResolver = provider.load()
        evidence = resolver(workload)
        if evidence is not None:
            matched.append(evidence)

    return tuple(matched)
