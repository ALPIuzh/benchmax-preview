"""Explainable resource audits for workloads."""

from __future__ import annotations

import os
import platform
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from benchmax.config import _resolve
from benchmax.execution.audit.evidence import ExecutionEvidence, execution_evidence
from benchmax.execution.specs import TrainingSpec, WorkloadSpec

_MAX_AUDIT_SECONDS = 45.0


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Target evidence attached to, but outside, a workload's identity."""

    workload: WorkloadSpec
    target: str
    budget_seconds: float
    duration_seconds: float
    requirements: dict[str, Any]
    estimates: dict[str, Any]
    machine: dict[str, Any]
    findings: tuple[str, ...]
    probes: tuple[dict[str, Any], ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    schema_version: ClassVar[Literal[1]] = 1

    @property
    def workload_digest(self) -> str:
        return self.workload.digest

    def to_document(self) -> dict[str, Any]:
        """Return the complete persisted audit document."""

        return {
            "schema_version": self.schema_version,
            "kind": "audit",
            "workload": self.workload.to_document(),
            "evidence": {
                "workload_digest": self.workload_digest,
                "target": self.target,
                "budget_seconds": self.budget_seconds,
                "duration_seconds": self.duration_seconds,
                "requirements": deepcopy(self.requirements),
                "estimates": deepcopy(self.estimates),
                "machine": deepcopy(self.machine),
                "findings": list(self.findings),
                "probes": deepcopy(list(self.probes)),
                "evidence": deepcopy(list(self.evidence)),
            },
        }


def _memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None

    return int(page_size * pages)


def _visible_accelerators() -> list[str]:
    visible: list[str] = []
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda and cuda not in {"-1", "none", "None"}:
        visible.append("cuda")

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # This is a capability hint. Actual MPS availability is checked by the
        # training process's Torch build when device=auto is resolved.
        visible.append("apple-gpu")

    return visible


def _machine() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "physical_memory_bytes": _memory_bytes(),
        "accelerators": _visible_accelerators(),
    }


def _accelerator_requirement(spec: TrainingSpec) -> tuple[str, str]:
    """Read the scheduling intent a training spec states in its device.

    This is the generic half of device handling: which accelerator a target
    must offer. Resolving an actual framework device stays with the provider.
    """

    if spec.device.partition(":")[0] == "cpu":
        return "none", "Training is pinned to CPU; a GPU is not requested."

    if spec.device == "auto":
        return (
            "preferred",
            "Neural training prefers a visible GPU; device=auto falls back to CPU when the target exposes none.",
        )

    return "required", f"Training explicitly requires the {spec.device!r} device."


def _requirements(
    spec: WorkloadSpec,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], tuple[ExecutionEvidence, ...]]:
    evidence = execution_evidence(spec)
    requirements: dict[str, Any] = {}
    estimates: dict[str, Any] = {}
    findings: list[str] = []

    if isinstance(spec, TrainingSpec):
        accelerator, placement = _accelerator_requirement(spec)
        requirements.update(accelerator=accelerator, host_memory="unknown")
        estimates["total_frames"] = spec.total_frames
        findings.append(placement)

        if not evidence:
            findings.append("Training memory is unknown for this provider; completed-attempt telemetry should establish it.")
    elif not evidence:
        requirements.update(accelerator="unknown", host_memory="unknown")
        findings.append(
            f"Evaluation for {spec.game.name!r} has not been profiled; "
            "the target should use conservative defaults and retain telemetry."
        )

    for record in evidence:
        requirements.update(record.requirements)
        estimates.update(record.measurements)
        findings.extend(record.findings)

    return requirements, estimates, tuple(findings), evidence


def _probe_configurations(
    spec: WorkloadSpec,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    entries = [("game", spec.game.name)]

    if isinstance(spec, TrainingSpec):
        entries.append(("algorithm", spec.algorithm.name))

    probes: list[dict[str, Any]] = []
    findings: list[str] = []

    for group, name in entries:
        started = time.perf_counter()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                _resolve(group, name)
        except ImportError as error:
            status = "missing-dependency"
            findings.append(str(error))
        except ValueError:
            # Unknown/broken names are semantic spec errors, not resource
            # uncertainty. Preserve the configuration layer's actionable text.
            raise
        else:
            status = "resolved"

        probes.append(
            {
                "kind": "configuration",
                "entry_group": group,
                "entry_name": name,
                "status": status,
                "duration_seconds": time.perf_counter() - started,
            }
        )

    return tuple(probes), tuple(findings)


def audit_workload(
    spec: WorkloadSpec, *, target: str = "local", budget_seconds: float = _MAX_AUDIT_SECONDS
) -> AuditReport:
    """Audit one workload without running it.

    The budget is advisory metadata, not an enforced timeout. Configuration
    resolution and provider evidence run synchronously and may exceed it.
    """

    if not 0 < budget_seconds <= _MAX_AUDIT_SECONDS:
        raise ValueError(f"audit budget must lie in (0, {_MAX_AUDIT_SECONDS}]")

    started = time.perf_counter()
    probes, probe_findings = _probe_configurations(spec)

    requirements, estimates, findings, evidence = _requirements(spec)

    return AuditReport(
        workload=spec, target=target, budget_seconds=budget_seconds, duration_seconds=time.perf_counter() - started,
        requirements=requirements, estimates=estimates, machine=_machine(), findings=(*findings, *probe_findings),
        probes=probes, evidence=tuple(record.to_document() for record in evidence)
    )
