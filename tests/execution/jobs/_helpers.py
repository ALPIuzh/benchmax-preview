from __future__ import annotations

from pathlib import Path

from benchmax.execution.artifacts.results import WorkloadResult
from benchmax.execution.audit.resources import AuditReport
from benchmax.execution.specs import EvaluationSpec, TrainingSpec, _ArtifactMetadata
from benchmax.execution.targets.local import LocalJobService
from benchmax.execution.targets.local.attempt import run_local_attempt


def _result(workload: TrainingSpec) -> WorkloadResult:
    return WorkloadResult.training(workload.digest, workload.seed, _ArtifactMetadata("policy/manifest.json", "0" * 64))


def _execute(service: LocalJobService, job_id: str, *, executor):
    return run_local_attempt(job_id, runs=service._runs, executor=executor)


class _Handle:
    def __init__(self, workload: TrainingSpec | EvaluationSpec) -> None:
        self.id = f"{workload.kind}-0000000000000001"
        self.state = "succeeded"
        self.workload = workload
        self.output = Path("result.json").resolve()

    def result(self, timeout: float | None = None) -> dict[str, object]:
        del timeout
        assert isinstance(self.workload, EvaluationSpec)
        return WorkloadResult.evaluation(
            self.workload.digest, {"exploitability": 0.25, "custom_player_metric": {0: 1.0, 1: -1.0}}
        ).to_document()

    async def result_async(self, timeout: float | None = None) -> dict[str, object]:
        return self.result(timeout)


class _Service:
    def __init__(self) -> None:
        self.submissions: list[tuple[object, object]] = []
        self.audits: list[object] = []
        self.handles: dict[str, _Handle] = {}

    def submit(self, workload, *, output=None):
        self.submissions.append((workload, output))
        handle = _Handle(workload)
        self.handles[handle.id] = handle
        return handle

    def open(self, job_id: str) -> _Handle:
        return self.handles[job_id]

    def audit(self, workload):
        self.audits.append(workload)
        return AuditReport(
            workload=workload, target="local", budget_seconds=45.0, duration_seconds=0.1,
            requirements={"host_memory": "ordinary"}, estimates={}, machine={"accelerators": []}, findings=()
        )
