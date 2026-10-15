"""Resource evidence resolver for exp-a-spiel evaluation kernels."""

from __future__ import annotations

from collections.abc import Mapping

from benchmax.execution.audit.evidence import ExecutionEvidence
from benchmax.execution.specs import EvaluationSpec, WorkloadSpec

_EVALUATOR = "benchmax.integrations.exp_a_spiel.ExpASpielEvaluator"
_KERNEL = "benchmax.integrations.exp_a_spiel.kernel:evaluate_exact_kernel"


def resolve(workload: WorkloadSpec) -> ExecutionEvidence | None:
    """Match resolved evaluator configuration to kernel evidence."""

    if not isinstance(workload, EvaluationSpec):
        return None

    game = workload.game.resolved
    evaluator = game.get("evaluator") if game is not None else None
    if not isinstance(evaluator, Mapping) or evaluator.get("_target_") != _EVALUATOR:
        return None

    traverser = evaluator.get("traverser")
    if not isinstance(traverser, str):
        return None

    findings = (
        (
            "Exact Liar's dice evaluation runs on the CPU in an isolated worker."
            if traverser.startswith("Ld")
            else "Exact exp-a-spiel evaluation is CPU-parallel and runs in an "
            "isolated worker so OpenMP remains enabled."
        ),
        (
            f"Host memory for traverser {traverser!r} is unknown without "
            "target-local active or completed-attempt evidence; BenchMAX does "
            "not ship developer-machine profiles as library data."
        ),
    )
    return ExecutionEvidence(
        operation=_KERNEL,
        requirements={"accelerator": "none", "host_memory": "unknown"},
        measurements={},
        provenance={
            "status": "structural-unmeasured",
            "traverser_name": traverser,
        },
        findings=findings,
    )
