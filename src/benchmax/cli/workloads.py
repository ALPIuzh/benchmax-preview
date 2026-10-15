"""Signature-driven CLI adapters for training and evaluation workloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import Group, Parameter, ValidationError, validators

from benchmax.execution.audit.resources import AuditReport
from benchmax.execution.jobs.job import Job
from benchmax.execution.specs import EvaluationSpec, SpecError, TrainingSpec, WorkloadSpec, load_workload
from benchmax.utils.json import write_document

from ._render import render_value

_MODE = Group("Execution mode", validator=validators.LimitedChoice(max=1))

InputPath = Annotated[Path | None, Parameter(help="Load a versioned workload JSON document.")]
OutputPath = Annotated[Path | None, Parameter(help="Write the primary durable result to this path.")]
SubmitFlag = Annotated[
    bool, Parameter(group=_MODE, negative=(), help="Return the durable job ID immediately instead of waiting.")
]
AuditFlag = Annotated[
    bool, Parameter(group=_MODE, negative=(), help="Inspect resource needs without launching the full workload.")
]
TargetName = Annotated[str, Parameter(help="Execution target configured for this machine or cluster.")]
JsonFlag = Annotated[bool, Parameter(negative=(), help="Print machine-readable JSON.")]


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    """Options shared by the train and evaluate commands."""

    input: InputPath = None
    output: OutputPath = None
    submit: SubmitFlag = False
    audit: AuditFlag = False
    target: TargetName = "local"
    json: JsonFlag = False


_DEFAULT_EXECUTION_OPTIONS = ExecutionOptions()
FlatExecutionOptions = Annotated[ExecutionOptions, Parameter(name="*")]

GameName = Annotated[str | None, Parameter(help="Registered game name.")]
AlgorithmName = Annotated[str | None, Parameter(help="Registered training algorithm; defaults to mappo.")]
Seed = Annotated[int | None, Parameter(help="Training seed; generated and persisted when omitted.")]
Device = Annotated[str | None, Parameter(help="Torch device such as auto, cpu, cuda, cuda:1, or mps.")]
FrameBudget = Annotated[int | None, Parameter(help="Positive training frame budget.")]
PolicyPath = Annotated[Path | None, Parameter(help="Content-pinned training result or policy manifest.")]


def _usage_error(command: str, error: Exception) -> ValidationError:
    return ValidationError(exception_message=str(error), command_chain=("benchmax", command))


def _training_spec(
    *, options: ExecutionOptions, game: str | None, algorithm: str | None,
    seed: int | None, device: str | None, total_frames: int | None,
) -> TrainingSpec:
    if options.input is not None:
        conflicting = [
            option
            for option, value in (
                ("--game", game), ("--algorithm", algorithm), ("--seed", seed),
                ("--device", device), ("--total-frames", total_frames),
            )
            if value is not None
        ]
        if conflicting:
            raise SpecError(f"--input cannot be combined with {', '.join(conflicting)}")

        spec = load_workload(options.input, expected_kind="training")
        assert isinstance(spec, TrainingSpec)
        return spec

    if game is None:
        raise SpecError("training requires --game or --input")

    return TrainingSpec.create(
        game=game, algorithm=algorithm or "mappo", seed=seed, device=device or "auto", total_frames=total_frames
    )


def _evaluation_spec(*, options: ExecutionOptions, game: str | None, policy: Path | None) -> EvaluationSpec:
    if options.input is not None:
        conflicting = [option for option, value in (("--game", game), ("--policy", policy)) if value is not None]
        if conflicting:
            raise SpecError(f"--input cannot be combined with {', '.join(conflicting)}")

        spec = load_workload(options.input, expected_kind="evaluation")
        assert isinstance(spec, EvaluationSpec)
        return spec

    if game is None or policy is None:
        raise SpecError("evaluation requires --game and --policy, or --input")

    return EvaluationSpec.create(game=game, policy=str(policy.expanduser().resolve()))


def _render_audit(report: AuditReport, *, output: Path | None, as_json: bool) -> None:
    if as_json:
        render_value(report.to_document(), as_json=True)
        return

    print(f"workload  {report.workload_digest}")
    print(f"target    {report.target}")
    memory = report.requirements.get("host_memory", "unknown")
    accelerator = report.requirements.get("accelerator", "unknown")
    print(f"memory    {memory}")
    print(f"compute   {accelerator}")

    if measured := report.estimates.get("measured_peak_rss_bytes"):
        print(f"peak RSS  {_format_bytes(measured)} (measured target evidence)")

    if logical := report.estimates.get("logical_policy_input_bytes"):
        print(f"policy IO {_format_bytes(logical)} logical observations")

    if model := report.estimates.get("model_parameter_bytes"):
        print(f"model     {_format_bytes(model)} actor/critic parameters")

    if working_set := report.estimates.get("training_state_lower_bound_bytes"):
        print(f"lower bd  {_format_bytes(working_set)} structural training state")

    physical_memory = report.machine.get("physical_memory_bytes")
    if physical_memory:
        print(f"machine   {_format_bytes(physical_memory)} physical memory")

    visible = report.machine.get("accelerators") or []
    print(f"visible   {', '.join(visible) if visible else 'no accelerator detected'}")

    for finding in report.findings:
        print(f"- {finding}")

    if output is not None:
        print(f"audited spec written to {output}")


def _format_bytes(value: int) -> str:
    amount = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]

    for unit in units:
        if abs(amount) < 1024 or unit == units[-1]:
            break

        amount /= 1024

    return f"{amount:.2f} {unit}"


def _handle(spec: WorkloadSpec, options: ExecutionOptions) -> int:
    job = Job._from_workload(spec, output=options.output)

    if options.audit:
        report = job.audit(options.target)

        if options.output is not None:
            write_document(report, options.output)

        _render_audit(report, output=options.output, as_json=options.json)
        return 0

    job.run(options.target)

    if options.submit:
        render_value(job.id, as_json=options.json)
    else:
        render_value(job._result_document().to_document(), as_json=options.json)

    return 0


def train(
    *, game: GameName = None, algorithm: AlgorithmName = None, seed: Seed = None, device: Device = None,
    total_frames: FrameBudget = None, options: FlatExecutionOptions = _DEFAULT_EXECUTION_OPTIONS,
) -> int:
    """Train a policy profile, as train() does in Python.

    By default, wait for the result. --submit runs the same job in the background,
    and --audit checks its resources without running it.
    """

    try:
        spec = _training_spec(
            options=options, game=game, algorithm=algorithm, seed=seed, device=device, total_frames=total_frames
        )
    except SpecError as error:
        raise _usage_error("train", error) from error

    return _handle(spec, options)


def evaluate(
    *, game: GameName = None, policy: PolicyPath = None, options: FlatExecutionOptions = _DEFAULT_EXECUTION_OPTIONS
) -> int:
    """Evaluate an artifact-backed policy on a registered game.

    Uses the game's default evaluator, as game.evaluate() does in Python. By
    default, wait for the result. --submit runs the same job in the background,
    and --audit checks its resources without running it.
    """

    try:
        spec = _evaluation_spec(options=options, game=game, policy=policy)
    except SpecError as error:
        raise _usage_error("evaluate", error) from error

    return _handle(spec, options)
