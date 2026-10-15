"""Inspectable job progress with passive notebook and terminal representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from typing import Any

from benchmax.execution.specs import TrainingSpec, WorkloadSpec
from benchmax.execution.targets.service import _now


@dataclass(frozen=True, slots=True, repr=False)
class JobProgress:
    """A fetched snapshot of a job's progress.

    ``state`` is the execution lifecycle, while ``phase`` explains dependency waits
    even when a local attempt process has already started. Missing measurements
    remain None. An expected scheduler start is distinct from completion ETA.
    """

    job_id: str | None
    operation: str
    state: str
    phase: str
    checked_at: str
    reason: str | None = None
    dependencies: tuple[JobProgress, ...] = ()
    scheduler_id: str | None = None
    scheduler_state: str | None = None
    submitted_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    observed_at: str | None = None
    elapsed_seconds: float | None = None
    expected_start: str | None = None
    completed: int | None = None
    total: int | None = None
    unit: str | None = None
    eta_seconds: float | None = None
    warnings: tuple[str, ...] = ()

    def to_document(self) -> dict[str, Any]:
        value = asdict(self)
        value["dependencies"] = [item.to_document() for item in self.dependencies]
        value["warnings"] = list(self.warnings)
        return value

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> JobProgress:
        fields = dict(document)
        fields["dependencies"] = tuple(cls.from_document(item) for item in fields.get("dependencies", ()))
        fields["warnings"] = tuple(fields.get("warnings", ()))
        return cls(**fields)

    @property
    def fraction(self) -> float | None:
        if self.completed is None or not self.total:
            return None

        return min(1.0, max(0.0, self.completed / self.total))

    def _details(self) -> list[str]:
        details = [self.reason] if self.reason else []

        if self.scheduler_id:
            details.append(f"Slurm job: {self.scheduler_id}")

        if self.elapsed_seconds is not None:
            details.append(f"Attempt elapsed: {self.elapsed_seconds:.0f}s")

        if self.completed is not None and self.total is not None:
            details.append(f"Progress: {self.completed:,} / {self.total:,} {self.unit or 'units'}")
        elif self.state not in {"succeeded", "failed", "cancelled", "created"}:
            details.append("Progress: not reported")

        if self.expected_start:
            details.append(f"Scheduler estimated start: {self.expected_start}")

        if self.state not in {"succeeded", "failed", "cancelled", "created"}:
            details.append("ETA: unavailable" if self.eta_seconds is None else f"ETA: ~{self.eta_seconds:.0f}s")

        details.extend(self.warnings)
        return details

    def _text(self, prefix: str = "") -> list[str]:
        title = f"{self.operation.capitalize()} {self.job_id or '(not submitted)'} — {self.phase.replace('_', ' ')}"
        lines = [prefix + title, *(prefix + "  " + item for item in self._details())]

        for child in self.dependencies:
            child_lines = child._text(prefix + "   ")
            child_lines[0] = prefix + "└─ " + child_lines[0][len(prefix) + 3:]
            lines.extend(child_lines)

        return lines

    def __repr__(self) -> str:
        return "\n".join([*self._text(), f"Checked: {self.checked_at}"])

    __str__ = __repr__

    def _html(self) -> str:
        title = escape(f"{self.operation.capitalize()} {self.job_id or '(not submitted)'}")
        phase = escape(self.phase.replace("_", " "))
        details = "".join(f"<div>{escape(item)}</div>" for item in self._details())
        bar = ""

        if self.fraction is not None:
            bar = f'<progress value="{self.fraction}" max="1" style="width:100%"></progress>'

        children = "".join(child._html() for child in self.dependencies)
        if children:
            children = ('<div style="margin:12px 0 0 10px;padding-left:16px;border-left:2px solid #888">'
                        '<div style="font-size:0.85em;opacity:0.75">Depends on</div>' + children + "</div>")

        return (f'<div style="padding:8px 0"><strong>{title}</strong> '
                f'<span style="border:1px solid #888;border-radius:10px;padding:1px 7px">{phase}</span>'
                f'<div style="margin-top:6px;line-height:1.6">{details}</div>{bar}{children}</div>')

    def _repr_html_(self) -> str:
        """Render the snapshot as HTML for Jupyter."""
        footer = f'<div style="font-size:0.85em;opacity:0.75">Checked: {escape(self.checked_at)}</div>'
        return ('<div style="border:1px solid #888;border-radius:8px;padding:12px 16px;max-width:800px">'
                + self._html() + footer + "</div>")


def unsubmitted_progress(workload: WorkloadSpec) -> JobProgress:
    """Inspect an unsubmitted workload without following remote references."""
    from benchmax.execution.jobs.outputs import dependency

    checked = _now()
    children = ()

    if reference := dependency(workload):
        children = (JobProgress(reference.job_id, "training", "unknown", "unknown", checked,
                                reason="Not queried: this evaluation has not been submitted."),)

    total = workload.total_frames if isinstance(workload, TrainingSpec) else None
    return JobProgress(None, workload.kind, "created", "created", checked, reason="Not submitted.",
                       dependencies=children, total=total)
