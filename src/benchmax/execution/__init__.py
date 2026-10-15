"""Jobs, execution targets, and target registration."""

from benchmax.execution.jobs.job import Job
from benchmax.execution.operation import executable
from benchmax.execution.targets.base import AdapterTarget, LocalTarget, Target
from benchmax.execution.targets.registry import list_available_targets, load_target, register_target
from benchmax.execution.targets.service import JobError, JobFailedError, JobTimeoutError
from benchmax.execution.targets.slurm import SlurmTarget

__all__ = [
    "AdapterTarget",
    "Job",
    "JobError",
    "JobFailedError",
    "JobTimeoutError",
    "LocalTarget",
    "SlurmTarget",
    "Target",
    "executable",
    "list_available_targets",
    "load_target",
    "register_target",
]
