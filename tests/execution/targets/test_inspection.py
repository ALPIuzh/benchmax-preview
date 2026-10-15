from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmax.execution import LocalTarget, SlurmTarget
from benchmax.execution.targets.service import JobError
from benchmax.execution.targets.slurm import bridge


@pytest.mark.parametrize("host", [None, "cluster"])
def test_slurm_inspection_uses_execution_environment_without_scheduler(tmp_path, monkeypatch, host):
    workspace = tmp_path / "workspace with spaces; $literal"
    workspace.mkdir()
    target = SlurmTarget(python=sys.executable, workspace=str(workspace), runs=str(tmp_path / "runs"), host=host)
    run = subprocess.run
    calls = []

    def transport(command, **kwargs):
        calls.append((command, kwargs))
        if host:
            # Execute the exact quoted SSH payload locally, with no cluster access.
            assert command[0] == "ssh"
            assert command[-2] == host
            command = ["/bin/sh", "-c", command[-1]]
        else:
            assert kwargs["cwd"] == str(workspace)

        return run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", transport)
    monkeypatch.setenv("PYTHONPATH", "/must/not/be/used")
    monkeypatch.setenv("PYTHONHOME", "/must/not/be/used")
    report = target.service().inspect_algorithm("cfr")

    assert report["available"]
    assert report["python"] == sys.executable
    assert report["workspace"] == str(workspace)
    assert "compute-node GPU/runtime availability is not tested" in report["scope"]
    assert len(calls) == 1
    assert not Path(target.runs).exists()
    assert not list(workspace.iterdir())


def test_slurm_bridge_inspection_never_calls_scheduler(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("inspection must not contact the scheduler")

    monkeypatch.setattr(bridge, "_run", unexpected)
    monkeypatch.setattr(bridge, "_submission", unexpected)
    target = SlurmTarget(python=sys.executable, workspace=str(tmp_path), runs=str(tmp_path / "runs"))
    document = bridge.dispatch({"schema_version": 1, "operation": "inspect", "target": target.to_document(),
                                "name": "cfr"})

    assert document["available"]
    assert not Path(target.runs).exists()


def test_response_must_match_requested_algorithm(tmp_path, monkeypatch):
    from benchmax.execution.targets.slurm.service import SlurmJobService

    document = LocalTarget().service().inspect_algorithm("cfr")
    monkeypatch.setattr(SlurmJobService, "_request", lambda *args, **kwargs: document)
    target = SlurmTarget(python=sys.executable, workspace=str(tmp_path), runs=str(tmp_path / "runs"))
    with pytest.raises(JobError, match="invalid algorithm inspection"):
        target.service().inspect_algorithm("mappo")
