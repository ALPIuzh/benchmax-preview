"""An unlaunched local store; each test owns its workload and state changes."""

import pytest

from benchmax.execution.targets.local import LocalJobService


@pytest.fixture
def local_service(tmp_path):
    return LocalJobService(tmp_path / "runs", launcher=lambda *args: None)
