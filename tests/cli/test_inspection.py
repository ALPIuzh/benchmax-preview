from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from benchmax import cli
from benchmax.execution import LocalTarget, SlurmTarget
from benchmax.execution.targets import registry
from benchmax.execution.targets.local import LocalJobService
from benchmax.execution.targets.service import JobError
from benchmax.providers import PROVIDERS


@pytest.fixture(autouse=True)
def profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMAX_TARGETS", str(tmp_path / "profiles.json"))
    monkeypatch.setattr(registry, "_REGISTERED", {})


def test_fresh_lightweight_client_inspects_persisted_target(tmp_path):
    if not PROVIDERS["benchmarl"].available:
        pytest.skip("requires a target with the training extra")

    target = SlurmTarget(python=sys.executable, workspace=str(tmp_path), runs=str(tmp_path / "runs"))
    target.register("university", persist=True)
    program = textwrap.dedent("""
        import importlib.abc
        import sys

        class NoTrainingImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'torch', 'torchrl', 'benchmarl', 'torchvision'}:
                    raise AssertionError('training dependency imported on client: ' + fullname)

        sys.meta_path.insert(0, NoTrainingImports())
        from benchmax import cli
        from benchmax.execution.targets import registry
        assert not registry._REGISTERED
        code = cli.main(['show', 'mappo', '--target', 'university'])
        assert 'torch' not in sys.modules and 'benchmarl' not in sys.modules
        sys.exit(code)
    """)
    process = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=60, check=False)

    assert process.returncode == 0, process.stdout + process.stderr
    assert "Target: university" in process.stdout
    assert "availability  available" in process.stdout
    assert "benchmarl" in process.stdout
    assert "login environment" in process.stdout
    assert not Path(target.runs).exists()


def test_unknown_target_explains_how_to_save_a_profile(capsys):
    assert cli.main(["show", "mappo", "--target", "missing"]) == 1
    error = capsys.readouterr().err

    assert "target.register('missing', persist=True)" in error
    assert os.environ["BENCHMAX_TARGETS"] in error


@pytest.mark.parametrize("error", [JobError("connection refused"), RuntimeError("broken runtime")])
def test_probe_failures_are_distinct_from_unavailable(tmp_path, monkeypatch, capsys, error):
    LocalTarget(runs=tmp_path / "runs").register("broken")

    def inspect(*args):
        raise error

    monkeypatch.setattr(LocalJobService, "inspect_algorithm", inspect)
    assert cli.main(["show", "mappo", "--target", "broken"]) == 2
    output = capsys.readouterr()

    assert "inspection failed on broken" in output.err
    assert "unavailable" not in output.out


def test_target_inspection_reuses_local_description_without_constructing_work(tmp_path, capsys, monkeypatch):
    from benchmax.training.cfr import CFRTrainingFunction

    def unexpected(*args, **kwargs):
        pytest.fail("inspection must not construct or execute training")

    monkeypatch.setattr(CFRTrainingFunction, "__init__", unexpected)
    monkeypatch.setattr(CFRTrainingFunction, "__call__", unexpected)
    destination = LocalTarget(runs=tmp_path / "runs").register("here")
    assert cli.main(["show", "cfr"]) == 0
    local = capsys.readouterr().out
    assert cli.main(["show", "cfr", "--target", "here"]) == 0
    target = capsys.readouterr().out

    assert "Target: here" in target
    assert target.endswith(local)
    assert not Path(destination.runs).exists()


def test_missing_dependency_during_resolution_is_reported(capsys, monkeypatch):
    from benchmax import registry

    def missing(*args):
        raise ModuleNotFoundError("provider dependency is not installed")

    monkeypatch.setattr(registry, "_resolve", missing)

    assert cli.main(["show", "kuhn_poker"]) == 1
    assert "provider dependency is not installed" in capsys.readouterr().out
