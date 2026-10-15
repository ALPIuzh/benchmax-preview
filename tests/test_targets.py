from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from benchmax import cli, load_game
from benchmax.execution import (
    AdapterTarget,
    Job,
    LocalTarget,
    SlurmTarget,
    Target,
    list_available_targets,
    load_target,
    register_target,
)
from benchmax.execution.targets import registry
from benchmax.execution.targets.service import JobError
from benchmax.training import train
from tests.execution.jobs._helpers import _Service


@pytest.fixture(autouse=True)
def profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMAX_TARGETS", str(tmp_path / "profiles.json"))
    monkeypatch.setattr(registry, "_REGISTERED", {})


@pytest.mark.parametrize("kind", ["local", "slurm"])
def test_registration_is_passive_and_persistent(tmp_path, monkeypatch, kind):
    def unexpected(*args, **kwargs):
        pytest.fail("constructing or loading a target must not launch a process")

    monkeypatch.setattr(subprocess, "run", unexpected)
    target = (LocalTarget(runs=tmp_path / "jobs") if kind == "local" else SlurmTarget(
        python="/env/bin/python", workspace="/project", runs=str(tmp_path / "jobs"), host="cluster.example",
        strict_source=True))
    assert target.register("research", persist=True) is target
    assert Target.load("research") is target
    assert list_available_targets() == ["local", "research"]
    registry._REGISTERED.clear()
    assert load_target("research") == target
    assert not (tmp_path / "jobs").exists()
    document = json.loads(Path(os.environ["BENCHMAX_TARGETS"]).read_text())
    assert document["targets"]["research"]["runs"] == str(tmp_path / "jobs")


def test_saved_profile_reconnects_to_a_completed_job_from_another_process(tmp_path, monkeypatch):
    from benchmax.execution.artifacts.results import WorkloadResult
    from benchmax.execution.specs import EntrySpec, EvaluationSpec, _ArtifactMetadata
    from benchmax.execution.targets.local.attempt import run_local_attempt

    target = LocalTarget(runs=tmp_path / "runs").register("research", persist=True)
    service = target.service()
    monkeypatch.setattr(service, "_launcher", lambda *args: None)
    monkeypatch.setattr(LocalTarget, "service", lambda self: service)
    workload = EvaluationSpec(EntrySpec("rock_paper_scissors"), _ArtifactMetadata("policy.json", "0" * 64))
    job = Job._from_workload(workload).run("research")
    run_local_attempt(job.id, runs=target.runs,
                      executor=lambda spec, output: WorkloadResult.evaluation(spec.digest, {"exploitability": 0.25}))
    program = ("from benchmax.execution import Job; import json, sys; "
               "print(json.dumps(Job.open(sys.argv[1], target='research').result()))")

    result = subprocess.run([sys.executable, "-c", program, job.id], capture_output=True, text=True, check=True)

    assert json.loads(result.stdout) == {"exploitability": 0.25}
    assert (Path(target.runs) / job.id / "result.json").is_file()


def test_process_registration_does_not_create_a_saved_profile(tmp_path):
    target = LocalTarget(runs=tmp_path / "runs").register("temporary")
    assert load_target("temporary") is target
    assert not Path(os.environ["BENCHMAX_TARGETS"]).exists()
    registry._REGISTERED.clear()

    with pytest.raises(JobError, match="unknown target"):
        load_target("temporary")


def test_saved_adapter_is_loaded_only_when_its_service_is_requested(monkeypatch, capsys):
    from benchmax.execution.targets import service as service_module

    service = _Service()
    loaded = []

    class EntryPoint:
        name = "test"

        @staticmethod
        def load():
            loaded.append("load")

            def factory(uri):
                assert uri == "test://worker"
                return service

            return factory

    monkeypatch.setattr(service_module, "entry_points", lambda **kwargs: [EntryPoint()])
    AdapterTarget("test://worker").register("remote", persist=True)
    registry._REGISTERED.clear()
    target = load_target("remote")
    assert target == AdapterTarget("test://worker")
    assert cli.main(["targets"]) == 0
    output = capsys.readouterr().out
    assert "remote" in output and "test" in output
    assert loaded == []
    assert target.service() is service
    assert loaded == ["load"]


def test_name_collisions_require_explicit_replacement(tmp_path):
    first, second = LocalTarget(tmp_path / "first"), LocalTarget(tmp_path / "second")
    register_target("team", first, persist=True)

    with pytest.raises(ValueError, match="already exists"):
        register_target("team", second)

    register_target("team", second, replace=True, persist=True)
    assert load_target("team") == second
    registry._REGISTERED.clear()
    assert load_target("team") == second

    with pytest.raises(ValueError, match="reserved"):
        register_target("local", first)


def test_untrusted_profile_cannot_import_arbitrary_code(tmp_path):
    path = Path(os.environ["BENCHMAX_TARGETS"])
    path.write_text(json.dumps({"schema_version": 1, "targets": {"bad": {"kind": "os.system", "command": "false"}}}))

    with pytest.raises(JobError, match="invalid"):
        load_target("bad")


def test_object_name_and_shortcut_share_one_binding(tmp_path, monkeypatch):
    service = _Service()
    monkeypatch.setattr(LocalTarget, "service", lambda self: service)
    target = LocalTarget(tmp_path / "runs").register("team")
    game = load_game("rock_paper_scissors")
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    job = game.evaluate(policy, submit=True, target=target)
    assert job.run("team") is job
    assert job.run(load_target("team")) is job
    assert len(service.submissions) == 1
    assert Job.open(job.id, target="team").result()["exploitability"] == 0.25
    register_target("team", replace(target, runs=str(tmp_path / "elsewhere")), replace=True)

    with pytest.raises(JobError, match="cannot be rebound"):
        job.run("team")

    assert job.result()["exploitability"] == 0.25
    training = train(game, seed=7, submit=True, target=target)
    assert training.id.startswith("training-")
    assert game.evaluate(policy, submit=True).audit(target).target == "local"

    with pytest.raises(ValueError, match="submit=True"):
        game.evaluate(policy, target=target)


def test_local_output_keeps_the_directory_where_the_job_was_created(tmp_path, monkeypatch):
    service = _Service()
    monkeypatch.setattr(LocalTarget, "service", lambda self: service)
    monkeypatch.chdir(tmp_path)
    job = train(load_game("rock_paper_scissors"), seed=7, submit=True, output="policy.json")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    job.run(LocalTarget(runs=tmp_path / "runs"))

    assert service.submissions[0][1] == tmp_path / "policy.json"


def test_custom_target_interprets_the_requested_output_in_its_namespace(tmp_path, monkeypatch):
    service = _Service()
    monkeypatch.chdir(tmp_path)

    class Destination(Target):
        def service(self):
            return service

        def prepare_output(self, output, *, caller_directory):
            assert output == "policy.json"
            assert caller_directory == tmp_path
            return tmp_path / "destination" / output

    target = Destination()
    job = train(load_game("rock_paper_scissors"), seed=7, submit=True, output="policy.json")
    job.run(target)

    assert service.submissions[0][1] == tmp_path / "destination" / "policy.json"


@pytest.mark.parametrize("fields", [
    {"runs": "relative"}, {"python": "/bad\npath"}, {"host": "-oProxyCommand=oops"}, {"cpus": 0},
    {"gpus": -1}, {"memory_mb": True}, {"poll_seconds": 0}, {"time_limit": "00:00:00"},
    {"time_limit": "00:99:00"}, {"user": "alice"}, {"port": 70000},
])
def test_slurm_configuration_rejects_ambiguous_or_invalid_values(fields):
    args = {"python": "/env/python", "workspace": "/project", "runs": "/runs", **fields}

    with pytest.raises(ValueError):
        SlurmTarget(**args)


@pytest.mark.parametrize("strict_source", ["false", 1])
def test_strict_source_requires_a_boolean(strict_source):
    with pytest.raises(TypeError, match="strict_source"):
        SlurmTarget(python="/env/python", workspace="/project", runs="/runs", strict_source=strict_source)
