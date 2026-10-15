"""Real local subprocess execution behind a simulated scheduler; no cluster I/O."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from benchmax import load_game
from benchmax.execution import Job, SlurmTarget
from benchmax.training import train
from tests.training_helpers import tiny_training_params


def test_slurm_control_and_attempt_processes_train_and_evaluate(tmp_path, monkeypatch):
    pytest.importorskip("benchmarl")
    monkeypatch.setenv("BENCHMAX_ARTIFACTS", str(tmp_path / "download-cache"))
    scheduler = tmp_path / "scheduler"
    scheduler.mkdir()
    # This local test dispatcher emulates afterok without contacting Slurm.
    (scheduler / "dispatch.py").write_text(f'''import json
import os
import pathlib
import subprocess
import sys
import time

root = pathlib.Path({str(scheduler)!r})
job_id, script, dependency = sys.argv[1:]
deadline = time.monotonic() + 30
while True:
    if dependency:
        record = root / (dependency + '.json')
        state = 'pending'
        if record.exists():
            run = pathlib.Path(json.loads(record.read_text())['run'])
            state = json.loads((run / 'state.json').read_text())['state']
        if state == 'succeeded':
            break
        if state in ('failed', 'cancelled'):
            sys.exit(1)
    elif (root / 'release-training').exists():
        break
    if time.monotonic() >= deadline:
        sys.exit(1)
    time.sleep(0.02)
sys.exit(subprocess.call(['sh', script], env={{**os.environ, 'SLURM_JOB_ID': job_id}}))
''')
    # This executable only launches a tiny local subprocess. It is not Slurm.
    (scheduler / "sbatch").write_text(f'''#!{sys.executable}
import json
import os
import pathlib
import subprocess
import sys
import time

job_id = str(time.time_ns())
script = pathlib.Path(sys.argv[-1])
log = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--output='))
dependency = next((arg.split(':', 1)[1] for arg in sys.argv if arg.startswith('--dependency=afterok:')), '')
with open(log, 'w') as stream:
    process = subprocess.Popen([sys.executable, {str(scheduler / 'dispatch.py')!r}, job_id, str(script), dependency],
                               stdout=stream, stderr=stream, start_new_session=True)
pathlib.Path({str(scheduler)!r}, job_id + '.json').write_text(json.dumps({{'run': str(script.parent)}}))
pathlib.Path({str(scheduler)!r}, job_id + '.argv.json').write_text(json.dumps(sys.argv[1:]))
print(job_id)
''')
    (scheduler / "sacct").write_text(f'''#!{sys.executable}
import json
import pathlib
import sys

job_id = sys.argv[sys.argv.index('-j') + 1]
run = pathlib.Path(json.loads(pathlib.Path({str(scheduler)!r}, job_id + '.json').read_text())['run'])
state = json.loads((run / 'state.json').read_text())['state']
print(job_id + '|' + {{'succeeded': 'COMPLETED', 'failed': 'FAILED'}}.get(state, 'RUNNING') + '|0:0')
''')

    for executable in scheduler.iterdir():
        executable.chmod(0o700)

    import os
    monkeypatch.setenv("PATH", str(scheduler) + os.pathsep + os.environ["PATH"])
    target = SlurmTarget(python=sys.executable, workspace=str(tmp_path), runs=str(tmp_path / "runs"),
                         artifacts=str(tmp_path / "policies"), poll_seconds=0.1)
    game = load_game("rock_paper_scissors")
    job = train(game, seed=7, device="cpu", total_frames=8, submit=True, params=tiny_training_params()).run(target)
    reference = job.reference()
    evaluation = game.evaluate(reference, submit=True).run(target)
    import json

    commands = [json.loads(path.read_text()) for path in scheduler.glob("*.argv.json")]
    assert sum(any(arg.startswith("--dependency=afterok:") for arg in argv) for argv in commands) == 1
    (scheduler / "release-training").touch()
    actual = evaluation.result(timeout=30)
    saved = reference.save(timeout=30, path=tmp_path / "downloaded")
    policy = saved.result()
    download = tmp_path / "downloaded/manifest.json"
    assert download.is_file()
    expected = game.evaluate(policy)
    assert game.evaluate(download)["exploitability"] == pytest.approx(expected["exploitability"])
    assert game.evaluate(saved)["exploitability"] == pytest.approx(expected["exploitability"])
    assert actual["exploitability"] == pytest.approx(expected["exploitability"])
    assert game.evaluate(reference)["exploitability"] == pytest.approx(expected["exploitability"])
    assert Job.open(evaluation.id, target=target).result() == actual
    assert job.output.parent.parent == tmp_path / "policies"

    from benchmax.storage import LocalStore

    stored = LocalStore(tmp_path / "store").publish_policy(policy)
    stored_job = game.evaluate(stored, submit=True)
    assert stored_job.progress().dependencies == ()
    stored_job.run(target)
    assert stored_job.result(timeout=30)["exploitability"] == pytest.approx(expected["exploitability"])
    resolved = json.loads((tmp_path / "runs" / stored_job.id / "resolved-spec.json").read_text())
    assert resolved["policy"]["digest"]
    assert Path(resolved["policy"]["uri"]).is_file()


def test_local_target_processes_wait_for_training_and_evaluate_saved_references(tmp_path, monkeypatch):
    import json
    import shutil
    import subprocess
    import time

    import numpy as np

    from benchmax.execution import LocalTarget
    from benchmax.execution.artifacts.policies import ArtifactError
    from benchmax.execution.targets.local import LocalJobService

    pytest.importorskip("benchmarl")
    monkeypatch.setenv("BENCHMAX_ARTIFACTS", str(tmp_path / "artifacts"))
    gate = tmp_path / "release-training"
    dispatcher = tmp_path / "dispatch.py"
    dispatcher.write_text(f'''import os
import pathlib
import sys
import time

deadline = time.monotonic() + 30
while not pathlib.Path({str(gate)!r}).exists():
    if time.monotonic() >= deadline:
        sys.exit(1)
    time.sleep(0.02)
os.execv(sys.argv[1], sys.argv[1:])
''')
    children = []

    def launch(command, environment, log):
        if command[-1].startswith("training-"):
            command = [sys.executable, str(dispatcher), *command]

        with log.open("wb") as stream:
            process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=stream, start_new_session=True)

        children.append(process)
        return process.pid

    target = LocalTarget(runs=str(tmp_path / "runs"))
    monkeypatch.setattr(LocalTarget, "service", lambda self: LocalJobService(self.runs, launcher=launch))
    game = load_game("rock_paper_scissors")
    training_output = tmp_path / "training.json"

    try:
        training = train(game, seed=7, device="cpu", total_frames=8, submit=True,
                         params=tiny_training_params(), output=training_output).run(target)
        evaluation = game.evaluate(training.reference(), submit=True, output=tmp_path / "evaluation.json").run(target)
        deadline = time.monotonic() + 10

        while evaluation.state == "pending" and time.monotonic() < deadline:
            time.sleep(0.02)

        assert evaluation.state == "running" and training.state == "pending"
        assert evaluation.progress().phase == "waiting_for_policy"
        run = tmp_path / "runs" / evaluation.id
        assert not (run / "resolved-spec.json").exists()
        gate.touch()
        metrics = evaluation.result(timeout=30)
        assert json.loads((run / "spec.json").read_text())["policy"]["kind"] == "job-output"
        assert "digest" in json.loads((run / "resolved-spec.json").read_text())["policy"]
        saved = training.save(path=tmp_path / "saved")
        expected = game.evaluate(saved)
        assert metrics == expected
        assert Job.open(evaluation.id, target=target).result() == metrics
        manifest = Path(json.loads(training_output.read_text())["policy"]["uri"])
        assert manifest.parent.parent == tmp_path

        with pytest.raises(ArtifactError, match="incompatible with game"):
            load_game("kuhn_poker").evaluate(training.reference())

        with monkeypatch.context() as guard:
            guard.setattr(np, "load", lambda *a, **kw: pytest.fail("saving must not load tensor arrays"))
            copied = saved.save(tmp_path / "copied-policy")

        local_evaluation = game.evaluate(saved, submit=True).run(target)
        assert local_evaluation.result(timeout=30)["exploitability"] == pytest.approx(expected["exploitability"])
        assert not (tmp_path / "runs" / local_evaluation.id / "resolved-spec.json").exists()
    finally:
        gate.touch(exist_ok=True)

        for process in children:
            if process.poll() is None:
                process.terminate()

            process.wait(timeout=5)

    shutil.rmtree(tmp_path / "runs")
    assert game.evaluate(copied) == metrics
    assert game.evaluate(training_output) == metrics
