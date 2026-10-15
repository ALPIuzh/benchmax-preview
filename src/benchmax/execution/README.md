# BenchMAX execution

`train(..., submit=True)` and `game.evaluate(..., submit=True)` create jobs.
`run()` submits a job to its target, and `result()` waits and returns the policy or metrics.

```python
from benchmax import load_game
from benchmax.execution import LocalTarget, Target
from benchmax.training import train

target = LocalTarget(runs="./research-runs")
game = load_game("rock_paper_scissors")
training = train(game, "mappo", seed=7, submit=True)
training.run(target)
policy = training.result()

evaluation = game.evaluate(policy, submit=True).run(target)
metrics = evaluation.result()
```

Use `await job.result_async()` for asynchronous waiting.
Import `Job` when reconnecting to a submitted job from another session:

```python
from benchmax.execution import Job

reopened = Job.open(training.id, target=target)
```

Jobs capture training settings and caller-local input paths at creation.
Remote output paths belong to the target's filesystem.

`job.cancel()` requests cancellation. Cancelling a producer prevents dependent jobs from running.
Cancellation does not create a checkpoint or remove partial outputs.

## Progress

`job.progress()` returns a snapshot of the job and its dependencies. Display it in Jupyter or print it:

```python
progress = evaluation.progress()
print(progress)
progress.state, progress.phase, progress.reason
```

Use `progress.to_document()` for structured data. Call `progress()` again to refresh.
Slurm refreshes contact the cluster. Warnings may accompany the last known state,
and completion percentages and ETA appear only when reported.

## References and saved results

A `Job[T]` manages execution. A `Reference[T]` identifies its output or saved local files.
Training returns a `PolicyProfile`; evaluation returns a metrics dictionary.

### Load a value or save its files

Jobs and references offer the same methods:

| Method | Returns | Effect |
| --- | --- | --- |
| `.result()` | `T` | Wait if needed, then load the value |
| `.save(path="./saved")` | `Reference[T]` | Save the output files without loading a model |
| `.result(path="./saved")` | `T` | Save the files, then load the value |

```python
reference = training.reference()
local_reference = reference.save(path="./policy")
policy = local_reference.result()
print(local_reference.path)
```

Destinations must be new local directories. Saved files remain usable after the job records are removed.
Loading requires the policy’s dependencies. Saving does not include training checkpoints.

`.save()` and `.result()` accept `timeout=` for waiting. A timeout stops waiting without cancelling
the producer and does not limit later transfers or model loading. Failed or cancelled jobs raise an error.
`await job.result_async(...)` also supports `path=`.

### Save a policy directly

```python
reference = policy.save("./policy")
restored = reference.result()
metrics = game.evaluate(reference)
```

For custom finite policies, use `PolicyProfile.from_table(table)` and pass `game=game` on the first save.
Custom callables work in memory, but saving requires a supported representation.

### Reuse a policy in evaluation

Pass a policy reference explicitly, since a `Job` or a metrics reference is not a policy:

```python
evaluation = game.evaluate(training.reference(), submit=True)
evaluation.run(target)
metrics = evaluation.result()
```

Evaluation can be submitted before training finishes. It waits for a successful producer and does not
compute after a failed or cancelled producer. Dependent jobs must use the same target endpoint and run store,
but their resource requests may differ.

`game.evaluate(reference)` loads and evaluates the policy on a compatible game. The evaluator may differ.

### Keep a reference for another session

`job.reference()` requires a submitted job.
Use [`job.progress()`](#progress) to check execution status.

```python
import json
from pathlib import Path
from benchmax import PolicyProfile, Reference

Path("reference.json").write_text(json.dumps(reference.to_document()), encoding="utf-8")
restored = Reference[PolicyProfile].from_document(
    json.loads(Path("reference.json").read_text(encoding="utf-8"))
)
policy = restored.result()

saved = Reference[PolicyProfile].from_path("./policy/manifest.json")
```

Serializing a reference saves its location and identity, not its result files.
A job reference needs the original target and run records, while a file reference needs its recorded local files.
Moving to another computer requires transferring those files separately.

`from_path()` reads and pins an artifact document without loading a model. It accepts a policy manifest or
training-result document. Use `Reference[dict].from_path("./metrics/result.json")` for saved metrics.
Resolving a job reference reconnects to its captured target, even if its saved profile name has since changed.
Retain a loaded value or saved file reference when repeatedly using a result locally.

## Targets and saved profiles

Register a name to reuse a target in Python or the CLI:

```python
target.register("research", persist=True)
target = Target.load("research")
reopened = Job.open(training.id, target=target)
```

`run()`, `audit()`, `Job.open()`, and the `target=` shortcuts accept a target object or its name.

- Without `persist=True`, a name lasts for the current Python process.
- Saved profiles use `$XDG_CONFIG_HOME/benchmax/targets.json` (default `~/.config/benchmax/targets.json`).
  Set `BENCHMAX_TARGETS` to use another file.
- Replacing a name requires `replace=True`, and `local` is reserved.
- Submitted jobs keep their original target configuration. Reopening by name uses the current profile.
- Relative local output paths use the directory where the job was created.

`register_target()` and `load_target()` are equivalent function-based helpers.
`AdapterTarget` supports URIs provided by installed target adapters.
In the CLI, use `--target NAME`, and list available profiles and adapter schemes with `benchmax targets`.

Use `benchmax show mappo --target university` to check algorithm availability in a saved target.
This requires no training dependencies on the client and submits no job. Slurm inspection checks
the login environment, not compute-node GPU availability.

### Slurm execution

Install BenchMAX on the client and cluster, including the `training` extra on the cluster
for BenchMARL algorithms. See [build requirements](../../../docs/getting-started/installation.md#build-requirements).
Submission uses this existing installation.

By default, Slurm checks the selected game and training configuration, but allows and records source differences.
Set `strict_source=True` on `SlurmTarget` to require identical BenchMAX Python/configuration source on the client
and cluster, including when a queued job starts.

SSH must authenticate without prompts. For password-based access, keep an authenticated connection open
in a local terminal:

```sh
mkdir -p -m 700 "$HOME/.ssh/benchmax-sockets"
ssh -M -S "$HOME/.ssh/benchmax-sockets/%C" -N alice@cluster.example.edu
```

Replace the login, allocation, and paths below with your cluster settings. With key/agent authentication,
omit `control_path`. To submit directly from the cluster, omit all SSH fields.

```python
from pathlib import Path
from benchmax.execution import SlurmTarget

cluster = SlurmTarget(
    host="cluster.example.edu", user="alice",
    control_path=str(Path.home() / ".ssh/benchmax-sockets/%C"),
    python="/shares/team/benchmax/.venv/bin/python", workspace="/shares/team/benchmax",
    runs="/scratch/alice/benchmax/runs", artifacts="/shares/team/benchmax/artifacts",
    account="team", partition="standard", cpus=1, memory_mb=2048, time_limit="00:10:00", gpus=0,
)
cluster.register("university", persist=True)
training = train(game, "mappo", seed=7, device="cpu", total_frames=6_000, submit=True).run(cluster)
evaluation = game.evaluate(training.reference(), submit=True).run(cluster)
metrics = evaluation.result()
```

Keep `runs` for reconnection and `artifacts` for saved results. All paths except `control_path` belong to
the cluster. Relative outputs use `artifacts`, falling back to `workspace`. Use new output paths.
`job.audit(cluster)` checks the installation without submitting work. Waiting polls every 300 seconds by
default, and `poll_seconds` changes the interval.

Reuse policies through [references](#reuse-a-policy-in-evaluation). `training.save(path="./policy")` downloads
files without loading the model, and `training.result()` also loads it and requires the training dependencies locally.
Policy uploads and downloads default to a 64 MB limit (`transfer_limit_mb`). Evaluating a portable local policy
on an SSH target automatically uploads its validated manifest and tensor files. The cluster must already have
a compatible BenchMAX/provider installation, since custom source code is not transferred.

```python
evaluation = game.evaluate(local_policy, submit=True).run(cluster)
```

Saved policy paths and file references work too. Uploads use a content-addressed `policy-cache` directory under
`artifacts` (or `runs` when `artifacts` is unset), and repeated submissions reuse validated cached files. Failed
uploads never publish a partial cache entry. `job.audit(cluster)` reports required policy bytes or cache reuse
without uploading. Cross-cluster transfers and dependencies on unfinished local training jobs remain unsupported.

If submission has an uncertain outcome, reopen and inspect the job before retrying. Timeouts and SSH
disconnections do not cancel submitted jobs. Automatic retries and checkpoint/resume are not supported.

See the [Slurm demo notebook](../../../notebooks/slurm_demo.ipynb) for a complete example.

## Package layout

| Location | Responsibility |
| --- | --- |
| `operation.py` | Shared routing decorator for `train()` and `Game.evaluate()` |
| `jobs/` | Job lifecycle, input preparation, references, synchronous outputs, and progress |
| `artifacts/` | Policy artifact files and durable result documents |
| `audit/` | Resource checks and provider-supplied execution evidence |
| `targets/` | Passive target definitions, service contracts, discovery, and local/Slurm backends |
| `worker/` | Evaluation kernels, policy oracles, prepared resources, and isolated processes |
| `specs.py`, `values.py` | Workload and reference descriptions, and portable values shared by the packages |
