# BenchMAX CLI

The command line runs the same two operations as Python:

```text
benchmax train    [domain arguments | --input SPEC] [--output DEST] [--submit | --audit]
benchmax evaluate [domain arguments | --input SPEC] [--output DEST] [--submit | --audit]
```

With no mode flag, BenchMAX runs one `Job` through the selected target, waits,
and prints the policy/result manifest. `--submit` creates and runs the identical
job but prints its durable ID immediately instead of calling `result()`.
`--audit` gathers configuration, provider, and machine evidence without running
the workload. Its 45-second budget is advisory, and slow provider calls are not
interrupted.

`--input` accepts either a versioned workload JSON document or the audited-spec
envelope produced by `--audit --output`. It cannot be mixed with domain arguments.
`train` rejects an evaluation input and vice versa.
`--output` always names the primary durable product, even with `--submit`.

For example:

```console
benchmax train --game phantom_tic_tac_toe --algorithm mappo --seed 7 \
  --audit --output training.json
benchmax train --input training.json
benchmax train --input training.json --submit

benchmax evaluate --game phantom_tic_tac_toe --policy policy.json \
  --output evaluation.json

job_id=$(benchmax evaluate --game phantom_tic_tac_toe \
  --policy policy.json --submit --output evaluation.json)
benchmax job "$job_id"
benchmax job "$job_id" --wait
```

`benchmax job ID` reconnects to the target that owns a submitted job and shows
its current state and output location. `--wait` blocks for the same validated
result document that a foreground command prints. Use `--target` when the job
belongs to a non-default target.

Game and algorithm names refer to registered configurations, so a spec can move
to another process or machine without pickling Python objects. Directly
constructed custom games and arbitrary callable policies still work through the
Python API, but scheduling them requires registering the game and publishing the
policy through a supported safe artifact format.

Run `benchmax train --help` or `benchmax evaluate --help` for all options.

## Policy stores

```sh
benchmax store info
benchmax store show --game kuhn_poker --algorithm cfr
benchmax store show --seed 7 --json
benchmax store info --use local:./policies
benchmax store show --use huggingface:owner/bucket/experiments/run-1
```

Both commands default to the configured reference store (`BENCHMAX_REFERENCE_BUCKET`).
`--use` selects another store: `local:./policies` is relative to the current directory,
`local:/absolute/path` is absolute, and `huggingface:owner/bucket[/prefix]` selects a bucket collection.
Use your saved Hugging Face login or `HF_TOKEN` for authentication.

`info` describes the location and validation requirements without network access. `show` lists one row
per policy across all configurations and seeds, reading metadata without downloading weights. Filters
combine with AND. Displayed identifiers are shortened, but `--json` includes full identifiers and training
configuration. The configuration identifier excludes the replication seed. Both commands support `--json`.

## Package boundary

The CLI only parses shell values, builds or loads the private portable payload,
creates the same `Job` as Python, and calls `audit()`, `run()`, or `result()`.
Validation, seed resolution, canonical digests, resource evidence, persistence,
process management, artifact handling, and scheduler translation live in
`benchmax.execution` or the integration that owns them, and are tested without
the CLI.

The package is split by command:

```text
cli/
  __init__.py    stable main export
  __main__.py    python -m benchmax.cli
  _main.py       root app and function registration only
  _render.py     shared output helpers
  attempts.py    private entry point for local job attempts
  inspect.py     providers/games/algorithms/show
  jobs.py        durable job status and waiting
  setup.py       pons and stockfish installation
  storage.py     store description and policy browsing
  workloads.py   train/evaluate syntax and rendering
  README.md      this file
```

`workloads.py` contains typed command functions and output rendering. Cyclopts
flattens the shared `ExecutionOptions` dataclass into flags and derives
conversions and help from its types. New targets, launchers, resource
heuristics, and policy formats belong in the execution service interfaces, not
in this package.

Below the CLI, `execution/jobs/` owns lifecycle orchestration, `execution/artifacts/` owns durable
files/results, `execution/audit/` owns resource checks, and `execution/targets/` owns destinations and
their services. Shared workload specs live at the execution root. `benchmax.execution.worker` is
the extension API for kernels that need process isolation and shared-memory policy batches.
Provider-specific serialization, such as BenchMARL tensor artifacts, stays with that integration.

Use saved profiles with `--target NAME`, and list profiles and adapter schemes with `benchmax targets`.
See [saved profiles](../execution/README.md#targets-and-saved-profiles) for configuration.

`benchmax show mappo --target university` checks algorithm availability in a saved target without
submitting a job. Slurm inspection covers the login environment, not compute-node GPUs.
