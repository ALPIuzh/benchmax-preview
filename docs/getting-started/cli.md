# Command line

The `benchmax` command is a shell interface to the Python API, so each command runs the same operation as its Python
equivalent and produces the same results. Below, you will find an overview of all commands, each with its Python
equivalent and a link to the general documentation.

The command is installed together with BenchMAX. Run it inside the project environment, either as `uv run benchmax`
or as `benchmax` after activating the environment. Commands that print results also accept `--json` for
machine-readable output, which is useful in scripts.

Every command documents its options. `benchmax --help` lists all commands, and `--help` after a command lists that
command's options with their defaults:

```sh
benchmax --help
benchmax train --help
```

## Overview

| Command | Python equivalent | Guide |
| --- | --- | --- |
| `benchmax providers` | `list_providers()` | [Games and evaluators](../reference/registry.md) |
| `benchmax games` | `list_available_games()` | [Games and evaluators](../reference/registry.md) |
| `benchmax algorithms` | `list_available_algorithms()` | [Training](../reference/training.md) |
| `benchmax show NAME` | `load_game(NAME).description` | [Configuration](../reference/config.md) |
| `benchmax train` | `train(game, algorithm, ...)` | [Training](../reference/training.md) |
| `benchmax evaluate` | `game.evaluate(policy)` | [Evaluation](../reference/evaluation.md) |
| `--submit`, `benchmax job ID` | `submit=True`, `Job.open(ID)` | [Jobs](../reference/execution.md) |
| `--audit` | `job.audit(target)` | [Jobs](../reference/execution.md) |
| `--target NAME`, `benchmax targets` | `.run(NAME)`, `list_available_targets()` | [Targets](../reference/execution.md) |
| `benchmax store info`, `store show` | `store.requirements`, `store.find_policies(...)` | [Policy storage](https://github.com/ALPIuzh/benchmax-preview/blob/main/src/benchmax/storage/README.md) |
| `benchmax setup pons`, `setup stockfish` | `install()` in `benchmax.integrations.pons` and `.stockfish` | [Evaluation](../reference/evaluation.md) |

## Inspect the installation

These commands show what `list_providers()`, `list_available_games()` and `list_available_algorithms()` return
in Python. [Games and evaluators](../reference/registry.md) and
[configuration](../reference/config.md) explain the names they list.

```sh
benchmax providers          # installed providers, and the extra that adds a missing one
benchmax games              # configured games and their default evaluators; --all adds provider games
benchmax algorithms         # configured algorithms and presets; --all adds provider algorithms
benchmax show kuhn_poker    # one game, algorithm or evaluator
benchmax --version
```

## Train and evaluate

The commands below are the CLI equivalents of the Python `train()` and `game.evaluate()` calls. For further
information, see [training](../reference/training.md) and
[evaluation](../reference/evaluation.md).

```sh
benchmax train --game rock_paper_scissors --algorithm mappo --total-frames 6000 --device cpu --seed 7 \
  --output training.json
benchmax evaluate --game rock_paper_scissors --policy training.json --output evaluation.json
```

`train` takes `--game`, `--algorithm` (default `mappo`), `--seed`, `--device` and `--total-frames`. `evaluate` takes
`--game` and `--policy`, a training result or policy manifest. Both wait, then print the result as JSON and write it
to `--output`, which must be a new path. Pipe the output to `jq .metrics` to print only the metrics.

The CLI trains MAPPO and IPPO, and evaluates with the game's default evaluator. For CFR, other evaluators or
algorithm-specific settings, use Python.

## Run in the background

`--submit` works like `submit=True` in Python, and `benchmax job` reopens a job like `Job.open()`. The
[jobs guide](../reference/execution.md) describes a job's lifecycle.

```sh
job_id=$(benchmax evaluate --game rock_paper_scissors --policy training.json --submit --output evaluation-2.json)
benchmax job "$job_id"          # state and output location
benchmax job "$job_id" --wait   # wait, then print the result
```

`--submit` prints the job ID instead of waiting. The job keeps running after the terminal closes.

## Check resources before running

`--audit` runs the same resource check as `job.audit(target)` in Python. See
[jobs](../reference/execution.md) for what the report contains.

```sh
benchmax train --game rock_paper_scissors --algorithm mappo --total-frames 6000 --device cpu --seed 7 \
  --audit --output audited-training.json
benchmax train --input audited-training.json --submit
```

`--audit` saves the checked workload to `--output`, and `--input` runs it later. `--input` replaces the workload
options such as `--game`, and cannot be combined with them. `--submit` and `--audit` exclude each other.

## Run on a Slurm cluster

Cluster jobs run on a `SlurmTarget`, which you create and save in Python as shown in
[targets](../reference/execution.md). Then pass its name to any workload command, and again when reconnecting:

```sh
benchmax targets                                  # saved targets
benchmax show mappo --target university           # check that the algorithm is installed there
job_id=$(benchmax train --game kuhn_poker --algorithm mappo --target university --submit)
benchmax job "$job_id" --target university --wait
```

## Browse policy stores

In Python, a store is a `LocalStore` or `HuggingFaceStore`, searched with `store.find_policies()`. The
[policy storage guide](https://github.com/ALPIuzh/benchmax-preview/blob/main/src/benchmax/storage/README.md) covers publishing and loading. From the shell, you can browse a store:

```sh
benchmax store show --use local:./policies --game rock_paper_scissors --algorithm mappo --seed 7
benchmax store show --use huggingface:alice/policies/experiments/run-1
benchmax store info --use huggingface:alice/policies   # describe the store without contacting it
```

```text
POLICY        GAME                 ALGORITHM  SEED  CONFIG
ea89e61e342e  rock_paper_scissors  mappo      7     75d4e80dd19c
```

`--use` takes `local:PATH` or `huggingface:OWNER/BUCKET[/PREFIX]`. Without it, both commands read the reference
store set by `BENCHMAX_REFERENCE_BUCKET`. The commands read metadata only, so publish and load policies in Python.

### Hugging Face buckets

Set up a bucket with the `hf` command, which comes with BenchMAX:

```sh
hf auth login                                    # or set HF_TOKEN
hf buckets create alice/policies --private       # once per bucket
benchmax store show --use huggingface:alice/policies/experiments/run-1
```

Reading needs read access and publishing needs write access. Publish policies to the bucket from Python with
`HuggingFaceStore("alice/policies", prefix="experiments/run-1")`, as shown in the
[policy storage guide](https://github.com/ALPIuzh/benchmax-preview/blob/main/src/benchmax/storage/README.md).

## Install reference opponents

These commands call the same `install()` functions as `benchmax.integrations.pons.install` and
`benchmax.integrations.stockfish.install` in Python. [`OpponentEvaluator`](../reference/evaluation.md) describes the
evaluators that use them.

```sh
benchmax setup pons --rows 6 --columns 7   # Connect Four solver, built for one board size
benchmax setup stockfish                   # Stockfish chess engine
```

The `pons` and `stockfish` evaluators need these programs, and evaluation never downloads them. `setup pons` needs
`make` and a C++ compiler, while `setup stockfish` supports macOS and x86-64 Linux. Both accept `--directory` and
`--timeout`, and print the installed executable.

## Files

`--output` matches the `output=` argument of `train()` and `game.evaluate()` in Python, and writes:

| Path | Written by | Contents |
| --- | --- | --- |
| `--output` of `train` | `train` | Training result, next to a `*.policy-*/` directory with the model |
| `--output` of `evaluate` | `evaluate` | Metrics and provenance |
| `--output` with `--audit` | `train`, `evaluate` | Checked workload, for `--input` |

Job records, saved targets and caches use the default locations listed in
[references and storage](../reference/storage.md).
