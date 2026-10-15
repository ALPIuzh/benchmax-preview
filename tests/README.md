# Finding and running tests

Start with the subsystem you are changing. Each test file covers one component. Exhaustive input checks live with that component, and its callers only test that they use it.

| Directory | What belongs here |
| --- | --- |
| `configuration/` | Named entries, overrides, registry discovery, provider availability |
| `games/` | Core environment contracts, game rules, state encodings and layouts |
| `policy/` | Callable adapters, probabilities, data-only artifact validation, model loading |
| `evaluation/` | Numerical metrics and evaluator compatibility |
| `training/` | Public training API and algorithm selection |
| `execution/jobs/` | Workload identity, durable jobs, output ownership, resource audit, persisted policies |
| `execution/targets/` | Target inspection and execution environments |
| `execution/worker/` | Dispatch, prepared resources, IPC, shared arrays, process lifecycle |
| `integrations/` | OpenSpiel, BenchMARL and native binding boundaries |
| `storage/` | Policy store contracts, local persistence, Hub SDK and live checks |
| `cli/` | Discovery, workload commands and job commands |
| `tooling/` | Read-only notebook checks and normalization examples |
| `test_seeding.py` | Seed derivation, entropy rules and independent environment streams |
| `test_public_api.py` | Supported imports |

Run a focused file or directory first:

```sh
uv run pytest tests/games -q
uv run pytest tests/execution/worker/test_dispatch.py -q
uv run --extra training pytest -q
uv run ruff check src tests
```

The base suite includes storage tests and skips tests requiring optional training dependencies. CI checks base Python 3.11/3.12 and training environments and verifies which packages are installed. Native binding/process tests must keep their clean-process isolation, so do not move native imports into global fixtures to save startup time.

## Shared inputs

- `conftest.py`: one lazily trained tiny policy used by artifact and store integration tests. Treat it as read-only. Tests needing independent training (especially reproducibility) train their own policies.
- `artifact_helpers.py`: a minimal valid artifact with no training dependencies. Mutations stay visible in the test.
- `training_helpers.py`: a fresh tiny training configuration on every call.
- `process_helpers.py`: assertions in a fresh interpreter, with captured diagnostics and explicit timeouts.
- `execution/jobs/conftest.py`: an unlaunched local service rooted at `tmp_path / "runs"`.
- `execution/worker/_kernels.py`: importable subprocess test kernels.

Keep helpers small and close to the tests that use them. Prefer a parametrized table of inputs and expected outputs over repeated test bodies, and test behaviour rather than private implementation details.

Storage contracts run against local and fake-Hub backends through `storage/conftest.py`. Live Hub tests remain opt-in (see `src/benchmax/storage/README.md`).
