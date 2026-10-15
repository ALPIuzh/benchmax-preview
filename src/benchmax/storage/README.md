# Policy storage

A store collects policies so they can be published, loaded, and reused. BenchMAX provides `reference_store()` to access the reference collection in the Hugging Face bucket [ALPIuzh/benchmax-preview](https://huggingface.co/buckets/ALPIuzh/benchmax-preview).

A `Reference` identifies a policy without loading it. See [References](../execution/jobs/reference.py) for loading and saving policies across stores, jobs, and local files.

## Publish and use a policy

```python
from benchmax.storage import HuggingFaceStore, LocalStore

store = LocalStore(".benchmax-data")  # or HuggingFaceStore("user/my-policies")
reference = store.publish_policy(policy)  # policy returned by train()
restored = reference.result()
local = reference.save(path="saved-policy")
metrics = game.evaluate(reference)
job = game.evaluate(reference, submit=True).run()
```

Publication also accepts an artifact path or policy reference. Training records producer versions,
source information, and configuration, and copying or republishing preserves them. Arbitrary Python
callables cannot be stored. `save()` requires a new destination directory.

For terminal browsing, use `benchmax store info` and `benchmax store show`. See the
[CLI guide](../cli/README.md#policy-stores) for filters and store selection.

## Find policies

```python
references = store.find_policies(game, algorithm="mappo")
seeds = [reference.training.seed for reference in references]
policy = store.load_policy(game, algorithm="mappo", seed=7)

# Include every training configuration, or select a specific one.
all_mappo = store.find_policies(game, algorithm="mappo", training_config="any")
custom = store.find_policies(game, algorithm="mappo", training_config={"total_frames": 200})
```

`game_config` and `training_config` accept `"default"`, `"any"`, or an overrides mapping:

- `"default"` uses registered defaults for names and preserves settings supplied by configured objects.
- `"any"` matches every configuration of the supplied name. Omitted names leave that dimension unrestricted.
- A mapping resolves overrides against defaults and matches the complete configuration, not a subset.
  Game overrides require a game name, and training overrides follow `train()`.

Filters combine with AND. Omit `seed` to include all matching seeds. Evaluator changes do not affect
selection. `find_policies()` reads metadata without downloading weights, while `load_policy()` requires
exactly one match and raises `PolicyNotFoundError` or `AmbiguousPolicyError` otherwise. A seed alone
may still match multiple publications.

## Shared stores

Hugging Face support is included in the standard installation. Create the bucket beforehand and
use `hf auth login` or `HF_TOKEN` to authenticate. Reads require read access; publication requires
write access. Use `prefix="experiments/run-1"` to isolate a collection within a bucket.

References round-trip through `to_document()` and `Reference.from_document()`. Credentials are
never serialized: each process uses its own login or token. Local-store paths must be accessible
on the executing host. Submitted evaluations pin stored policies into their run directory, and only
references to training jobs create job dependencies.

## Reference store

`reference_store()` opens the configured reference collection through the same API as other stores,
with strict metadata requirements. It defaults to the public `ALPIuzh/benchmax-preview` bucket.
Set `BENCHMAX_REFERENCE_BUCKET` to override the configured bucket.

```python
from benchmax.storage import reference_store

store = reference_store()
references = store.find_policies(game, algorithm="mappo")
```

Each result is a policy reference: inspect its training metadata, load it with `reference.result()`,
or pass it to `game.evaluate(reference)`. To load the pinned expert for a game, use `load_expert(game)`.

## Validation and reproducibility

- Artifact integrity is checked on loading. `validate_policy()` checks validity without loading a model,
  and `STRICT_REQUIREMENTS` additionally requires resolved configurations and producer provenance.
- Legacy artifacts without producer metadata need an explicit `release=` when published.
- Bucket artifacts use content hashes and verified local caching. Exact cached references can work
  offline, but discovery requires the network. Retain a local copy if remote deletion is a concern.
- `snapshot()` freezes, in memory, the discovery entries observed by that store instance.

See [validation requirements](requirements.py) for custom checks and the
[opt-in live test](../../../tests/storage/test_live.py) for testing a development bucket.
