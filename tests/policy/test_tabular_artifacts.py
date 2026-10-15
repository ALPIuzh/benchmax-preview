"""Finite policies use the existing durable lifecycle without neural dependencies."""

import hashlib
import json
import shutil

import numpy as np
import pytest

from benchmax import InvalidPolicyError, PolicyProfile, Reference, load_game
from benchmax.execution import Job, LocalTarget
from benchmax.execution.artifacts.policies import ArtifactError, load_policy, validate_policy_artifact
from benchmax.execution.artifacts.tabular import POLICY_FORMAT
from benchmax.execution.specs import _ArtifactMetadata
from benchmax.execution.targets.slurm.transfer import pack_policy, unpack_policy
from benchmax.training import train
from tests.artifact_helpers import forbid_neural_imports


@pytest.mark.parametrize("algorithm", ["cfr", "cfr_plus"])
def test_cfr_output_reload_and_export_without_neural_imports(tmp_path, monkeypatch, algorithm):
    forbid_neural_imports(monkeypatch, "tabular artifacts")
    game = load_game("kuhn_poker")
    output = tmp_path / "training.json"
    policy = train(game, algorithm, iterations=100, seed=7, output=output)
    expected = game.evaluate(policy)
    reference = Reference.from_path(output)
    path, manifest, _ = validate_policy_artifact(reference._artifact_metadata())
    assert manifest["format"] == POLICY_FORMAT
    assert game.evaluate(reference) == expected
    assert game.evaluate(reference.result()) == expected
    saved = policy.save(tmp_path / "exported")
    assert game.evaluate(saved.result()) == expected
    assert path.is_file()

    with monkeypatch.context() as guard:
        guard.setattr(np, "load", lambda *a, **kw: pytest.fail("copying must not load arrays"))
        copied = saved.save(tmp_path / "copied")
        validate_policy_artifact(copied._artifact_metadata())

    assert game.evaluate(copied) == expected


def test_cfr_submitted_lifecycle_and_transfer(tmp_path):
    target = LocalTarget(runs=str(tmp_path / "runs"))
    game = load_game("kuhn_poker")
    training = train(game, "cfr", iterations=100, seed=9, submit=True).run(target)
    evaluation = game.evaluate(training.reference(), submit=True).run(target)
    metrics = evaluation.result(timeout=30)
    exported = training.save(tmp_path / "exported", timeout=30)
    assert Job.open(evaluation.id, target=target).result() == metrics
    assert game.evaluate(exported) == metrics

    metadata = exported._artifact_metadata()
    bundle = pack_policy(metadata, limit=1024**2)
    copied = unpack_policy(bundle, tmp_path / "transferred", digest=metadata.digest, limit=1024**2)
    shutil.rmtree(tmp_path / "runs")
    assert game.evaluate(Reference.from_path(copied.uri)) == metrics


def test_explicit_custom_table_save_reload_and_evaluation_job(tmp_path):
    game = load_game("rock_paper_scissors")
    step = game.new_env().reset(seed=0)
    table = {(p, step.observations[p], step.legal_actions[p]): {a: 1 / 3 for a in game.action_spaces[p]}
             for p in game.players}
    policy = PolicyProfile.from_table(table, fallback="uniform")
    reference = policy.save(tmp_path / "table", game=game)
    table.clear()
    expected = game.evaluate(policy)
    assert expected["nash_conv"] == 0
    assert game.evaluate(reference) == expected
    assert game.evaluate(reference.result().save(tmp_path / "copied")) == expected

    target = LocalTarget(runs=str(tmp_path / "runs"))
    assert game.evaluate(policy, submit=True).run(target).result(timeout=30) == expected
    unknown = np.full((1, *game.observation_spaces[0]), 42.)
    mask = np.array([[True, False, True]])
    restored = reference.result()
    np.testing.assert_array_equal(restored(0, unknown, game.action_spaces[0], mask), [[.5, 0, .5]])


def test_tabular_game_tables_are_snapshots_and_portable(tmp_path):
    game = load_game("gridworld_3x3")
    tables = {p: np.full((game.num_states, len(game.action_spaces[p])), .25) for p in game.players}
    policy = game.tabular_profile(tables)
    tables[0][:] = 0
    reference = policy.save(tmp_path / "grid", game=game)
    assert game.evaluate(reference) == game.evaluate(policy)
    assert game.evaluate(policy, output=tmp_path / "metrics.json") == game.evaluate(policy)


def test_table_query_preserves_noncontiguous_actions_masks_and_input_copy():
    distribution = {7: .75, 2: .25}
    policy = PolicyProfile.from_table({(3, (1., 0.), (7, 2)): distribution})
    distribution[7] = 0
    probabilities = policy(3, np.array([[1., 0.]]), (7, 11, 2), np.array([[True, False, True]]))
    np.testing.assert_array_equal(probabilities, [[.75, 0., .25]])
    with pytest.raises(KeyError, match="not materialized"):
        policy(3, np.array([[0., 1.]]), (7, 11, 2), np.array([[True, False, True]]))


@pytest.mark.parametrize("distribution", [{0: 1.1, 1: -.1}, {0: float("nan"), 1: 0.}, {0: .2, 1: .2}])
def test_invalid_table_probabilities_are_rejected(distribution):
    with pytest.raises(InvalidPolicyError):
        PolicyProfile.from_table({(0, (0.,), (0, 1)): distribution})


@pytest.fixture
def saved_table(tmp_path):
    game = load_game("kuhn_poker")
    reference = train(game, "cfr", iterations=10, seed=0).save(tmp_path / "table")
    path = reference.path
    return game, path, json.loads(path.read_text())


def test_table_inspection_checks_bounds_before_loading(saved_table, monkeypatch):
    _, path, _ = saved_table
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("inspection must not load arrays"))
    validate_policy_artifact(str(path))
    with pytest.raises(ArtifactError, match="load limit"):
        validate_policy_artifact(str(path), max_tensor_bytes=1)


@pytest.mark.parametrize("damage", ["checksum", "dtype", "shape", "path", "fallback", "missing_array"])
def test_invalid_table_metadata_is_rejected(saved_table, damage):
    _, path, manifest = saved_table
    if damage == "checksum":
        manifest["tensor_sha256"] = "0" * 64
    elif damage == "dtype":
        manifest["tensors"][0]["dtype"] = "|O"
    elif damage == "shape":
        manifest["tensors"][0]["shape"][1] += 1
    elif damage == "path":
        manifest["tensor_file"] = "../tensors.npz"
    elif damage == "fallback":
        manifest["fallback"] = "execute"
    else:
        manifest["tensors"].pop()

    path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError):
        validate_policy_artifact(str(path))


@pytest.mark.parametrize("damage", ["nan_observation", "nan_probability", "negative", "illegal", "duplicate"])
def test_invalid_table_payload_is_rejected(saved_table, damage):
    _, path, manifest = saved_table
    tensor_path = path.parent / manifest["tensor_file"]
    with np.load(tensor_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    if damage == "nan_observation":
        arrays["tensor_0"][0, 0] = np.nan
    elif damage == "nan_probability":
        arrays["tensor_2"][0, 0] = np.nan
    elif damage == "negative":
        arrays["tensor_2"][0] = [-.1, 1.1]
    elif damage == "illegal":
        arrays["tensor_1"][0] = False
    else:
        for name in ("tensor_0", "tensor_1", "tensor_2"):
            arrays[name][1] = arrays[name][0]

    np.savez(tensor_path, **arrays)
    manifest["tensor_sha256"] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError):
        Reference.from_path(path).result()


def test_table_rejects_incompatible_game_and_missing_provenance(saved_table, tmp_path):
    game, path, _ = saved_table
    with pytest.raises(ArtifactError, match="incompatible with game"):
        load_game("rock_paper_scissors").evaluate(path)

    with pytest.raises(ValueError, match="requires game"):
        PolicyProfile.from_table({}, fallback="uniform").save(tmp_path / "unknown")

    policy = train(game, "cfr", iterations=10, seed=1)
    with pytest.raises(ValueError, match="differs"):
        policy.save(tmp_path / "wrong", game=load_game("rock_paper_scissors"))

    with pytest.raises(FileExistsError):
        policy.save(path.parent)

    assert _ArtifactMetadata.from_path(path).digest is not None


@pytest.mark.parametrize("retain_source", [True, False])
def test_loaded_table_can_be_resaved_after_original_files_disappear(tmp_path, retain_source):
    game = load_game("rock_paper_scissors")
    original = PolicyProfile.from_table({}, fallback="uniform").save(tmp_path / "original", game=game)
    manifest = json.loads(original.path.read_text())
    annotations = {"producer": {"name": "test"}, "evaluation": {"nash_conv": 0}, "progress": {"iterations": 12}}
    original.path.write_text(json.dumps(manifest | annotations | {"ignored": "metadata"}))
    _, _, source = validate_policy_artifact(str(original.path))
    policy = load_policy(str(original.path), expected_game=source.game, retain_source=retain_source)
    assert (policy._portable_reference is not None) == retain_source
    shutil.rmtree(tmp_path / "original")
    copied = policy.save(tmp_path / "copied")
    assert game.evaluate(copied)["nash_conv"] == 0
    saved = json.loads(copied.path.read_text())
    assert all(saved[key] == value for key, value in annotations.items())
    assert "ignored" not in saved


def test_unsaved_custom_table_can_be_submitted_directly(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMAX_ARTIFACTS", str(tmp_path / "artifacts"))
    game = load_game("rock_paper_scissors")
    table = PolicyProfile.from_table({}, fallback="uniform")
    target = LocalTarget(runs=str(tmp_path / "runs"))
    assert game.evaluate(table, submit=True).run(target).result(timeout=30)["nash_conv"] == 0


def test_table_reuse_does_not_depend_on_evaluator(saved_table):
    _, path, _ = saved_table
    game = load_game("kuhn_poker", evaluator=None)
    assert game.evaluate(path, evaluator="nash_conv")["nash_conv"] >= 0
