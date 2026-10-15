"""Pinned experts reuse storage without discovery or live ranking."""

import sys
from types import SimpleNamespace

import pytest

from benchmax import experts, load_expert, load_game
from benchmax.storage import HuggingFaceStore, PolicyNotFoundError
from tests.storage._fake_hub import _fake_huggingface


@pytest.fixture
def expert_collection(tmp_path, monkeypatch, tiny_trained_policy):
    game, policy, _ = tiny_trained_policy
    bucket = _fake_huggingface(tmp_path, monkeypatch)
    store = HuggingFaceStore("benchmax/data", cache_dir=tmp_path / "writer")
    selected = store.publish_policy(policy)
    document = store.read_document(selected.object)
    # A competing publication must never change a library-pinned expert.
    store.publish_policy(policy, evaluation={
        "workload_digest": selected.training.digest,
        "tensor_sha256": document["tensor"]["digest"],
        "metrics": {"exploitability": 0.0},
    })
    monkeypatch.setattr(experts, "_expert_references", lambda: (
        "benchmax/data", {"rock_paper_scissors": selected.object.digest},
    ))

    def no_listing(*args, **kwargs):
        raise AssertionError("expert loading must not list the bucket")

    monkeypatch.setattr(sys.modules["huggingface_hub"].HfApi, "list_bucket_tree", no_listing)
    bucket.downloads.clear()
    bucket.tokens.clear()
    return SimpleNamespace(game=game, policy=policy, bucket=bucket, selected=selected,
                           document=document, cache=tmp_path / "reader")


def test_expert_downloads_only_the_pin_and_reuses_verified_cache_offline(expert_collection, monkeypatch):
    c = expert_collection
    monkeypatch.setenv("BENCHMAX_REFERENCE_BUCKET", "unrelated/collection")
    expert = load_expert("rock_paper_scissors", token="test-expert-token", cache_dir=c.cache)
    expected = c.game.evaluate(c.policy)
    actual = c.game.evaluate(expert)
    for metric, value in expected.items():
        assert actual[metric] == pytest.approx(value)
    paths = [path for batch in c.bucket.downloads for path in batch]
    tensor = c.document["tensor"]["digest"]
    assert paths == [c.selected.object.path, f"blobs/{tensor[:2]}/{tensor}"]
    assert all(token == "test-expert-token" for _, token in c.bucket.tokens)

    def offline(*args, **kwargs):
        raise AssertionError("a fully cached expert must load offline")

    monkeypatch.setattr(sys.modules["huggingface_hub"], "HfApi", offline)
    # Evaluator composition does not change the expert's game identity.
    restored = load_expert(load_game("rock_paper_scissors", evaluator=None), cache_dir=c.cache)
    actual = c.game.evaluate(restored)
    for metric, value in expected.items():
        assert actual[metric] == pytest.approx(value)


def test_expert_rejects_changed_game_rules_before_downloading_weights(expert_collection):
    c = expert_collection
    different_game = load_game("rock_paper_scissors", max_episode_steps=2)
    with pytest.raises(ValueError, match="does not match the requested game configuration"):
        load_expert(different_game, cache_dir=c.cache)
    assert c.bucket.downloads == [[c.selected.object.path]]


def test_unavailable_expert_fails_before_network_access(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported experts must fail before constructing a store")

    monkeypatch.setattr(HuggingFaceStore, "__init__", unexpected)
    with pytest.raises(PolicyNotFoundError, match="available games:.*kuhn_poker"):
        load_expert("gridworld_3x3")
