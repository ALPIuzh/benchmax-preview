"""Opt-in live policy round trip; creates and removes one temporary bucket prefix.

Set BENCHMAX_HF_TEST_BUCKET and optionally BENCHMAX_HF_TEST_TOKEN_FILE.
Otherwise the SDK uses its normal authentication. No local login is changed.
"""

import os
from pathlib import Path
from uuid import uuid4

import pytest

from benchmax.storage import STRICT_REQUIREMENTS, HuggingFaceStore
from tests.training_helpers import tiny_training_params


@pytest.mark.skipif(not os.environ.get("BENCHMAX_HF_TEST_BUCKET"), reason="live Hub test is opt-in")
def test_live_strict_policy_round_trip(tmp_path) -> None:
    import huggingface_hub as hub
    pytest.importorskip("benchmarl")
    from benchmax import load_game
    from benchmax.training import train

    bucket_id = os.environ["BENCHMAX_HF_TEST_BUCKET"]
    token_file = os.environ.get("BENCHMAX_HF_TEST_TOKEN_FILE")
    token = Path(token_file).expanduser().read_text().strip() if token_file else None
    api = hub.HfApi(token=token)
    prefix = f"benchmax-test-{uuid4().hex}"
    game = load_game("rock_paper_scissors")
    seed = int(uuid4().hex[:7], 16)
    policy = train(game, "mappo", seed=seed, device="cpu", total_frames=8, verbose=False, params=tiny_training_params())
    original_metrics = game.evaluate(policy)
    try:
        writer = HuggingFaceStore(bucket_id, prefix=prefix, token=token, cache_dir=tmp_path / "writer",
                                  requirements=STRICT_REQUIREMENTS)
        assert writer.find() == ()
        published = writer.publish_policy(policy)
        other = writer.publish({"kind": "result", "value": "catalog batch"})

        # A new client and empty cache must fetch the actual uploaded data.
        reader = HuggingFaceStore(bucket_id, prefix=prefix, token=token, cache_dir=tmp_path / "reader",
                                  requirements=STRICT_REQUIREMENTS)
        assert {entry.object for entry in reader.find()} == {published.object, other.object}
        matches = reader.find_policies(game, training_config="any", seed=seed)
        assert len(matches) == 1 and matches[0].object == published.object
        assert reader.validate_policy(published.object).valid
        restored = matches[0].result()
        assert game.evaluate(restored)["nash_conv"] == pytest.approx(original_metrics["nash_conv"], abs=1e-5)
    finally:
        paths = [item.path for item in api.list_bucket_tree(bucket_id, prefix=prefix + "/", recursive=True)
                 if item.type == "file"]
        if paths:
            api.batch_bucket_files(bucket_id, delete=paths)
    assert list(api.list_bucket_tree(bucket_id, prefix=prefix + "/", recursive=True)) == []
