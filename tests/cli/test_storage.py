"""Storage inspection remains metadata-only and uses one store selector."""

import json

import pytest

from benchmax import cli
from benchmax.storage import HuggingFaceStore, LocalStore
from tests.artifact_helpers import write_policy_artifact
from tests.storage._helpers import _release, _training


def test_info_is_offline_and_does_not_expose_credentials(monkeypatch, capsys):
    monkeypatch.setenv("BENCHMAX_REFERENCE_BUCKET", "owner/reference")
    monkeypatch.setenv("HF_TOKEN", "secret-must-not-appear")
    monkeypatch.setattr(HuggingFaceStore, "_api", lambda self: pytest.fail("info must be offline"))
    assert cli.main(["store", "info", "--json"]) == 0
    output = capsys.readouterr().out
    info = json.loads(output)
    assert info["location"] == "owner/reference"
    assert info["requirements"]["name"] == "strict-v1"
    assert "secret-must-not-appear" not in output
    assert cli.main(["store", "info", "--use", "huggingface:owner/data/experiments/run-1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["prefix"] == "experiments/run-1"


def test_show_all_configurations_and_filters_without_weights(tmp_path, monkeypatch, capsys):
    store = LocalStore(tmp_path / "store")
    for index, (seed, frames) in enumerate(((7, 100), (8, 100), (7, 200))):
        artifact = write_policy_artifact(tmp_path / f"policy-{index}", _training(seed=seed, total_frames=frames))
        store.publish_policy(artifact, release=_release())

    monkeypatch.setattr(LocalStore, "_verified_blob_path", lambda *args: pytest.fail("must not load weights"))
    args = ["store", "show", "--use", f"local:{store.root}", "--game", "test-game",
            "--algorithm", "test-algorithm"]
    assert cli.main([*args, "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 3
    assert len({row["configuration"] for row in rows}) == 2
    assert all(len(row["id"]) == 64 for row in rows)
    assert {row["training"]["total_frames"] for row in rows} == {100, 200}
    assert cli.main([*args, "--seed", "8"]) == 0
    output = capsys.readouterr().out
    assert "POLICY" in output and "test-game" in output
    assert len(output.splitlines()) == 2


def test_local_relative_path_and_empty_store(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["store", "info", "--use", "local:./policies", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["location"] == str(tmp_path / "policies")
    assert not (tmp_path / "policies").exists()
    assert cli.main(["store", "show", "--use", "local:./policies", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


@pytest.mark.parametrize("uri", ["", "policies", "s3:bucket", "huggingface:owner", "local:", "local://old"])
def test_invalid_store_selector_is_a_concise_error(uri, capsys):
    assert cli.main(["store", "show", "--use", uri]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "--use expects" in output.err


def test_missing_reference_configuration_is_a_concise_error(monkeypatch, capsys):
    from benchmax.storage.reference import OmegaConf

    config = OmegaConf.create({"bucket_id": None, "requirements": "strict-v1"})
    monkeypatch.setattr(OmegaConf, "load", lambda stream: config)
    monkeypatch.delenv("BENCHMAX_REFERENCE_BUCKET", raising=False)
    assert cli.main(["store", "info"]) == 1
    assert "BENCHMAX_REFERENCE_BUCKET" in capsys.readouterr().err


def test_catalog_failure_returns_error_instead_of_empty_success(monkeypatch, capsys):
    from benchmax.storage import StorageError

    def unavailable(self):
        raise StorageError("cannot list publications")

    monkeypatch.setattr(HuggingFaceStore, "_catalog", unavailable)
    assert cli.main(["store", "show", "--use", "huggingface:owner/data", "--json"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "cannot list publications" in output.err
