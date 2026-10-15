from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmax.storage import LocalStore
from benchmax.storage._model import BlobReference, IntegrityError, StorageError
from tests.artifact_helpers import write_policy_artifact
from tests.storage._helpers import _release, _result_document, _training


def test_local_publication_is_idempotent_and_rejects_conflicting_metadata(tmp_path: Path) -> None:
    root = tmp_path / "store"
    document = {"kind": "result", "version": "1"}
    first = LocalStore(root).publish(document, dimensions={"version": "1"})
    second = LocalStore(root).publish({"version": "1", "kind": "result"}, dimensions={"version": "1"})

    assert first == second
    assert LocalStore(root).find() == (first,)
    with pytest.raises(StorageError, match="conflicting catalog metadata"):
        LocalStore(root).publish(document, dimensions={"version": "2"})


def test_local_store_detects_document_and_blob_corruption(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "store")
    blob = BlobReference.from_source(b"weights")
    published = store.publish(_result_document(blob), blobs={blob: b"weights"})
    (store.root / published.object.path).write_text("{}", encoding="utf-8")

    with pytest.raises(IntegrityError, match="stored document"):
        store.read_document(published.object)

    (store.root / blob.path).write_bytes(b"changed")
    with pytest.raises(IntegrityError, match="stored blob"):
        store.materialize_blob(blob, tmp_path / "copy")


def test_blob_sources_are_verified_before_publication(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "store")
    reference = BlobReference.from_source(b"expected")

    with pytest.raises(IntegrityError, match="blob source"):
        store.publish({"kind": "result"}, blobs={reference: b"different"})

    assert not store.root.exists()


def test_idempotent_publication_revalidates_its_blobs(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "store")
    weights = b"weights"
    blob = BlobReference.from_source(weights)
    document = _result_document(blob)
    store.publish(document, blobs={blob: weights})
    (store.root / blob.path).write_bytes(b"damaged")

    with pytest.raises(IntegrityError, match="stored blob"):
        store.publish(document, blobs={blob: weights})


def test_stored_blob_corruption_fails_validation_and_loading(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "store")
    training = _training(seed=7)
    published = store.publish_policy(write_policy_artifact(tmp_path / "policy", training), release=_release())
    document = store.read_document(published.object)
    tensor = BlobReference(document["tensor"]["digest"], document["tensor"]["size"])
    (store.root / tensor.path).write_bytes(b"corrupt")
    assert not store.validate_policy(published.object).valid
    with pytest.raises(IntegrityError):
        store.load_policy(training.game)


@pytest.mark.parametrize("field,value", [("seed", "99"), ("algorithm", "wrong"), ("release", "wrong")])
def test_lookup_rechecks_catalog_claims_against_the_policy_record(tmp_path: Path, field, value) -> None:
    store = LocalStore(tmp_path / "store")
    training = _training(seed=7)
    store.publish_policy(write_policy_artifact(tmp_path / "policy", training), release=_release())
    path = store.root / "catalog.json"
    document = json.loads(path.read_text())
    document["entries"][0]["dimensions"][field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(StorageError, match="authoritative record"):
        store.find_policies(training.game)


def test_concurrent_publishers_preserve_every_catalog_entry(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    root = tmp_path / "store"
    ready = Barrier(8)

    def publish(value):
        ready.wait(timeout=5)
        return LocalStore(root).publish({"kind": "result", "value": value})

    with ThreadPoolExecutor(max_workers=8) as pool:
        entries = list(pool.map(publish, range(8)))
    assert {entry.object for entry in LocalStore(root).find()} == {entry.object for entry in entries}


def test_failed_catalog_commit_keeps_previous_publication_and_can_retry(tmp_path, monkeypatch):
    from benchmax.storage import local

    store = LocalStore(tmp_path / "store")
    first = store.publish({"kind": "result", "value": 1})
    original = local.write_document

    def interrupt(value, path):
        if path.name == "catalog.json":
            raise OSError("catalog commit interrupted")
        original(value, path)

    with monkeypatch.context() as patch:
        patch.setattr(local, "write_document", interrupt)
        with pytest.raises(OSError, match="interrupted"):
            store.publish({"kind": "result", "value": 2})
    assert store.find() == (first,)
    second = store.publish({"kind": "result", "value": 2})
    assert {entry.object for entry in store.find()} == {first.object, second.object}


@pytest.mark.parametrize("with_blob", [False, True], ids=["document", "blob"])
def test_interrupted_publication_removes_temporary_files(tmp_path, monkeypatch, with_blob):
    from benchmax.utils import files as local

    store = LocalStore(tmp_path / "store")
    blob = BlobReference.from_source(b"weights")

    def fail_sync(descriptor):
        raise OSError("disk write interrupted")

    monkeypatch.setattr(local.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="interrupted"):
        store.publish({"kind": "result"}, blobs={blob: b"weights"} if with_blob else None)
    assert store.find() == ()
    assert [path for path in store.root.rglob(".*") if path.is_file()] == [store.root / ".publish.lock"]
