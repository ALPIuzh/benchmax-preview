from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from benchmax.storage import STRICT_REQUIREMENTS, HuggingFaceStore, StorageAuthenticationError, reference_store
from benchmax.storage._model import BlobReference, StorageError
from tests.storage._fake_hub import _fake_huggingface
from tests.storage._helpers import _result_document


def test_snapshot_keeps_original_catalog_after_publication(tmp_path, monkeypatch):
    bucket = _fake_huggingface(tmp_path, monkeypatch)
    token = "hf_test_only_credential"
    store = HuggingFaceStore("benchmax/data", token=token, requirements=STRICT_REQUIREMENTS)
    weights = b"remote weights"
    blob = BlobReference.from_source(weights)
    document = _result_document(blob)
    published = store.publish(document, dimensions={"study": "first"}, blobs={blob: weights})
    snapshot = store.snapshot()
    newer = store.publish({"kind": "result", "value": 2}, dimensions={"study": "second"})

    assert snapshot.token == token
    assert snapshot.requirements is STRICT_REQUIREMENTS
    assert all(token.encode() not in value for value in bucket.files.values())
    assert snapshot.find() == (published,)
    assert {entry.object for entry in store.find()} == {published.object, newer.object}
    assert snapshot.read_document(published.object) == document
    assert snapshot.materialize_blob(blob, tmp_path / "weights").read_bytes() == weights
    with pytest.raises(StorageError, match="read-only"):
        snapshot.publish({"kind": "another"})


@pytest.mark.parametrize("operation,status", [("read", 401), ("write", 403)])
def test_hub_access_failures_explain_login_and_required_permission(tmp_path, monkeypatch, operation, status) -> None:
    _fake_huggingface(tmp_path, monkeypatch)
    module = sys.modules["huggingface_hub"]

    class Denied(Exception):
        response = SimpleNamespace(status_code=status)

    def deny(*args, **kwargs):
        raise Denied("sensitive response must not be echoed")

    store = HuggingFaceStore("benchmax/data", token=False)
    if operation == "read":
        monkeypatch.setattr(module.HfApi, "list_bucket_tree", deny)
        action = store.find
    else:
        monkeypatch.setattr(module.HfApi, "batch_bucket_files", deny)
        action = lambda: store.publish({"kind": "result"})
    with pytest.raises(StorageAuthenticationError, match="hf auth login") as caught:
        action()
    assert f"{operation} access" in str(caught.value)
    assert "sensitive" not in str(caught.value)


def test_reference_store_uses_the_same_methods_requirements_and_credentials(monkeypatch, tmp_path) -> None:
    repository = _fake_huggingface(tmp_path, monkeypatch)
    monkeypatch.setenv("BENCHMAX_REFERENCE_BUCKET", "benchmax/data")
    reference = reference_store(token=True, cache_dir=tmp_path / "cache")
    user = HuggingFaceStore("benchmax/data", requirements=STRICT_REQUIREMENTS, token=True)
    assert type(reference) is type(user)
    assert reference.requirements == user.requirements == STRICT_REQUIREMENTS
    assert repository.tokens == []  # Constructing the configured store is offline.
    assert reference.find_policies() == user.find_policies() == ()
    assert reference_store(token=False).token is False
    assert reference.bucket_id == "benchmax/data"


def test_partial_upload_never_publishes_a_discovery_record(tmp_path, monkeypatch):
    bucket = _fake_huggingface(tmp_path, monkeypatch)
    store = HuggingFaceStore("benchmax/data")
    first = store.publish({"kind": "result", "value": 1})
    api = sys.modules["huggingface_hub"].HfApi
    original = api.batch_bucket_files

    def interrupted(self, bucket_id, *, add):
        original(self, bucket_id, add=add[:1])
        raise OSError("interrupted after first object")

    with monkeypatch.context() as patch:
        patch.setattr(api, "batch_bucket_files", interrupted)
        with pytest.raises(StorageError, match="retry"):
            store.publish({"kind": "result", "value": 2})

    assert store.find() == (first,)
    assert len([path for path in bucket.files if path.startswith("entries/")]) == 1
    second = store.publish({"kind": "result", "value": 2})
    assert {entry.object for entry in store.find()} == {first.object, second.object}


def test_concurrent_writers_preserve_distinct_publications(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    _fake_huggingface(tmp_path, monkeypatch)
    barrier = Barrier(4)
    original = HuggingFaceStore._catalog

    def together(self, reference=None):
        entries = original(self, reference)
        barrier.wait(timeout=10)
        return entries

    with monkeypatch.context() as patch:
        patch.setattr(HuggingFaceStore, "_catalog", together)
        with ThreadPoolExecutor(max_workers=4) as pool:
            publications = list(pool.map(lambda i: HuggingFaceStore("benchmax/data").publish(
                {"kind": "result", "value": i}), range(4)))

    assert {entry.object for entry in HuggingFaceStore("benchmax/data").find()} == {
        item.object for item in publications
    }


def test_retry_after_marker_response_loss_is_idempotent(tmp_path, monkeypatch):
    bucket = _fake_huggingface(tmp_path, monkeypatch)
    api = sys.modules["huggingface_hub"].HfApi
    original = api.batch_bucket_files

    def lost_response(self, bucket_id, *, add):
        original(self, bucket_id, add=add)
        if add[0][1].startswith("entries/"):
            raise OSError("response lost")

    store = HuggingFaceStore("benchmax/data")
    with monkeypatch.context() as patch:
        patch.setattr(api, "batch_bucket_files", lost_response)
        with pytest.raises(StorageError):
            store.publish({"kind": "result"})

    uploads = len(bucket.uploads)
    published = store.publish({"kind": "result"})
    assert store.find() == (published,)
    assert len(bucket.uploads) == uploads


def test_hash_checks_reject_remote_corruption_and_repair_local_cache(tmp_path, monkeypatch):
    from benchmax.storage import IntegrityError

    bucket = _fake_huggingface(tmp_path, monkeypatch)
    store = HuggingFaceStore("benchmax/data", cache_dir=tmp_path / "cache")
    published = store.publish({"kind": "result"})
    reference = published.object
    assert store.read_document(reference) == {"kind": "result"}
    cache = store.cache_dir / store.bucket_id / reference.path
    cache.write_bytes(b"corrupt cache")
    assert store.read_document(reference) == {"kind": "result"}
    cache.unlink()
    bucket.files[reference.path] = b"corrupt remote"
    with pytest.raises(IntegrityError):
        store.read_document(reference)
    assert not cache.exists()


def test_prefixes_isolate_collections(tmp_path, monkeypatch):
    _fake_huggingface(tmp_path, monkeypatch)
    first = HuggingFaceStore("benchmax/data", prefix="tests/one")
    second = HuggingFaceStore("benchmax/data", prefix="tests/two")
    published = first.publish({"kind": "result"})
    assert first.find() == (published,)
    assert second.find() == ()
    assert first.snapshot().prefix == "tests/one"


@pytest.mark.parametrize("prefix", ["/root", "../escape", "a/../b", "a//b", "a/", "a\\b", None, 1])
def test_invalid_prefix_is_rejected(prefix):
    with pytest.raises(ValueError):
        HuggingFaceStore("benchmax/data", prefix=prefix)


def test_reference_bucket_must_be_configured(monkeypatch):
    from benchmax.storage.reference import OmegaConf

    config = OmegaConf.create({"bucket_id": None, "requirements": "strict-v1"})
    monkeypatch.setattr(OmegaConf, "load", lambda stream: config)
    monkeypatch.delenv("BENCHMAX_REFERENCE_BUCKET", raising=False)
    with pytest.raises(StorageError, match="not configured"):
        reference_store()


def test_conflicting_publication_records_are_detected(tmp_path, monkeypatch):
    from benchmax.storage._model import CatalogEntry, ObjectReference, canonical_document

    bucket = _fake_huggingface(tmp_path, monkeypatch)
    store = HuggingFaceStore("benchmax/data")
    published = store.publish({"kind": "result"}, dimensions={"study": "one"})
    conflicting = CatalogEntry("result", published.object, {"study": "two"})
    digest = ObjectReference.from_document(conflicting.to_document()).digest
    bucket.files[f"entries/{published.object.digest}/{digest}.json"] = canonical_document(conflicting.to_document())
    with pytest.raises(StorageError, match="conflicting catalog metadata"):
        store.find()


def test_publication_does_not_download_unrelated_catalog_entries(tmp_path, monkeypatch):
    bucket = _fake_huggingface(tmp_path, monkeypatch)
    store = HuggingFaceStore("benchmax/data")
    first = store.publish({"kind": "result", "value": 1})
    second = store.publish({"kind": "result", "value": 2})
    assert not any(operation == "download" for operation, _ in bucket.tokens)
    assert {entry.object for entry in store.find()} == {first.object, second.object}


def test_catalog_downloads_missing_markers_together_and_reuses_verified_cache(tmp_path, monkeypatch):
    bucket = _fake_huggingface(tmp_path, monkeypatch)
    store = HuggingFaceStore("benchmax/data", prefix="experiment")
    entries = [store.publish({"kind": "result", "value": value}) for value in range(3)]
    assert bucket.downloads == []
    assert set(store.find()) == set(entries)
    assert len(bucket.downloads) == 1
    markers = bucket.downloads[0]
    assert len(markers) == 3
    assert all(path.startswith("experiment/entries/") for path in markers)
    assert set(store.find()) == set(entries)
    assert len(bucket.downloads) == 1

    # A damaged cache entry is fetched again; correct neighbors stay cached.
    (store.cache_dir / store.bucket_id / markers[0]).write_bytes(b"corrupt")
    assert set(store.find()) == set(entries)
    assert bucket.downloads[1] == [markers[0]]


@pytest.mark.parametrize("corrupt_copy", [False, True])
def test_upload_staging_verifies_temporary_copies_without_disk_sync(tmp_path, monkeypatch, corrupt_copy):
    from pathlib import Path

    from benchmax.storage import IntegrityError, huggingface

    bucket = _fake_huggingface(tmp_path, monkeypatch)
    source = tmp_path / "weights"
    source.write_bytes(b"weights")
    blob = BlobReference.from_source(source)
    store = HuggingFaceStore("benchmax/data")
    original = huggingface.shutil.copyfile
    copies = []

    def copy_blob(source, destination):
        result = original(source, destination)
        copies.append(Path(destination))
        if corrupt_copy:
            Path(destination).write_bytes(b"corrupt")
        return result

    def unexpected_sync(descriptor):
        raise AssertionError("temporary upload copies do not need durable disk writes")

    monkeypatch.setattr(huggingface.shutil, "copyfile", copy_blob)
    monkeypatch.setattr(huggingface.os, "fsync", unexpected_sync)
    if corrupt_copy:
        with pytest.raises(IntegrityError):
            store.publish(_result_document(blob), blobs={blob: source})
        assert bucket.uploads == []
    else:
        published = store.publish(_result_document(blob), blobs={blob: source})
        assert bucket.files[blob.path] == b"weights"
        assert store.find() == (published,)

    assert len(copies) == 1 and not copies[0].exists()


def test_reference_store_defaults_to_public_preview(monkeypatch):
    monkeypatch.delenv("BENCHMAX_REFERENCE_BUCKET", raising=False)
    store = reference_store(token=False)
    assert store.bucket_id == "ALPIuzh/benchmax-preview"
    assert store.requirements == STRICT_REQUIREMENTS
