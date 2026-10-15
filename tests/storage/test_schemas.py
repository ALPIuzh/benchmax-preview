"""Stored document compatibility and rejection at the decoding boundary."""

import json

import pytest

from benchmax.storage._model import CatalogEntry, ObjectReference, StorageError, _read_catalog, canonical_document
from benchmax.storage._policy_record import _PolicyRecord
from tests.artifact_helpers import write_policy_artifact
from tests.storage._helpers import _release, _training


def test_canonical_identity_preserves_existing_utf8_and_number_encoding():
    document = {"name": "é", "metrics": {"score": 0.25}, "kind": "result"}
    assert canonical_document(document) == b'{"kind":"result","metrics":{"score":0.25},"name":"\xc3\xa9"}'
    assert ObjectReference.from_document(document).digest == "f4304eeb1aa6151aa9f4fc6da1c778bbb16dd2a2bc3a8cc4aa50775814975953"


@pytest.mark.parametrize("patch", [{"extra": 1}, {"schema_version": True}, {"schema_version": 2}, {"entries": {}}])
def test_catalog_schema_rejects_unknown_fields_and_invalid_headers(patch):
    document = {"schema_version": 1, "entries": [], **patch}
    with pytest.raises(StorageError):
        _read_catalog(json.dumps(document).encode())


@pytest.mark.parametrize("patch", [{"extra": 1}, {"object": 1}, {"dimensions": {"seed": 7}}])
def test_catalog_entry_schema_rejects_wrong_types_without_coercion(patch):
    document = {"kind": "policy", "object": "a" * 64, "dimensions": {}, **patch}
    with pytest.raises(StorageError):
        CatalogEntry.from_document(document)


@pytest.fixture(scope="module")
def stored_policy(tmp_path_factory):
    artifact = write_policy_artifact(tmp_path_factory.mktemp("schema-policy") / "policy", _training(seed=7))
    manifest = json.loads(artifact.read_text())
    manifest["producer"] = {"release": _release().to_document()}
    document = {
        "schema_version": 1, "kind": "policy", "manifest": manifest,
        "tensor": {"digest": manifest["tensor_sha256"], "size": (artifact.parent / manifest["tensor_file"]).stat().st_size},
    }
    return artifact, document


def test_stored_policy_preserves_the_document_format(stored_policy):
    artifact, document = stored_policy
    record, _ = _PolicyRecord.from_artifact(str(artifact), _release())
    assert record.to_document() == document
    assert _PolicyRecord.from_document(document).to_document() == document


@pytest.mark.parametrize("patch", [
    {"extra": 1}, {"schema_version": True}, {"schema_version": 2}, {"kind": "result"},
    {"tensor": {"digest": "a" * 64, "size": True}}, {"tensor": {"digest": "a" * 64, "size": -1}},
    {"tensor": {"digest": "invalid", "size": 1}}, {"tensor": {"digest": "a" * 64, "size": 1, "extra": 1}},
    {"tensor": {"digest": "a" * 64, "size": 1}}, {"manifest": []},
])
def test_stored_policy_rejects_malformed_documents(stored_policy, patch):
    _, document = stored_policy
    with pytest.raises(StorageError):
        _PolicyRecord.from_document({**document, **patch})
