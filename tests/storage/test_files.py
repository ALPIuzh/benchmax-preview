"""Failure guarantees of the file operations shared by storage backends."""

import os
import stat

import pytest

from benchmax.storage import _files as module
from benchmax.storage._model import BlobReference


def test_materialization_never_replaces_a_destination_created_during_copy(tmp_path, monkeypatch):
    blob = BlobReference.from_source(b"weights")
    source = tmp_path / "source"
    source.write_bytes(b"weights")
    destination = tmp_path / "copies" / "weights"
    original = module._digest_file

    def create_competing_output(path):
        if path.parent == destination.parent and path.name.startswith(".weights."):
            destination.write_bytes(b"owned by concurrent caller")
        return original(path)

    monkeypatch.setattr(module, "_digest_file", create_competing_output)
    with pytest.raises(FileExistsError):
        module._atomic_blob(destination, blob, source, replace_existing=False)
    assert destination.read_bytes() == b"owned by concurrent caller"
    assert list(destination.parent.iterdir()) == [destination]


def test_failed_blob_copy_cleans_temporary_files(tmp_path, monkeypatch):
    blob = BlobReference.from_source(b"weights")
    source = tmp_path / "source"
    source.write_bytes(b"weights")
    destination = tmp_path / "copies" / "weights"

    def fail_copy(source, target, **kwargs):
        target.write(b"partial")
        raise OSError("copy interrupted")

    monkeypatch.setattr(module.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="copy interrupted"):
        module._atomic_blob(destination, blob, source, replace_existing=False)
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []


def test_atomic_publication_syncs_buffered_data_through_a_writable_handle(tmp_path, monkeypatch):
    destination = tmp_path / "weights"
    original = os.fsync
    synced_sizes = []

    def require_write_access(descriptor):
        info = os.fstat(descriptor)
        if stat.S_ISREG(info.st_mode):
            # Windows requires write access when flushing; enforce it on every platform.
            os.write(descriptor, b"")
            synced_sizes.append(info.st_size)
            assert not destination.exists()
        original(descriptor)

    monkeypatch.setattr(os, "fsync", require_write_access)
    module._atomic_bytes(destination, b"weights")
    assert destination.read_bytes() == b"weights"
    assert synced_sizes == [len(b"weights")]
    assert list(tmp_path.iterdir()) == [destination]
