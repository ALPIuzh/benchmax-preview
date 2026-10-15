import hashlib
import io
import tarfile

import pytest

from benchmax.integrations.stockfish import install as recipe


def archive_bytes():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        member = tarfile.TarInfo("stockfish/test-engine")
        member.size = 6
        archive.addfile(member, io.BytesIO(b"engine"))
    return buffer.getvalue()


def configure(monkeypatch, data, checksum):
    monkeypatch.setattr(recipe.platform, "system", lambda: "Test")
    monkeypatch.setattr(recipe.platform, "machine", lambda: "cpu")
    monkeypatch.setattr(recipe, "ASSETS", {("Test", "cpu"): ("test-engine", checksum)})
    monkeypatch.setattr(recipe.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(data))


def test_install_validates_archive_and_installs_executable(monkeypatch, tmp_path):
    data = archive_bytes()
    configure(monkeypatch, data, hashlib.sha256(data).hexdigest())
    target = recipe.install(directory=tmp_path)
    assert target.read_bytes() == b"engine"
    assert target.stat().st_mode & 0o111
    assert not list(tmp_path.glob(".install-*"))


def test_bad_checksum_preserves_existing_executable(monkeypatch, tmp_path):
    target = tmp_path / "stockfish"
    target.write_bytes(b"existing")
    configure(monkeypatch, archive_bytes(), "invalid")
    with pytest.raises(RuntimeError, match="checksum"):
        recipe.install(directory=tmp_path)
    assert target.read_bytes() == b"existing"


def test_unsupported_host(monkeypatch, tmp_path):
    monkeypatch.setattr(recipe.platform, "system", lambda: "Unsupported")
    with pytest.raises(RuntimeError, match="No bundled"):
        recipe.install(directory=tmp_path)


def test_default_cache_separates_platforms(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(recipe.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(recipe.platform, "machine", lambda: "arm64")
    assert recipe.default_executable() == tmp_path / "benchmax/stockfish/17.1/darwin-arm64/stockfish"
    monkeypatch.setattr(recipe.platform, "machine", lambda: "x86_64")
    assert recipe.default_executable().parent.name == "darwin-x86_64"
