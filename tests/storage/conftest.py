import pytest

from benchmax.storage import HuggingFaceStore, LocalStore
from tests.storage._fake_hub import _fake_huggingface


@pytest.fixture(params=["local", "huggingface"])
def policy_store(request, tmp_path, monkeypatch):
    if request.param == "local":
        return LocalStore(tmp_path / "store")
    _fake_huggingface(tmp_path, monkeypatch)
    return HuggingFaceStore("benchmax/data")
