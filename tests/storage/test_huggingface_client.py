"""Exercise the real Hub SDK against an in-memory HTTP transport, never a live repository."""

from __future__ import annotations

import httpx
import huggingface_hub as hub
import pytest
from huggingface_hub.utils import _http

from benchmax.storage import HuggingFaceStore, StorageAuthenticationError


@pytest.fixture
def mock_hub():
    previous_factory = _http._GLOBAL_CLIENT_FACTORY

    def install(handler):
        hub.set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True))

    try:
        yield install
    finally:
        hub.set_client_factory(previous_factory)


@pytest.mark.parametrize(
    "token,credential_source",
    [
        (None, "environment"),
        (True, "environment"),
        (False, "environment"),
        ("hf_explicit_test_token", "environment"),
        (None, "saved"),
        (True, "saved"),
    ],
)
def test_real_client_authenticates_bucket_discovery_and_pagination(
    tmp_path, monkeypatch, mock_hub, token, credential_source,
) -> None:
    monkeypatch.delenv("HF_OIDC_RESOURCE", raising=False)
    if credential_source == "saved":
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        token_path = tmp_path / "saved-token"
        token_path.write_text("hf_saved_test_token")
        monkeypatch.setattr(hub.constants, "HF_TOKEN_PATH", str(token_path))
        monkeypatch.setattr(hub.constants, "HF_STORED_TOKENS_PATH", str(tmp_path / "stored-tokens"))
    else:
        monkeypatch.setenv("HF_TOKEN", "hf_environment_test_token")
    monkeypatch.setattr(hub.constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)
    monkeypatch.setattr(hub.constants, "HF_HUB_DISABLE_TELEMETRY", True)
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET"
        assert request.url.path == "/api/buckets/benchmax/data/tree/entries/"
        if "page" not in request.url.params:
            return httpx.Response(200, json=[], headers={"Link":
                '<https://huggingface.co/api/buckets/benchmax/data/tree/entries%2F?page=2>; rel="next"'})
        return httpx.Response(200, json=[])

    mock_hub(handle)
    store = HuggingFaceStore("benchmax/data", token=token, cache_dir=tmp_path / "hub-cache")
    assert store.find() == ()

    implicit = "hf_saved_test_token" if credential_source == "saved" else "hf_environment_test_token"
    credential = token if isinstance(token, str) else implicit
    expected = None if token is False else "Bearer " + credential
    assert len(requests) == 2
    assert all(request.headers.get("authorization") == expected for request in requests), [
        (
            request.method,
            request.url.path,
            request.headers.get("authorization") == expected,
            request.headers.get("authorization") is None,
        )
        for request in requests
    ]


def test_missing_required_saved_login_fails_before_any_network_request(tmp_path, monkeypatch, mock_hub) -> None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_OIDC_RESOURCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hub.constants, "HF_TOKEN_PATH", str(tmp_path / "missing-token"))
    monkeypatch.setattr(hub.constants, "HF_STORED_TOKENS_PATH", str(tmp_path / "missing-stored-tokens"))
    monkeypatch.setattr(hub.constants, "HF_HUB_DISABLE_TELEMETRY", True)
    requests = []

    def unexpected(request):
        requests.append(request)
        raise AssertionError("authentication must fail before network access")

    mock_hub(unexpected)
    with pytest.raises(StorageAuthenticationError, match="hf auth login"):
        HuggingFaceStore("benchmax/data", token=True).find()
    assert requests == []


@pytest.mark.parametrize("partial_failure", [False, True], ids=["round-trip", "partial-upload"])
def test_real_bucket_sdk_transfers_and_partial_failure(tmp_path, monkeypatch, mock_hub, partial_failure):
    import hashlib
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from huggingface_hub.utils import _xet

    from benchmax.storage import BlobReference, StorageError

    monkeypatch.setattr(hub.constants, "HF_HUB_DISABLE_TELEMETRY", True)
    files, content = {}, {}
    seen_headers, uploaded = [], []
    token = "hf_explicit_bucket_test"

    class Transfer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def start_upload_bytes(self, data, **kwargs):
            digest = hashlib.sha256(data).hexdigest()
            content[digest] = data
            return SimpleNamespace(result=lambda: SimpleNamespace(
                xet_info=SimpleNamespace(hash=digest, file_size=len(data))))

        def start_upload_file(self, path, **kwargs):
            return self.start_upload_bytes(Path(path).read_bytes())

        def start_download_file(self, info, path):
            Path(path).write_bytes(content[info.hash])

    class Session:
        def new_upload_commit(self, **kwargs):
            seen_headers.append(kwargs["token_refresh_headers"])
            return Transfer()

        def new_file_download_group(self, **kwargs):
            seen_headers.append(kwargs["token_refresh_headers"])
            return Transfer()

    monkeypatch.setattr(_xet, "get_xet_session", lambda: Session())

    def metadata(path):
        data = files[path]
        return {"type": "file", "path": path, "size": len(data), "xetHash": hashlib.sha256(data).hexdigest()}

    def handle(request):
        seen_headers.append(request.headers)
        path = request.url.path
        if request.method == "GET" and "/tree/" in path:
            prefix = path.split("/tree/", 1)[1]
            return httpx.Response(200, json=[metadata(name) for name in files if name.startswith(prefix)])
        if request.method == "POST" and path.endswith("/batch"):
            operations = [json.loads(line) for line in request.content.splitlines()]
            uploaded.append([op["path"] for op in operations])
            if partial_failure:
                # A successful status with a failed operation must stop publication.
                files[operations[0]["path"]] = content[operations[0]["xetHash"]]
                return httpx.Response(200, json={"failed": [{"path": operations[-1]["path"],
                                                          "error": "simulated failure"}]})
            for op in operations:
                files[op["path"]] = content[op["xetHash"]]
            return httpx.Response(200, json={"failed": []})
        if request.method == "POST" and path.endswith("/paths-info"):
            paths = json.loads(request.content)["paths"]
            return httpx.Response(200, json=[metadata(name) for name in paths if name in files])
        if request.method == "HEAD" and "/resolve/" in path:
            name = path.split("/resolve/", 1)[1]
            info = metadata(name)
            return httpx.Response(200, headers={
                "Content-Length": str(info["size"]), "X-Xet-Hash": info["xetHash"],
                "X-Xet-Refresh-Route": "https://huggingface.co/api/buckets/benchmax/data/xet-read-token",
            })
        raise AssertionError(f"unexpected request {request.method} {path}")

    mock_hub(handle)
    store = HuggingFaceStore("benchmax/data", token=token, cache_dir=tmp_path / "cache")
    blob = BlobReference.from_source(b"weights")
    document = {"kind": "result", "tensor": blob.to_document()}
    if partial_failure:
        with pytest.raises(StorageError, match="retry") as caught:
            store.publish(document, blobs={blob: b"weights"})
        assert type(caught.value.__cause__).__name__ == "BucketBatchError"
        assert not any(name.startswith("entries/") for batch in uploaded for name in batch)
        assert store.find() == ()
    else:
        published = store.publish(document, blobs={blob: b"weights"})
        assert store.find() == (published,)
        assert store.read_document(published.object) == document
        assert store.materialize_blob(blob, tmp_path / "weights").read_bytes() == b"weights"
        assert len(uploaded) == 2
        assert all(not name.startswith("entries/") for name in uploaded[0])
        assert all(name.startswith("entries/") for name in uploaded[1])

    assert seen_headers
    assert all(httpx.Headers(headers).get("authorization") == f"Bearer {token}" for headers in seen_headers), [
        (list(headers), httpx.Headers(headers).get("authorization") == f"Bearer {token}") for headers in seen_headers
    ]
