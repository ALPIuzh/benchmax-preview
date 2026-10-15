import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _fake_huggingface(tmp_path, monkeypatch):
    bucket = SimpleNamespace(files={}, tokens=[], uploads=[], downloads=[])
    module = types.ModuleType("huggingface_hub")

    class HfApi:
        def __init__(self, *, token=None):
            self.token = token
            bucket.tokens.append(("api", token))

        def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
            assert bucket_id == "benchmax/data"
            assert recursive is True
            return [SimpleNamespace(type="file", path=path) for path in list(bucket.files) if path.startswith(prefix)]

        def batch_bucket_files(self, bucket_id, *, add):
            assert bucket_id == "benchmax/data"
            bucket.uploads.append([path for _, path in add])
            for source, path in add:
                bucket.files[path] = source if isinstance(source, bytes) else Path(source).read_bytes()

        def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files):
            assert bucket_id == "benchmax/data"
            assert raise_on_missing_files is True
            bucket.tokens.append(("download", self.token))
            bucket.downloads.append([remote for remote, _ in files])
            for remote, local in files:
                Path(local).write_bytes(bucket.files[remote])

    module.HfApi = HfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    # Keep default caches out of the real user's home.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return bucket
