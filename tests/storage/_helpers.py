from benchmax.execution.specs import EntrySpec, TrainingSpec
from benchmax.release import LibraryRelease
from benchmax.storage._model import BlobReference
from tests import REPOSITORY


def _result_document(blob: BlobReference) -> dict[str, object]:
    return {"schema_version": 1, "kind": "result", "payload": blob.to_document()}


def _training(*, seed: int, total_frames: int = 100) -> TrainingSpec:
    game = EntrySpec("test-game", {"size": 3}, {"name": "test-game", "size": 3})
    algorithm = EntrySpec("test-algorithm", {"width": 32}, {"name": "test-algorithm", "width": 32})
    return TrainingSpec(game, algorithm, seed=seed, device="cpu", total_frames=total_frames)


def _release() -> LibraryRelease:
    return LibraryRelease.from_pyproject(REPOSITORY / "pyproject.toml")


def _provenance(release: LibraryRelease) -> dict[str, object]:
    return {
        "source_revision": "1" * 40,
        "python_version": "3.11.12",
        "packages": {release.name: release.version},
    }
