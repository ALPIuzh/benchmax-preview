"""Policy formats come from installed codecs, not algorithm-specific core branches."""

import json
from importlib.metadata import EntryPoint
from types import SimpleNamespace

import numpy as np
import pytest

from benchmax import PolicyProfile, Reference, load_game
from benchmax.execution.artifacts import policies, tabular
from benchmax.execution.specs import EntrySpec, TrainingSpec, _ArtifactMetadata
from tests.artifact_helpers import write_policy_artifact

_CUSTOM_FORMAT = "example.wrapped-table.npz"


class _CustomPolicy(PolicyProfile):
    def __init__(self, table):
        self.table = table
        super().__init__(table)


def _write_custom(profile, source, destination):
    reference = tabular.write_policy(profile.table, source, destination)
    path = Reference.from_path(reference.uri).path
    manifest = json.loads(path.read_text())
    manifest["format"] = _CUSTOM_FORMAT
    path.write_text(json.dumps(manifest))
    return _ArtifactMetadata.from_path(path)


def _load_custom(path, manifest, source, **kwargs):
    return _CustomPolicy(tabular.load_policy_document(path, manifest, source, **kwargs))


CUSTOM_CODEC = SimpleNamespace(
    POLICY_FORMAT=_CUSTOM_FORMAT, REQUIRES_TRAINING=False,
    supports=lambda profile: isinstance(profile, _CustomPolicy), write_policy=_write_custom,
    validate_policy_document=tabular.validate_policy_document, load_policy_document=_load_custom
)


def test_installed_codec_supports_save_inspect_copy_and_load(tmp_path, monkeypatch):
    distribution = tmp_path / "example_codec-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Metadata-Version: 2.1\nName: example-codec\nVersion: 1.0\n")
    (distribution / "entry_points.txt").write_text(
        f"[benchmax.policy_codecs]\n{_CUSTOM_FORMAT} = tests.policy.test_codecs:CUSTOM_CODEC\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    game = load_game("rock_paper_scissors")
    profile = _CustomPolicy(PolicyProfile.from_table({}, fallback="uniform"))
    reference = profile.save(tmp_path / "policy", game=game)
    assert json.loads(reference.path.read_text())["format"] == _CUSTOM_FORMAT

    with monkeypatch.context() as inspection:
        inspection.setattr(np, "load", lambda *args, **kwargs: pytest.fail("inspection must not load arrays"))
        policies.validate_policy_artifact(str(reference.path))
        copied = reference.save(tmp_path / "copied")

    restored = copied.result()
    assert isinstance(restored, _CustomPolicy)
    assert game.evaluate(restored)["nash_conv"] == 0
    assert game.evaluate(restored.save(tmp_path / "resaved"))["nash_conv"] == 0

    monkeypatch.setenv("BENCHMAX_ARTIFACTS", str(tmp_path / "artifacts"))
    unsaved = _CustomPolicy(PolicyProfile.from_table({}, fallback="uniform"))
    job = game.evaluate(unsaved, submit=True)
    job.audit()
    assert game.evaluate(Reference.from_path(unsaved._portable_reference.uri))["nash_conv"] == 0


@pytest.mark.parametrize("format_name", ["os:system", None, [], {}])
def test_unknown_format_cannot_import_a_codec(format_name, monkeypatch):
    entry = SimpleNamespace(name=tabular.POLICY_FORMAT, load=lambda: pytest.fail("must not load a codec"))
    monkeypatch.setattr(policies, "entry_points", lambda **kwargs: [entry])
    manifest = {"schema_version": 1, "kind": "policy-artifact", "format": format_name}
    with pytest.raises(policies.ArtifactError, match="unsupported policy artifact"):
        policies.validate_policy_manifest(manifest)


def test_duplicate_registration_is_rejected_before_importing(monkeypatch):
    entry = EntryPoint(name=tabular.POLICY_FORMAT, value="does_not_exist", group="benchmax.policy_codecs")
    monkeypatch.setattr(policies, "entry_points", lambda **kwargs: [entry, entry])
    with pytest.raises(policies.ArtifactError, match="multiple policy codecs are registered"):
        policies._policy_codec(tabular.POLICY_FORMAT)


@pytest.mark.parametrize("field,value", [("POLICY_FORMAT", "wrong"), ("REQUIRES_TRAINING", "yes"),
                                       ("load_policy_document", None)])
def test_invalid_codec_registration_is_rejected(monkeypatch, field, value):
    codec = SimpleNamespace(**(vars(CUSTOM_CODEC) | {field: value}))

    entry = SimpleNamespace(name=_CUSTOM_FORMAT, load=lambda: codec)
    monkeypatch.setattr(policies, "entry_points", lambda **kwargs: [entry])
    with pytest.raises(policies.ArtifactError, match="invalid policy codec"):
        policies._policy_codec(_CUSTOM_FORMAT)


def test_ambiguous_codec_match_does_not_write(tmp_path, monkeypatch):
    entries = list(policies.entry_points(group="benchmax.policy_codecs"))
    duplicate = SimpleNamespace(**vars(CUSTOM_CODEC))
    duplicate.supports = lambda profile: True
    entries.append(SimpleNamespace(name=_CUSTOM_FORMAT, load=lambda: duplicate))
    monkeypatch.setattr(policies, "entry_points", lambda **kwargs: entries)
    profile = PolicyProfile.from_table({}, fallback="uniform")
    with pytest.raises(policies.ArtifactError, match="multiple policy codecs support"):
        profile.save(tmp_path / "policy", game=load_game("rock_paper_scissors"))

    assert not (tmp_path / "policy").exists()


def test_unregistered_policy_does_not_write(tmp_path):
    profile = PolicyProfile(lambda *args: None)
    with pytest.raises(policies.ArtifactError, match="no registered safe representation"):
        profile.save(tmp_path / "policy", game=load_game("rock_paper_scissors"))

    assert not (tmp_path / "policy").exists()


def test_neural_format_still_requires_training_provenance(tmp_path):
    training = TrainingSpec(EntrySpec("test-game"), EntrySpec("test-algorithm"), seed=7, device="cpu", total_frames=8)
    path = write_policy_artifact(tmp_path / "policy", training)
    manifest = json.loads(path.read_text())
    manifest.update(training=None, game=training.game.to_document())
    path.write_text(json.dumps(manifest))
    with pytest.raises(policies.ArtifactError, match="requires training provenance"):
        policies.validate_policy_artifact(str(path))
