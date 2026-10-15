from __future__ import annotations

import json
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from benchmax import Reference


@pytest.mark.parametrize("target,label,output", [
    ({"kind": "local", "runs": "/tmp/runs"}, "Local", "policy"),
    ({"kind": "slurm", "host": "cluster.example", "user": "researcher", "port": 2222, "runs": "/scratch/runs"},
     "Slurm · researcher@cluster.example:2222", "metrics"),
    ({"kind": "slurm", "host": None, "runs": "/scratch/runs"}, "Slurm · direct", "policy"),
    ({"kind": "adapter", "uri": "batch://example"}, "Adapter · batch://example", "metrics"),
])
def test_reference_display_exposes_identity_and_target(target, label, output):
    reference = Reference({"schema_version": 1, "kind": "job-output", "job_id": "training-0000000000000001",
                           "target": target, "output": output, "workload_digest": "a" * 64})
    text, html = repr(reference), reference._repr_html_()
    assert reference.job_id in text and reference.job_id in html
    assert label in text and label in html
    assert f"Output: {output}" in text
    assert f"{output.capitalize()} reference" in html
    assert "a" * 64 in text and "a" * 64 in html

    if "runs" in target:
        assert target["runs"] in text and target["runs"] in html

    assert "Target configuration and identity" in html


@pytest.mark.parametrize("kind", ["job-output", "file"])
def test_notebook_reference_display_is_passive_and_keeps_reference_immutable(monkeypatch, kind):
    DisplayFormatter = pytest.importorskip("IPython.core.formatters").DisplayFormatter

    formatter = DisplayFormatter()
    document = {"schema_version": 1, "kind": kind, "output": "policy"}

    if kind == "job-output":
        document.update(job_id="training-0000000000000001", workload_digest="a" * 64,
                        target={"kind": "slurm", "host": "cluster.example", "runs": "/scratch/runs"})
    else:
        document.update(uri="/not-present/policy.json", digest="a" * 64)

    reference = Reference(document)
    before, fingerprint = reference.to_document(), hash(reference)

    def forbidden(*args, **kwargs):
        pytest.fail("display must not read files, connect, resolve, wait, or fetch status")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr("time.sleep", forbidden)
    monkeypatch.setattr(Reference, "_open", forbidden)
    monkeypatch.setattr("benchmax.execution.jobs.outputs.file_result", forbidden)
    monkeypatch.setattr("benchmax.execution.targets.registry.load_target", forbidden)
    monkeypatch.setattr("benchmax.execution.jobs.reference.target_from_document", forbidden)
    monkeypatch.setattr(Reference, "result", forbidden)
    mime, _ = formatter.format(reference)
    assert (reference.job_id if kind == "job-output" else str(reference.path)) in mime["text/plain"]
    assert "Policy reference" in mime["text/html"]
    assert "no status query" in mime["text/html"]
    assert reference.to_document() == before and hash(reference) == fingerprint
    assert Reference.from_document(json.loads(json.dumps(before))) == reference

    if kind == "job-output":
        reference.target["host"] = "different.example"
        assert reference.target["host"] == "cluster.example"

    with pytest.raises(FrozenInstanceError):
        reference.job_id = "different"


def test_reference_html_escapes_all_external_text():
    reference = Reference({"schema_version": 1, "kind": "job-output", "job_id": '<script>alert("job")</script>',
                           "target": {"kind": "slurm", "host": '<img src=x onerror="bad()">',
                                      "runs": "</td><script>bad()</script>", "extra": {"nested": "<svg onload=bad()>"}},
                           "output": "policy", "workload_digest": "b" * 64})
    html = reference._repr_html_()

    for unsafe in ("<script>", "<img", "<svg"):
        assert unsafe not in html

    for escaped in ("&lt;script&gt;", "&lt;img", "&lt;svg"):
        assert escaped in html


@pytest.mark.parametrize("output", ["policy", "metrics"])
def test_file_reference_display_includes_source_and_content_identity(output):
    reference = Reference({"schema_version": 1, "kind": "file", "output": output,
                           "uri": "/saved/<script>policy</script>.json", "digest": "c" * 64})
    text, html = repr(reference), reference._repr_html_()
    assert "Source: Local file" in text
    assert str(reference.path) in text
    assert f"Output: {output}" in text
    assert "File digest" in text and "c" * 64 in text
    assert "File identity" in html and "c" * 64 in html
    assert "&lt;script&gt;" in html and "<script>" not in html
    assert "Target configuration" not in html
