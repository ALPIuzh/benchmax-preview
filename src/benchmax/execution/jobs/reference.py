"""A ``Reference`` identifies a result without loading it. It can point to a :doc:`job's output
</reference/execution/jobs>`, a policy in a :doc:`store </reference/storage>`, or saved local files. A policy reference
loads a ``PolicyProfile``, and an evaluation reference loads a metrics dictionary.

.. list-table::
   :header-rows: 1

   * - Operation
     - Effect
   * - ``reference.result()``
     - Wait if needed, then load the policy or metrics.
   * - ``reference.save(path="./saved")``
     - Copy the files to a new local directory and return a reference to them.
   * - ``reference.result(path="./saved")``
     - Save the files, then load the result.

Job outputs
-----------

After submitting a training job, obtain its policy reference without waiting for training to finish:

.. code-block:: python

    reference = job.reference()
    policy = reference.result()
    metrics = game.evaluate(reference)

Here ``job`` is a submitted training job and ``game`` is its compatible game. To chain jobs without loading the policy
locally, use ``game.evaluate(reference, submit=True).run(target)``. The evaluation waits for successful training, and both
jobs must use the same target endpoint and run store.

Stored policies and local files
-------------------------------

A store returns a reference when publishing a supported policy. Continuing with the trained ``policy`` above:

.. code-block:: python

    from benchmax import Reference
    from benchmax.storage import LocalStore

    store = LocalStore("./policy-store")
    stored = store.publish_policy(policy)
    local = stored.save(path="./saved-policy")
    restored = Reference.from_path("./saved-policy").result()

``stored`` points into the store, while ``local`` points to the copied artifact. Both can be passed to
``game.evaluate(...)``. You can also save a supported policy directly with ``policy.save(path)``, but arbitrary Python
callables need a supported representation before they can be saved.

Keep a reference for later
--------------------------

Serialize the reference to retain its identity without copying the result files:

.. code-block:: python

    import json
    from pathlib import Path

    Path("reference.json").write_text(json.dumps(stored.to_document()))
    reference = Reference.from_document(json.loads(Path("reference.json").read_text()))
    policy = reference.result()

The source must remain accessible. Use ``save()`` when you want a separate local copy of the artifacts. Displaying or
serializing a reference does not query job status or load a model.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from html import escape
from typing import Generic, Self, TypeVar, cast

from benchmax.execution.artifacts import policies
from benchmax.execution.artifacts.results import result_from_document
from benchmax.execution.jobs import outputs
from benchmax.execution.specs import EntrySpec, TrainingSpec, WorkloadSpec, _ReferenceSpec
from benchmax.execution.targets.registry import target_from_document, target_label
from benchmax.execution.targets.service import JobError, SubmittedJob, _validate_result, resolve_job_service
from benchmax.policy import PolicyProfile

ResultT_co = TypeVar("ResultT_co", covariant=True)


@dataclass(frozen=True, init=False)
class Reference(_ReferenceSpec, Generic[ResultT_co]):
    """An immutable output handle with explicit loading and saving.

    Job sources capture their target settings, and file sources pin their content.
    """

    def _display_fields(self) -> list[tuple[str, str]]:
        if self.kind == "stored-policy":
            source = self.to_document()["store"]
            return [("Source", source["kind"]), ("Store", source.get("root", source.get("bucket_id", ""))),
                    ("Output", self.output), ("Seed", str(self.training.seed))]

        if self.kind == "file":
            return [("Source", "Local file"), ("Path", str(self.path)), ("Output", self.output)]

        target = self.target
        fields = [("Job", self.job_id), ("Target", target_label(target)), ("Output", self.output)]

        if target.get("runs"):
            fields.append(("Run store", str(target["runs"])))

        return fields

    def __repr__(self) -> str:
        """Display the reference's source and identity."""
        if self.kind == "job-output":
            identity = ("Workload digest", self.workload_digest)
        else:
            label = "Content digest" if self.kind == "stored-policy" else "File digest"
            identity = (label, self.to_document()["digest"])
        fields = [*self._display_fields(), identity]
        return "\n".join([f"Reference ({self.output})", *(f"  {key}: {value}" for key, value in fields)])

    def _repr_html_(self) -> str:
        """Jupyter renders an inspectable card, using only immutable reference data."""
        rows = "".join(
            f'<tr><th style="text-align:left;vertical-align:top;padding:4px 12px 4px 0">{escape(key)}</th>'
            f'<td style="text-align:left;padding:4px 0;overflow-wrap:anywhere">{escape(value)}</td></tr>'
            for key, value in self._display_fields()
        )
        configuration = escape(json.dumps(self.to_document(), indent=2, ensure_ascii=False))
        detail_title = {"job-output": "Target configuration and identity", "file": "File identity",
                        "stored-policy": "Source and content identity"}[self.kind]
        return (
            '<div style="border:1px solid #888;border-radius:8px;padding:12px 16px;max-width:800px">'
            f'<strong>{escape(self.output.capitalize())} reference</strong>'
            '<div style="font-size:0.85em;opacity:0.75;margin:4px 0 8px">Reference only; no status query.</div>'
            f'<table style="border-collapse:collapse;width:100%">{rows}</table>'
            f'<details style="margin-top:10px"><summary>{detail_title}</summary>'
            f'<pre style="white-space:pre-wrap;overflow-wrap:anywhere">{configuration}</pre></details></div>'
        )

    def result(self, timeout: float | None = None, *, path: str | os.PathLike[str] | None = None) -> ResultT_co:
        """Load T, optionally saving its files into a new local directory first."""
        if self.kind == "stored-policy":
            outputs._file_timeout(timeout)
            if path is not None:
                return self.save(path=path).result()
            return cast(ResultT_co, self._storage()._load_reference(self, expected_game=self.training.game))

        if self.kind == "file":
            reference = self if path is None else self.save(path=path, timeout=timeout)
            return cast(ResultT_co, outputs.file_result(reference, timeout=timeout))

        handle, workload = self._open()
        destination = outputs._result_path(path)
        result = result_from_document(handle.result(timeout))
        return cast(ResultT_co, outputs.load_result(result, workload, handle=handle, path=destination))

    def save(self, path: str | os.PathLike[str], *, timeout: float | None = None) -> Reference[ResultT_co]:
        """Wait and copy files to a new local directory, without loading the value."""
        if self.kind == "stored-policy":
            outputs._file_timeout(timeout)
            return self._storage()._save_reference(self, path)

        if self.kind == "file":
            saved = outputs.save_file(self, path=path, timeout=timeout)
        else:
            handle, workload = self._open()
            destination = outputs._save_path(path)
            result = result_from_document(handle.result(timeout))
            saved = outputs.save_result(result, workload, destination, handle=handle)

        return Reference.from_document(saved.to_document())

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> Self:
        """Pin an existing policy manifest or result document, without loading a model."""
        return cls.from_document(outputs.pin_file(path).to_document())

    def _open(self) -> tuple[SubmittedJob, WorkloadSpec]:
        """Reconnect only at an explicit resolution boundary, verifying producer identity."""
        handle = resolve_job_service(target_from_document(self.target)).open(self.job_id)
        workload = handle.workload
        if workload.digest != self.workload_digest:
            raise JobError("referenced producer workload digest mismatch")

        if self.output != workload.output_kind:
            raise JobError("reference output does not match the producing job")

        return handle, workload

    def _load_policy(self, *, expected_game: EntrySpec) -> PolicyProfile:
        """Wait and load a policy against the consuming game's policy interface."""

        if self.kind == "stored-policy":
            return self._storage()._load_reference(self, expected_game=expected_game)

        if self.kind == "file":
            return policies.load_policy(self._artifact_metadata(), expected_game=expected_game)

        handle, workload = self._open()
        outputs.validate_producer(self, workload)

        result = result_from_document(handle.result())
        _validate_result(result, workload=workload)
        loader = getattr(handle, "load_policy", None)
        if loader is not None:
            return loader(result, expected_game=expected_game)

        return policies.load_policy(result.policy, expected_game=expected_game)

    @property
    def training(self) -> TrainingSpec:
        """The recorded training configuration of a discovered storage policy."""
        return TrainingSpec.from_document(self._source("stored-policy")["training"])

    @property
    def object(self):
        from benchmax.storage._model import ObjectReference

        return ObjectReference(self._source("stored-policy")["digest"])

    def _storage(self):
        from benchmax.storage.reference import _store_from_document

        return getattr(self, "_store", None) or _store_from_document(self._source("stored-policy")["store"])
