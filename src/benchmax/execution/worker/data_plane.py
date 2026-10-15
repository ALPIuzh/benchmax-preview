"""Bounded shared-memory layout for synchronous policy batches."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Annotated, Any, TypeAlias

import msgspec
import numpy as np

from benchmax.execution.worker.oracle import PolicyBatch

_ALIGNMENT = 64
_SUPPORTED_DTYPE_KINDS = "buif"
_NonnegativeInt: TypeAlias = Annotated[int, msgspec.Meta(ge=0)]


class DataPlaneError(RuntimeError):
    """Raised when a batch cannot be represented safely in the shared arena."""


class ArrayDescriptor(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Validated location and NumPy interpretation of one shared array."""

    offset: _NonnegativeInt
    shape: tuple[_NonnegativeInt, ...]
    dtype: str
    nbytes: _NonnegativeInt

    @classmethod
    def for_array(cls, array: np.ndarray, offset: int) -> ArrayDescriptor:
        """Describe a C-order copy of ``array`` beginning at ``offset``."""

        if array.dtype.kind not in _SUPPORTED_DTYPE_KINDS:
            raise DataPlaneError(f"unsupported shared-array dtype {array.dtype!r}")

        return cls(
            offset=offset, shape=tuple(int(size) for size in array.shape), dtype=array.dtype.str,
            nbytes=int(array.size * array.dtype.itemsize)
        )

    def __post_init__(self) -> None:
        try:
            numpy_dtype = np.dtype(self.dtype)
        except TypeError as error:
            raise ValueError(f"invalid shared-array dtype {self.dtype!r}") from error

        if numpy_dtype.kind not in _SUPPORTED_DTYPE_KINDS:
            raise ValueError(f"unsupported shared-array dtype {numpy_dtype!r}")

        expected_nbytes = math.prod(self.shape) * numpy_dtype.itemsize
        if self.nbytes != expected_nbytes:
            raise ValueError(f"array descriptor declares {self.nbytes} bytes; expected {expected_nbytes}")

    def view(self, arena: Any, *, writable: bool = True) -> np.ndarray:
        """Construct a checked NumPy view over ``arena``."""

        arena_bytes = len(arena)

        if self.offset % _ALIGNMENT:
            raise DataPlaneError(
                f"shared-array offset {self.offset} is not {_ALIGNMENT}-byte aligned"
            )

        if self.offset + self.nbytes > arena_bytes:
            raise DataPlaneError(
                f"shared array ends at {self.offset + self.nbytes} bytes, beyond "
                f"the {arena_bytes}-byte arena"
            )

        array = np.ndarray(self.shape, dtype=np.dtype(self.dtype), buffer=arena, offset=self.offset, order="C")

        if not writable:
            array.flags.writeable = False

        return array


class PackedPolicyBatch(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Descriptors for one synchronous query in the shared arena."""

    observations: ArrayDescriptor
    legal_action_mask: ArrayDescriptor
    probabilities: ArrayDescriptor

    def __post_init__(self) -> None:
        _reject_overlapping_arrays(self)


def pack_policy_batch(arena: Any, batch: PolicyBatch, /) -> PackedPolicyBatch:
    """Copy policy inputs into the arena and reserve its probability output."""

    arrays = (
        np.asarray(batch.observations),
        np.asarray(batch.legal_action_mask),
        np.asarray(batch.probabilities),
    )
    descriptors: list[ArrayDescriptor] = []
    offset = 0

    for array in arrays:
        offset = _align(offset)
        descriptor = ArrayDescriptor.for_array(array, offset)

        if offset + descriptor.nbytes > len(arena):
            required = offset + descriptor.nbytes
            raise DataPlaneError(
                f"policy batch needs {required} shared bytes, but the arena has "
                f"{len(arena)}"
            )

        descriptors.append(descriptor)
        offset += descriptor.nbytes

    observations, mask, probabilities = descriptors
    np.copyto(observations.view(arena), arrays[0], casting="no")
    np.copyto(mask.view(arena), arrays[1], casting="no")
    probabilities.view(arena).fill(np.nan)
    return PackedPolicyBatch(observations, mask, probabilities)


def _align(offset: int) -> int:
    return (offset + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _reject_overlapping_arrays(batch: PackedPolicyBatch) -> None:
    intervals = sorted(
        (
            descriptor.offset,
            descriptor.offset + descriptor.nbytes,
            name,
        )
        for name, descriptor in (
            ("observations", batch.observations),
            ("legal_action_mask", batch.legal_action_mask),
            ("probabilities", batch.probabilities),
        )
        if descriptor.nbytes
    )

    for (_, previous_end, previous_name), (start, _, name) in pairwise(intervals):
        if start < previous_end:
            raise ValueError(f"shared arrays {previous_name!r} and {name!r} overlap")
