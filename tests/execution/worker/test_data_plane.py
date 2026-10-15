import mmap

import msgspec
import numpy as np
import pytest

from benchmax.execution.worker import PolicyBatch
from benchmax.execution.worker.data_plane import ArrayDescriptor, DataPlaneError, PackedPolicyBatch, pack_policy_batch


def _batch() -> PolicyBatch:
    observations = np.arange(24, dtype=np.float64).reshape(3, 8)[:, ::2]
    mask = np.array([[True, False, True], [False, True, True], [True, True, False]], dtype=bool)
    return PolicyBatch(
        player=0, observations=observations, action_space=(0, 1, 2), legal_action_mask=mask,
        probabilities=np.empty((3, 3), dtype=np.float64)
    )


def test_policy_batches_round_trip_through_aligned_shared_views() -> None:
    batch = _batch()

    with mmap.mmap(-1, 4096) as arena:
        packed = pack_policy_batch(arena, batch)
        decoded = msgspec.json.decode(msgspec.json.encode(packed), type=PackedPolicyBatch)

        assert decoded == packed
        assert packed.observations.offset % 64 == 0
        assert packed.legal_action_mask.offset % 64 == 0
        assert packed.probabilities.offset % 64 == 0
        np.testing.assert_array_equal(packed.observations.view(arena, writable=False), batch.observations)
        np.testing.assert_array_equal(packed.legal_action_mask.view(arena, writable=False), batch.legal_action_mask)

        output = packed.probabilities.view(arena)
        assert np.isnan(output).all()
        output[:] = 1 / 3
        np.copyto(batch.probabilities, output)
        np.testing.assert_array_equal(batch.probabilities, np.full((3, 3), 1 / 3))


def test_a_batch_larger_than_the_bounded_arena_is_rejected() -> None:
    with mmap.mmap(-1, 128) as arena, pytest.raises(DataPlaneError, match="needs .* arena has 128"):
        pack_policy_batch(arena, _batch())


@pytest.mark.parametrize(
    "value",
    (
        {},
        {"offset": False, "shape": (2,), "dtype": "<f8", "nbytes": 16},
        {"offset": 0, "shape": (True,), "dtype": "<f8", "nbytes": 8},
        {"offset": 0, "shape": (2,), "dtype": "<f8", "nbytes": -1},
        {"offset": 0, "shape": (2,), "dtype": "<f8", "nbytes": 16, "extra": 0},
        {"offset": 0, "shape": (2,), "dtype": "invalid", "nbytes": 16},
        {"offset": 0, "shape": (2,), "dtype": "<f8", "nbytes": 3},
        {"offset": -1, "shape": (2,), "dtype": "<f8", "nbytes": 16},
        {"offset": 0, "shape": (-2,), "dtype": "<f8", "nbytes": 16},
        {"offset": 0, "shape": (2,), "dtype": "object", "nbytes": 16},
    ),
)
def test_invalid_array_descriptors_are_rejected(value: object) -> None:
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(msgspec.json.encode(value), type=ArrayDescriptor)


def test_descriptors_must_be_aligned_and_within_the_arena() -> None:
    with mmap.mmap(-1, 128) as arena:
        with pytest.raises(DataPlaneError, match="aligned"):
            ArrayDescriptor(1, (1,), "<f8", 8).view(arena)

        with pytest.raises(DataPlaneError, match="beyond"):
            ArrayDescriptor(128, (1,), "<f8", 8).view(arena)


def test_policy_batch_descriptors_cannot_overlap() -> None:
    descriptor = {"offset": 0, "shape": (2,), "dtype": "<f8", "nbytes": 16}
    value = {
        "observations": descriptor,
        "legal_action_mask": {"offset": 64, "shape": (2,), "dtype": "|b1", "nbytes": 2},
        "probabilities": descriptor,
    }

    with pytest.raises(msgspec.ValidationError, match="overlap"):
        msgspec.json.decode(msgspec.json.encode(value), type=PackedPolicyBatch)


def test_descriptor_size_validation_uses_unbounded_integer_arithmetic() -> None:
    value = {"offset": 0, "shape": (2**62, 8), "dtype": "<f8", "nbytes": 0}
    with pytest.raises(msgspec.ValidationError, match="expected"):
        msgspec.json.decode(msgspec.json.encode(value), type=ArrayDescriptor)
