"""Placement-independent policy-query contract for evaluation kernels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from benchmax.policy import PolicyProfile
from benchmax.types import ActionSpace, Player


@dataclass(frozen=True, slots=True)
class PolicyBatch:
    """One policy query and the writable destination for its probabilities.

    Kernel code supplies the input arrays and a destination view, commonly a
    slice of its final strategy array. An in-process oracle writes that view
    directly. Worker runtimes may back the same fields with pooled shared
    memory without changing the kernel's numerical code.
    """

    player: Player
    observations: np.ndarray
    action_space: ActionSpace
    legal_action_mask: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        """Reject structurally invalid buffers before invoking a policy."""

        observations = np.asarray(self.observations)
        mask = np.asarray(self.legal_action_mask)
        probabilities = np.asarray(self.probabilities)

        if observations.ndim < 2:
            raise ValueError("policy observations must include a batch dimension")

        if observations.dtype.kind not in "buif":
            raise TypeError("policy observations must have a numeric or boolean dtype")

        rows = observations.shape[0]
        expected_shape = (rows, len(self.action_space))
        if mask.shape != expected_shape:
            raise ValueError(
                f"legal-action mask has shape {mask.shape!r}; "
                f"expected {expected_shape!r}"
            )

        if not np.issubdtype(mask.dtype, np.bool_):
            raise TypeError("legal-action mask must have boolean dtype")

        if probabilities.shape != expected_shape:
            raise ValueError(
                f"probability destination has shape {probabilities.shape!r}; "
                f"expected {expected_shape!r}"
            )

        if not np.issubdtype(probabilities.dtype, np.floating):
            raise TypeError("probability destination must have floating-point dtype")

        if not probabilities.flags.writeable:
            raise ValueError("probability destination must be writable")

        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "legal_action_mask", mask)
        object.__setattr__(self, "probabilities", probabilities)


class PolicyOracle(ABC):
    """Evaluates policy batches without exposing where the policy executes."""

    @abstractmethod
    def evaluate(self, batch: PolicyBatch, /) -> None:
        """Validate and write probabilities into ``batch.probabilities``."""

        raise NotImplementedError


class PolicyProfileOracle(PolicyOracle):
    """In-process oracle adapting BenchMAX's public policy-profile contract."""

    def __init__(self, policy_profile: PolicyProfile) -> None:
        self._policy_profile = policy_profile

    def evaluate(self, batch: PolicyBatch, /) -> None:
        """Call the profile, validate its result, and fill the destination."""

        probabilities = self._policy_profile(
            batch.player, batch.observations, batch.action_space, batch.legal_action_mask
        )
        np.copyto(batch.probabilities, probabilities, casting="same_kind")
