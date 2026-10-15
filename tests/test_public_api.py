"""Supported import paths used by callers."""

from benchmax import Game, GameInstance, PolicyProfile, ScalarPolicyProfile, TimeStep
from benchmax.core import Game as CoreGame
from benchmax.core import GameInstance as CoreGameInstance
from benchmax.core import TimeStep as CoreTimeStep
from benchmax.policy import PolicyProfile as Profile
from benchmax.policy import ScalarPolicyProfile as ScalarProfile
from benchmax.training import train, train_ippo, train_mappo


def test_public_imports_resolve() -> None:
    assert (Game, GameInstance, TimeStep) == (CoreGame, CoreGameInstance, CoreTimeStep)
    assert (PolicyProfile, ScalarPolicyProfile) == (Profile, ScalarProfile)
    assert all(callable(function) for function in (train, train_ippo, train_mappo))
