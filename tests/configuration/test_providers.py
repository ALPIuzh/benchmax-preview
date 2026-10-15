import subprocess
import sys
import textwrap

import pytest

from benchmax.providers import Provider, list_providers, require_provider
from benchmax.registry import _inspect_entry


def test_lists_known_providers_and_whether_they_are_installed() -> None:
    providers = list_providers()

    assert set(providers) == {"openspiel", "exp_a_spiel", "benchmarl"}
    assert providers["openspiel"] is True


def test_probing_availability_does_not_import_the_native_binding() -> None:
    program = textwrap.dedent(
        """
        import sys
        from benchmax.providers import PROVIDERS

        binding = "benchmax.integrations.exp_a_spiel._exp_a_spiel"
        assert binding not in sys.modules
        assert PROVIDERS["exp_a_spiel"].available is True
        assert binding not in sys.modules
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_unknown_provider_reports_the_known_ones() -> None:
    with pytest.raises(ValueError, match="unknown provider 'wumpus'"):
        require_provider("wumpus")


def test_a_provider_without_an_implementation_supplies_no_entries() -> None:
    from benchmax import load_game

    with pytest.raises(ValueError, match="does not supply game entries"):
        load_game("exp_a_spiel.something")


def test_missing_transitive_dependency_has_installation_guidance(monkeypatch):
    from benchmax.integrations.benchmarl import provider

    def missing():
        raise ModuleNotFoundError("No module named 'torchrl'", name="torchrl")

    monkeypatch.setattr(Provider, "available", property(lambda self: True))
    monkeypatch.setattr(provider, "available_entries", missing)
    report = _inspect_entry("algorithm", "mappo")

    assert not report["available"]
    assert "torchrl" in report["error"] and "benchmax[training]" in report["error"]


@pytest.mark.parametrize("group, provider, module", [
    ("algorithm", "benchmarl", "benchmarl"), ("game", "openspiel", "pyspiel")
])
def test_inspection_checks_provider_entries(group, provider, module):
    pytest.importorskip(module)
    report = _inspect_entry(group, f"{provider}.no_such_entry")

    assert not report["available"]
    assert f"unknown {provider} {group}" in report["error"]


@pytest.mark.parametrize("exception", [RuntimeError, ValueError, ImportError])
def test_broken_import_is_a_probe_failure(monkeypatch, exception):
    from benchmax import registry

    original = registry.importlib.import_module

    def broken(name, *args, **kwargs):
        if name == "benchmax.training.cfr":
            raise exception("broken runtime")

        return original(name, *args, **kwargs)

    monkeypatch.setattr(registry.importlib, "import_module", broken)
    with pytest.raises(exception, match="broken runtime"):
        _inspect_entry("algorithm", "cfr")
