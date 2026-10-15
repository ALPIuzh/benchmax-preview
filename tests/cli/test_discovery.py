import pytest

from benchmax.providers import PROVIDERS
from tests.cli._helpers import run_cli


def test_providers_reports_each_one_and_whether_it_is_installed(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run_cli(capsys, "providers")

    assert code == 0

    for name in PROVIDERS:
        assert name in out

    assert "installed" in out


def test_games_lists_what_benchmax_offers_not_the_provider_catalogue(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run_cli(capsys, "games")
    listed = out.partition("not configured")[0]

    assert code == 0
    assert "dark_hex_3x3" in listed
    assert "rock_paper_scissors" in listed
    # a provider's own entries are reachable but are not the same kind of name
    assert "openspiel.leduc_poker" not in listed
    assert "openspiel.dark_hex (alias)" in out
    # a variant names its origin without claiming to be an alias
    assert "abrupt_dark_hex_3x3" in out
    variant = next(line for line in out.splitlines() if line.startswith("  abrupt_dark_hex_3x3"))
    assert "openspiel.dark_hex" in variant
    assert "(alias)" not in variant


def test_all_lists_the_entries_a_provider_supplies_beyond_the_configured_ones(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, brief, _ = run_cli(capsys, "games")
    _, full, _ = run_cli(capsys, "games", "--all")

    assert "leduc_poker" not in brief
    assert "leduc_poker" in full
    assert "not configured but loadable" in full


def test_show_explains_an_alias(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run_cli(capsys, "show", "dark_hex_3x3")

    assert code == 0
    assert "based on openspiel.dark_hex" in out
    assert "board_size" in out
    # configuration mechanics are not the user's problem
    assert "_target_" not in out
    assert "_extends_" not in out


def test_show_says_when_an_entry_is_the_providers_own(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run_cli(capsys, "show", "openspiel.amazons")

    assert code == 0
    assert "straight from openspiel" in out
    assert "has not configured it" in out


def test_show_reaches_every_group(capsys: pytest.CaptureFixture[str]) -> None:
    for name, expected in [("kuhn_poker", "game"), ("nash_conv", "evaluator")]:
        code, out, _ = run_cli(capsys, "show", name)
        assert code == 0
        assert expected in out

    pytest.importorskip("benchmarl")
    code, out, _ = run_cli(capsys, "show", "mappo")
    assert code == 0
    assert "algorithm" in out


def test_show_reports_a_missing_provider_rather_than_raising(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # the entry exists; the library behind it does not
    monkeypatch.setattr(type(PROVIDERS["benchmarl"]), "available", property(lambda self: False))

    code, out, _ = run_cli(capsys, "show", "mappo")

    assert code == 1
    assert "not available here" in out
    assert "benchmax[training]" in out


def test_show_fails_on_an_unknown_name(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run_cli(capsys, "show", "wumpus")

    assert code == 1
    assert "nothing named 'wumpus'" in out


def test_algorithms_omit_the_evaluator_column(capsys: pytest.CaptureFixture[str]) -> None:
    _, out, _ = run_cli(capsys, "algorithms")

    # an evaluator is a property of a game, so listing algorithms says nothing
    assert "EVALUATOR" not in out
    assert "benchmarl.mappo (alias)" in out
