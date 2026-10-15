import hashlib
import io
import stat
import sys
import tarfile
import time

import pytest

from benchmax import load_game
from benchmax.evaluation import IncompatibleEvaluatorError
from benchmax.integrations.pons import install as recipe
from benchmax.integrations.pons.opponent import PonsError, PonsOpponent


def executable(tmp_path, body):
    path = tmp_path / "c4solver"
    path.write_text(f"#!{sys.executable}\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def uniform(player, observations, actions, mask):
    return mask / mask.sum(axis=1, keepdims=True)


def test_loading_and_missing_installation_never_download(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    def no_network(*args, **kwargs):
        pytest.fail("evaluation must not download")
    monkeypatch.setattr(recipe.urllib.request, "urlopen", no_network)
    game = load_game("connect_four")
    with pytest.raises(FileNotFoundError, match="benchmax setup pons"):
        game.evaluate(uniform)


def test_installed_engine_reports_results_and_exact_value(tmp_path):
    engine = executable(tmp_path, '''import sys
for line in sys.stdin:
    position = line.strip()
    scores = [6-c if position.count(str(c+1)) < 6 else -1000 for c in range(7)]
    print(" ".join([position, *map(str, scores)]).lstrip(), flush=True)
''')
    game = load_game("connect_four", evaluator={"name": "pons", "episodes": 4, "opponent": {"executable": str(engine)}})
    result = game.evaluate(uniform)
    assert result["game_value"] == {0: 1, 1: -1}
    assert result["is_perfect"] is True
    assert "solver_exploitability" in result
    assert all(sum(result[key][seat] for key in ("win_rates", "draw_rates", "loss_rates")) == 1 for seat in (0, 1))


def test_hung_engine_times_out_and_is_reaped(tmp_path):
    engine = executable(tmp_path, "import time\ntime.sleep(30)")
    session = PonsOpponent(executable=engine, timeout=0.05).open(load_game("connect_four"))
    started = time.monotonic()
    try:
        with pytest.raises(PonsError, match="timed out"):
            session.game_value()
    finally:
        session.close()
    assert time.monotonic() - started < 5
    assert session._process.poll() is not None


@pytest.mark.parametrize("body,match", [
    ("import sys\nprint('engine broke', file=sys.stderr)", "exited before answering.*engine broke"),
    ("import sys\nfor line in sys.stdin: print('garbage', flush=True)", "invalid response"),
    ("import sys\nfor line in sys.stdin: print('x x x x x x x', flush=True)", "non-integer"),
])
def test_bad_engine_responses_fail_cleanly(tmp_path, body, match):
    session = PonsOpponent(executable=executable(tmp_path, body)).open(load_game("connect_four"))
    try:
        with pytest.raises(PonsError, match=match):
            session.game_value()
    finally:
        session.close()


def test_installation_board_mismatch_is_rejected(tmp_path):
    engine = executable(tmp_path, "pass")
    (tmp_path / "board.txt").write_text("4 5\n")
    with pytest.raises(IncompatibleEvaluatorError, match="different board"):
        PonsOpponent(executable=engine).open(load_game("connect_four"))


@pytest.mark.parametrize("name,params", [("openspiel.tic_tac_toe", {}),
                                       ("openspiel.connect_four", {"x_in_row": 3}),
                                       ("openspiel.connect_four", {"rows": 8, "columns": 9})])
def test_unsupported_games_fail_without_fallback(name, params):
    with pytest.raises(IncompatibleEvaluatorError):
        PonsOpponent().open(load_game(name, params=params))


def test_installation_patches_pinned_source_and_reuses_complete_build(tmp_path, monkeypatch):
    archive = tmp_path / "source.tar.gz"
    files = {"Position.hpp": b"WIDTH = 7;\nHEIGHT = 6;\n", "LICENSE": b"AGPL test fixture"}
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in files.items():
            member = tarfile.TarInfo(f"connect4-{recipe.COMMIT}/{name}")
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    monkeypatch.setattr(recipe, "SOURCE_URL", archive.as_uri())
    def build(args, *, cwd, **kwargs):
        assert (cwd / "Position.hpp").read_text() == "WIDTH = 5;\nHEIGHT = 4;\n"
        executable(cwd, "pass")
    monkeypatch.setattr(recipe.subprocess, "run", build)
    directory = tmp_path / "installed"
    path = recipe.install(rows=4, columns=5, directory=directory)
    assert path.is_file()
    assert (directory / "board.txt").read_text() == "4 5\n"
    assert (directory / "LICENSE").is_file()
    monkeypatch.setattr(recipe.urllib.request, "urlopen", lambda *a, **k: pytest.fail("already installed"))
    assert recipe.install(rows=4, columns=5, directory=directory) == path


def test_opening_book_integrity(tmp_path, monkeypatch):
    book = tmp_path / "book"
    book.write_bytes(b"test book")
    monkeypatch.setattr(recipe, "BOOK_URL", book.as_uri())
    monkeypatch.setattr(recipe, "BOOK_SHA256", hashlib.sha256(b"test book").hexdigest())
    recipe._download_book(tmp_path / "copy", 1)
    monkeypatch.setattr(recipe, "BOOK_SHA256", "0" * 64)
    with pytest.raises(OSError, match="checksum mismatch"):
        recipe._download_book(tmp_path / "bad-copy", 1)


def test_cli_setup_dispatches_explicitly(tmp_path, monkeypatch, capsys):
    from benchmax.cli import main
    def install(**kwargs):
        assert kwargs["rows"] == 4 and kwargs["columns"] == 5
        return tmp_path / "c4solver"
    monkeypatch.setattr(recipe, "install", install)
    assert main(["setup", "pons", "--rows", "4", "--columns", "5"]) == 0
    assert "c4solver" in capsys.readouterr().out


def test_storage_and_engines_share_cache_root(tmp_path, monkeypatch):
    from benchmax.storage.huggingface import HuggingFaceStore
    from benchmax.utils.files import cache_directory

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert cache_directory() == tmp_path / "benchmax"
    assert HuggingFaceStore("benchmax/data").cache_dir == cache_directory() / "buckets"
    assert recipe.default_directory(4, 5).is_relative_to(cache_directory())
    assert recipe.default_directory(4, 5) != recipe.default_directory(6, 7)
    assert HuggingFaceStore("benchmax/data", cache_dir=tmp_path / "explicit").cache_dir == tmp_path / "explicit"


def test_setup_cannot_reuse_a_build_for_different_dimensions(tmp_path):
    executable(tmp_path, "pass")
    (tmp_path / "board.txt").write_text("4 5\n")
    with pytest.raises(FileExistsError, match="mismatched"):
        recipe.install(rows=5, columns=5, directory=tmp_path)
