"""``python -m emode``: the command line of the client, and its ``login`` subcommand."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


import emode.__main__ as entry
from emode.types import LicenseError

ROOT = Path(__file__).resolve().parents[1]
HEADLESS_MESSAGE = "EMode is not signed in. Run `emode login` on a machine with a browser."


def _run(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "packaging" / "emodeconnection" / "src"), str(ROOT)]))
    env["EMODE_HEADLESS"] = "1"
    return subprocess.run([sys.executable, "-m", "emode", *args], capture_output=True, text=True, env=env, timeout=60)


def test_help_exits_zero_and_names_login() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "login" in result.stdout


def test_no_subcommand_prints_help_without_launching_anything() -> None:
    result = _run()
    assert result.returncode == 0
    assert "login" in result.stdout


def test_login_without_a_browser_prints_the_instructions_instead_of_failing(monkeypatch, capsys) -> None:
    def refuse(*args, **kwargs):
        raise LicenseError(HEADLESS_MESSAGE, None)

    monkeypatch.setattr(entry, "EMode", refuse)
    assert entry.main(["login"]) == 1
    assert HEADLESS_MESSAGE in capsys.readouterr().err


def test_login_closes_the_session_it_opened(monkeypatch, capsys) -> None:
    calls: list[str] = []

    class Session:
        def __init__(self, **kwargs) -> None:
            calls.append(f"open {sorted(kwargs)}")

        def close(self, **kwargs) -> None:
            calls.append("close")

    monkeypatch.setattr(entry, "EMode", Session)
    assert entry.main(["login"]) == 0
    assert calls[-1] == "close"
    assert "signed in" in capsys.readouterr().out.lower()
