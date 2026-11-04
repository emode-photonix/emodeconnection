"""The greeting the client sends when it connects.

It reads ``emodeconnection/<version> connected with <environment>``. An EMode
takes the last space-separated token as the client's environment, so that token
must stay what it was before the version was added, and an EMode that reads
the version takes it from the first token.
"""

from __future__ import annotations

import builtins
import importlib.metadata

import pytest

from emodeconnection import emodeclient
from emodeconnection.emodeclient import EModeClient
from emodeconnection.types import client_version


def _greeting(monkeypatch: pytest.MonkeyPatch, environ: dict[str, str]) -> str:
    monkeypatch.setattr(emodeclient.os, "environ", dict(environ))
    client = EModeClient.__new__(EModeClient)  # no socket, no solver
    return client.get_connection_string()


def test_the_client_version_is_the_installed_one() -> None:
    assert client_version() == importlib.metadata.version("emodeconnection")


def test_the_client_version_falls_back_when_there_is_no_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert client_version() == "unknown"


@pytest.mark.parametrize(
    "environ, environment",
    [
        ({}, "Python-cmd"),
        ({"VSCODE_PID": "1"}, "Python-VSCode"),
        ({"SPYDER_ARGS": "[]"}, "Python-Spyder"),
        ({"JPY_PARENT_PID": "1"}, "Python-Jupyter"),
        ({"PYCHARM_HOSTED": "1"}, "Python-PyCharm"),
        ({"HOME": "/home/user"}, "Python-IDLE"),
    ],
)
def test_the_greeting_carries_the_version_and_keeps_the_environment_last(
    monkeypatch: pytest.MonkeyPatch, environ: dict[str, str], environment: str
) -> None:
    greeting = _greeting(monkeypatch, environ)
    version = importlib.metadata.version("emodeconnection")
    assert greeting == f"emodeconnection/{version} connected with {environment}"
    assert greeting.startswith(f"emodeconnection/{version} ")
    assert greeting.split(" ")[-1] == environment


def test_inside_ipython_the_environment_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtins, "get_ipython", lambda: object(), raising=False)
    greeting = _greeting(monkeypatch, {"JPY_PARENT_PID": "1"})
    assert greeting.split(" ")[-1] == "iPython-Jupyter"
    assert greeting.startswith("emodeconnection/")


def test_the_version_token_has_no_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    first, *_ = _greeting(monkeypatch, {}).split(" ")
    assert first == f"emodeconnection/{client_version()}"
    assert len(_greeting(monkeypatch, {}).split(" ")) == 4

