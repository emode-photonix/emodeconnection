"""How ``EMode.__init__`` launches the solver process.

The solver is never started here: ``Popen`` is replaced by a recorder, the
socket client and the first call by stubs. What is under test is the launch
itself, in a notebook (stdout piped and forwarded into the cell) and in plain
Python (stdout inherited, so the solver writes to the terminal directly).
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
from typing import Any, BinaryIO, Iterator

import pytest

import emodeconnection.emodeconnection as launch_module
from emodeconnection.emodeconnection import EMode


class _FakeProcess:
    def __init__(self, stdout: BinaryIO | None) -> None:
        self.stdout = stdout

    def wait(self, timeout: float | None = None) -> int:
        return 0


class _PopenRecorder:
    """Records the ``Popen`` call and hands back a process with a real pipe."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.write_end: int | None = None
        self.read_end: BinaryIO | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> _FakeProcess:
        self.calls.append((args, kwargs))
        if kwargs.get("stdout") is subprocess.PIPE:
            read_fd, self.write_end = os.pipe()
            self.read_end = open(read_fd, "rb", buffering=0)
            return _FakeProcess(self.read_end)
        return _FakeProcess(None)

    @property
    def kwargs(self) -> dict[str, Any]:
        assert len(self.calls) == 1, self.calls
        return self.calls[0][1]

    def send_eof(self) -> None:
        """Close the write end, so whatever reads the pipe sees EOF."""
        if self.write_end is not None:
            os.close(self.write_end)
            self.write_end = None

    def finish(self) -> None:
        self.send_eof()
        if self.read_end is not None and not self.read_end.closed:
            self.read_end.close()


class _StubClient:
    connected = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def popen(monkeypatch: pytest.MonkeyPatch) -> Iterator[_PopenRecorder]:
    recorder = _PopenRecorder()
    monkeypatch.setattr(launch_module, "Popen", recorder)
    monkeypatch.setattr(launch_module, "EModeClient", _StubClient)
    monkeypatch.setattr(EMode, "call", lambda self, function, **kwargs: "sim:emode")
    # Keep the test process's own signal handlers and exit hooks out of it.
    monkeypatch.setattr(signal, "signal", lambda signum, handler: None)
    monkeypatch.setattr(atexit, "register", lambda func, *args, **kwargs: func)
    yield recorder
    recorder.finish()


def test_ipython_launch_pipes_stdout(popen: _PopenRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EMode, "in_ipython", lambda self: True)

    em = EMode()

    assert popen.kwargs["stdout"] is subprocess.PIPE
    assert popen.kwargs["bufsize"] == 0
    assert not popen.kwargs.get("text", False)
    assert not popen.kwargs.get("universal_newlines", False)
    assert "encoding" not in popen.kwargs
    assert em._print_thread.is_alive()

    popen.send_eof()
    em._print_thread.join(5.0)
    assert not em._print_thread.is_alive()


def test_plain_python_launch_inherits_stdout(
    popen: _PopenRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(EMode, "in_ipython", lambda self: False)

    em = EMode()

    assert popen.kwargs.get("stdout") is None
    # Not hasattr: EMode.__getattr__ turns any unknown name into a solver call.
    assert "_print_thread" not in vars(em)
