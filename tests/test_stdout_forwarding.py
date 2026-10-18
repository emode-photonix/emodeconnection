"""The thread that carries the solver's stdout into a notebook cell.

Each test runs a real child process whose stdout is a byte pipe, hands that
pipe to ``_forward_stdout`` on a thread, and records what the forwarder prints
through a stand-in for ``builtins.print``. The child stands in for the solver,
so nothing here needs a solver install or a license.
"""

from __future__ import annotations

import builtins
import subprocess
import sys
import textwrap
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from emodeconnection.emodeconnection import _forward_stdout

# Generous: a forwarder that works returns within milliseconds of the child
# exiting. One that does not (a line reader on a byte pipe never sees its
# text sentinel at EOF) would spin forever, so the join is bounded.
_FORWARD_TIMEOUT = 10.0
_MAX_RECORDED = 10_000


@dataclass
class _Recorder:
    """Stand-in for ``print`` that keeps the text of each call, in order."""

    calls: list[str] = field(default_factory=list)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        if len(self.calls) >= _MAX_RECORDED:
            return
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        self.calls.append(sep.join(str(a) for a in args) + end)

    @property
    def text(self) -> str:
        return "".join(self.calls)


@dataclass
class _Forwarded:
    recorder: _Recorder
    finished: bool
    proc: subprocess.Popen[bytes]


def _forward_child(code: str, monkeypatch: pytest.MonkeyPatch) -> _Forwarded:
    """Run ``code`` in a child Python and forward its stdout until EOF."""
    recorder = _Recorder()
    monkeypatch.setattr(builtins, "print", recorder)
    proc = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code)],
        stdout=subprocess.PIPE,
        bufsize=0,
    )
    assert proc.stdout is not None
    thread = threading.Thread(target=_forward_stdout, args=(proc.stdout,), daemon=True)
    thread.start()
    thread.join(_FORWARD_TIMEOUT)
    finished = not thread.is_alive()
    proc.wait(_FORWARD_TIMEOUT)
    if not finished:
        # Unblock a stuck forwarder so it does not outlive the test.
        proc.stdout.close()
        thread.join(1.0)
    return _Forwarded(recorder, finished, proc)


def test_forward_stdout_prints_lines_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    code = """
        for i in range(20):
            print(f"line {i}", flush=True)
    """
    forwarded = _forward_child(code, monkeypatch)
    assert forwarded.finished
    assert forwarded.recorder.text == "".join(f"line {i}\n" for i in range(20))


def test_forward_stdout_streams_partial_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """A progress message without a newline shows up before its line completes."""
    code = """
        import time
        print("Solving modes...", end="", flush=True)
        time.sleep(0.5)
        print(" completed in 0.1 sec", flush=True)
    """
    forwarded = _forward_child(code, monkeypatch)
    assert forwarded.finished
    assert forwarded.recorder.calls[0] == "Solving modes..."
    assert forwarded.recorder.text == "Solving modes... completed in 0.1 sec\n"


def test_forward_stdout_decodes_split_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    """A multibyte character split across two reads is decoded whole."""
    code = """
        import sys, time
        out = sys.stdout.buffer
        out.write(bytes([0xCE]))
        out.flush()
        time.sleep(0.3)
        out.write(bytes([0xBC, 0x0A]))
        out.flush()
    """
    forwarded = _forward_child(code, monkeypatch)
    assert forwarded.finished
    assert forwarded.recorder.text == "μ\n"
    assert "�" not in forwarded.recorder.text


def test_forward_stdout_closes_pipe_on_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    forwarded = _forward_child('print("done")', monkeypatch)
    assert forwarded.finished
    assert forwarded.recorder.text == "done\n"
    assert forwarded.proc.stdout is not None
    assert forwarded.proc.stdout.closed
