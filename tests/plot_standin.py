"""A stand-in session for the plot tests: an ``EMode`` with no solver behind it.

The socket client answers each call by its function name from a table, records
what was sent, and can hold a call open (``hold``) to test the call lock.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

from emode import EMode

from plot_fixtures import zx_view

#: What a 1.0.9 EMode answers ``version()`` with.
VERSION_1_0_9 = {"emode": "1.0.9", "protocol": 1, "plot": "view"}
#: What a 1.0.8 EMode answers ``version()`` with: no ``'plot'``.
VERSION_1_0_8 = {"emode": "1.0.8", "protocol": 1}


class Proc:
    def wait(self, timeout: Optional[float] = None) -> int:
        return 0


class Client:
    """The socket client: each request answered by `answers[function]` (a value or a callable)."""

    def __init__(self, answers: Dict[str, Any]) -> None:
        self.answers = answers
        self.sent: List[Dict[str, Any]] = []
        self.connected = True
        self.ndarray_wire = False
        self.events: List[str] = []
        self._pending: Optional[Dict[str, Any]] = None
        self.hold: Optional[threading.Event] = None

    def send(self, data: Dict[str, Any]) -> None:
        self.events.append(f"send {data['function']}")
        self.sent.append(dict(data))
        self._pending = data

    def recv(self) -> Any:
        data = self._pending
        assert data is not None, "a reply was read with no request in flight"
        if self.hold is not None and data["function"] == "EM_hold":
            self.hold.wait(5)
        self.events.append(f"recv {data['function']}")
        self._pending = None
        answer = self.answers.get(data["function"])
        if callable(answer):
            return answer(data)
        return answer

    def close(self) -> None:
        self.connected = False

    def calls(self, function: str) -> List[Dict[str, Any]]:
        return [d for d in self.sent if d["function"] == function]


def session(
    version: Optional[Dict[str, Any]] = VERSION_1_0_9,
    plot: Callable[[Dict[str, Any]], Any] = lambda request: zx_view(),
    ipython: bool = False,
) -> "tuple[EMode, Client]":
    em = EMode.__new__(EMode)
    em._dsim = "emode"
    em._save = False
    em._in_flight = False
    em.running = True
    em._proc = Proc()  # type: ignore[assignment]
    client = Client({"EM_version": version, "EM_plot": plot, "EM_close": None, "EM_hold": "held"})
    em._client = client  # type: ignore[assignment]
    em._ipython = ipython
    em.in_ipython = lambda: ipython  # type: ignore[method-assign]
    return em, client
