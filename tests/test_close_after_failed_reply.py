"""A reply the client cannot decode must not leave the session waiting at close.

``close()`` drains one reply when a call was cut short mid-read (Ctrl-C), since
EMode answers that call before it can see ``EM_close``. A reply that arrived in
full and only failed to decode leaves nothing to drain, so ``close()`` must go
straight to ``EM_close``, or the script hangs at exit (#90).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from emodeconnection import EMode
from emodeconnection.types import EModeError


class _Proc:
    def wait(self, timeout: float | None = None) -> int:
        return 0


class _Client:
    """The socket client, answering each read with the next reply.

    ``reads`` records which call each read answered. A read with no reply left
    is a read that would block forever.
    """

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.sent: list[str] = []
        self.reads: list[str] = []
        self.connected = True

    def send(self, data: dict) -> None:
        self.sent.append(data["function"])

    def recv(self) -> Any:
        if not self.replies:
            raise AssertionError("close() waited for a reply that EMode never sends")
        self.reads.append(self.sent[-1])
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def close(self) -> None:
        self.connected = False


def _session(*replies: Any) -> tuple[EMode, _Client]:
    em = EMode.__new__(EMode)
    em._dsim = "emode"
    em._save = False
    em._in_flight = False
    em._emode_version_cache = "1.0.7"
    em.running = True
    em._proc = _Proc()  # type: ignore[assignment]
    client = _Client(*replies)
    em._client = client  # type: ignore[assignment]
    em.in_ipython = lambda: False  # type: ignore[method-assign]
    return em, client


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("a field this client does not accept"),
        AttributeError("no such attribute"),
        json.JSONDecodeError("Expecting value", "", 0),
    ],
    ids=["validation", "attribute", "json"],
)
def test_close_after_a_reply_that_failed_to_decode_goes_straight_to_close(
    failure: Exception,
) -> None:
    em, client = _session(failure, None)
    with pytest.raises(type(failure)):
        em.report()
    em.close()
    assert client.reads == ["EM_report", "EM_close"]
    assert not em.running


def test_close_after_an_error_reply_goes_straight_to_close() -> None:
    em, client = _session(EModeError("bad argument"), None)
    with pytest.raises(EModeError):
        em.report()
    em.close()
    assert client.reads == ["EM_report", "EM_close"]


def test_close_after_an_interrupted_read_drains_the_reply_first() -> None:
    em, client = _session(KeyboardInterrupt(), "the interrupted reply", None)
    with pytest.raises(KeyboardInterrupt):
        em.report()
    em.close()
    assert client.reads == ["EM_report", "EM_report", "EM_close"]
