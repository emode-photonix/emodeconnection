"""``call()`` holds one lock, so the plot tab's thread and the script never interleave.

The socket carries one request and its reply at a time. The tab asks EMode
for data from the local server's thread while the script may be in a solve,
so a second call waits for the first rather than sending into its reply.
"""
from __future__ import annotations

import logging
import threading
import time

from plot_standin import session


def test_two_threads_never_interleave_on_the_socket():
    em, client = session()
    client.hold = threading.Event()
    first = threading.Thread(target=em.call, args=("EM_hold",))
    first.start()
    time.sleep(0.1)
    second = threading.Thread(target=em.call, args=("EM_version",))
    second.start()
    time.sleep(0.1)
    # the second call has not sent while the first waits for its reply
    assert client.events == ["send EM_hold"]
    client.hold.set()
    first.join(5)
    second.join(5)
    assert client.events == ["send EM_hold", "recv EM_hold", "send EM_version", "recv EM_version"]


def test_a_call_during_a_held_lock_waits_rather_than_failing():
    em, client = session()
    results = []
    with em._lock():
        t = threading.Thread(target=lambda: results.append(em.call("EM_version")))
        t.start()
        time.sleep(0.1)
        assert results == [] and t.is_alive()
    t.join(5)
    assert results == [{"emode": "1.0.9", "protocol": 1, "plot": "view"}]


def test_the_lock_is_reentrant_for_a_call_made_inside_a_call():
    em, client = session()
    with em._lock():
        assert em.call("EM_version")["emode"] == "1.0.9"


def test_the_debug_log_never_carries_a_password(caplog):
    em, client = session()
    client.answers["EM_login"] = "ok"
    with caplog.at_level(logging.DEBUG, logger="emode.emode"):
        em.call("EM_login", email="a@b.c", password="hunter2-secret")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "EM_login" in text and "a@b.c" in text
    assert "hunter2-secret" not in text
    assert client.sent[-1]["password"] == "hunter2-secret"
