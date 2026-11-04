"""A result whose ``__data_type__`` this emodeconnection does not know.

An emodeconnection older than its EMode receives result types it has no class
for. Such a result still arrives as a dict, so code that treats it as one keeps
working, but reading an attribute from it says which type it is and how to get
an emodeconnection that knows it, in place of ``'dict' object has no attribute``.
"""

from __future__ import annotations

import copy
import json
import logging
import pickle

import pytest

from emodeconnection.geometry import LayoutReport
from emodeconnection.types import _TYPE_REGISTRY, UnreadWireType, reconstruct

_FUTURE = {"__data_type__": "FutureReport", "value": 1, "names": ["a", "b"]}


def test_an_unknown_type_is_a_dict_equal_to_what_was_sent() -> None:
    got = reconstruct(dict(_FUTURE))
    assert isinstance(got, dict)
    assert isinstance(got, UnreadWireType)
    assert got == _FUTURE
    assert _FUTURE == got


def test_a_missing_attribute_names_the_type_and_the_fix() -> None:
    got = reconstruct(dict(_FUTURE))
    with pytest.raises(AttributeError) as info:
        got.max_closure()
    message = str(info.value)
    assert "FutureReport" in message
    assert "max_closure" in message
    assert "pip install -U emodeconnection" in message


def test_the_dict_methods_still_work() -> None:
    got = reconstruct(dict(_FUTURE))
    assert got.get("value") == 1
    assert got.get("missing", 7) == 7
    assert dict(got.items()) == _FUTURE
    assert list(got.keys()) == list(_FUTURE.keys())
    assert got["names"] == ["a", "b"]
    assert len(got) == 3


def test_it_serializes_as_json() -> None:
    got = reconstruct(dict(_FUTURE))
    assert json.loads(json.dumps(got)) == _FUTURE


def test_it_pickles_and_copies() -> None:
    got = reconstruct(dict(_FUTURE))
    for other in (pickle.loads(pickle.dumps(got)), copy.copy(got), copy.deepcopy(got)):
        assert type(other) is UnreadWireType
        assert other == _FUTURE
    assert copy.deepcopy(got)["names"] is not got["names"]


def test_hasattr_is_false_for_a_missing_attribute() -> None:
    got = reconstruct(dict(_FUTURE))
    assert not hasattr(got, "max_closure")
    assert not hasattr(got, "_repr_html_")


def test_a_known_type_is_unaffected() -> None:
    report = LayoutReport().model_dump()
    got = reconstruct(report)
    assert type(got) is LayoutReport


def test_a_plain_dict_without_a_tag_stays_a_plain_dict() -> None:
    got = reconstruct({"value": 1, "inner": {"x": 2}})
    assert type(got) is dict
    assert type(got["inner"]) is dict


def test_unknown_types_nested_in_lists_and_dicts() -> None:
    got = reconstruct({"results": [dict(_FUTURE), {"deep": dict(_FUTURE)}], "one": dict(_FUTURE)})
    assert type(got) is dict
    assert type(got["results"][0]) is UnreadWireType
    assert type(got["results"][1]) is dict
    assert type(got["results"][1]["deep"]) is UnreadWireType
    assert type(got["one"]) is UnreadWireType
    assert got == {"results": [_FUTURE, {"deep": _FUTURE}], "one": _FUTURE}


def test_an_unknown_type_inside_an_unknown_type() -> None:
    got = reconstruct({"__data_type__": "Outer", "inner": dict(_FUTURE)})
    assert type(got) is UnreadWireType
    assert type(got["inner"]) is UnreadWireType
    with pytest.raises(AttributeError, match="Outer"):
        got.anything


def test_a_type_this_client_knows_but_cannot_read_is_also_helpful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A newer EMode may send a value this client's class refuses.
    got = reconstruct({"__data_type__": "PulsedProcess", "raman_mode": "future"})
    assert type(got) is UnreadWireType
    assert got == {"__data_type__": "PulsedProcess", "raman_mode": "future"}
    with pytest.raises(AttributeError, match="pip install -U emodeconnection"):
        got.max_rank


def test_the_warning_is_logged_once_per_type(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="emodeconnection.types")
    for _ in range(3):
        reconstruct({"__data_type__": "OnceOnlyType", "value": 1})
    reconstruct({"__data_type__": "AnotherOnceOnlyType", "value": 1})
    messages = [r.getMessage() for r in caplog.records]
    assert sum("OnceOnlyType" in m and "AnotherOnceOnlyType" not in m for m in messages) == 1
    assert sum("AnotherOnceOnlyType" in m for m in messages) == 1
    assert all("pip install -U emodeconnection" in m for m in messages)


def test_a_type_missing_from_this_client_reads_as_an_unread_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The customer's case: a client from before ``LayoutReport`` existed."""
    payload = json.loads(json.dumps(LayoutReport().model_dump()))
    monkeypatch.delitem(_TYPE_REGISTRY, "LayoutReport")
    report = reconstruct(payload)
    with pytest.raises(AttributeError, match="LayoutReport.*pip install -U emodeconnection"):
        report.max_closure()
