"""NumPy array arguments are sent as `__ndarray__`, as replies are (#88).

`serialize` used to send an array as `np.squeeze(data).tolist()`, which made a
complex array fail in `json.dumps` and dropped a size-1 axis. An EMode of release
1.0.8 or newer reads the same `__ndarray__` encoding in a request that the client
reads in a reply (base64 with dtype and shape). An older EMode does not, so a real
array without a size-1 axis goes as a list to it, and a complex array is refused
before anything is sent.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from emodeconnection import compat
from emodeconnection.emodeclient import obj_hook
from emodeconnection.emodeconnection import EMode
from emodeconnection.types import ArgumentError, EModeError, serialize


def _round_trip(value: Any, *, ndarray: bool = True) -> Any:
    wire = json.dumps(serialize({"x": value}, ndarray=ndarray))
    return json.loads(wire, object_hook=obj_hook)["x"]


@pytest.mark.parametrize(
    "array",
    [
        np.array([1 + 1j, 2.0]),
        np.array([[1.0, 2.0]]),
        np.array([[1.0], [2.0]]),
        np.array([3.5]),
        np.arange(12, dtype=np.int32).reshape(3, 4),
        np.array([True, False]),
        np.arange(6.0).reshape(2, 3)[:, ::2],
    ],
    ids=["complex", "1x2", "2x1", "size-1", "int32", "bool", "non-contiguous"],
)
def test_an_array_round_trips_with_its_dtype_and_shape(array: np.ndarray) -> None:
    back = _round_trip(array)
    assert isinstance(back, np.ndarray)
    assert back.dtype == array.dtype
    assert back.shape == array.shape
    np.testing.assert_array_equal(back, array)


def test_the_default_form_is_still_a_list_of_the_squeezed_array() -> None:
    assert serialize({"x": np.arange(6.0).reshape(2, 3)})["x"] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_an_array_inside_a_list_and_a_dict_is_encoded() -> None:
    back = _round_trip({"a": [np.array([[1.0, 2.0]])]})
    assert back["a"][0].shape == (1, 2)


class _Client:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.sent: list[dict] = []
        self.ndarray_wire = False

    def send(self, data: dict) -> None:
        self.sent.append(json.loads(json.dumps(serialize(data, ndarray=self.ndarray_wire))))

    def recv(self) -> Any:
        return self.replies.pop(0)


def _session(*replies: Any) -> tuple[EMode, _Client]:
    em = EMode.__new__(EMode)
    em._dsim = "emode"
    em._in_flight = False
    client = _Client(list(replies))
    em._client = client  # type: ignore[assignment]
    em.in_ipython = lambda: False  # type: ignore[method-assign]
    return em, client


def test_a_new_emode_is_sent_arrays_as_ndarray() -> None:
    em, client = _session({"emode": "1.0.8", "protocol": 1}, "done")
    array = np.array([[1 + 2j, 3.0]])
    em.call("EM_x", values=array)
    assert [m["function"] for m in client.sent] == ["EM_version", "EM_x"]
    sent = client.sent[1]["values"]
    assert sent["shape"] == [1, 2] and sent["dtype"] == "complex128"
    np.testing.assert_array_equal(obj_hook(sent), array)


def test_a_dev_build_of_1_0_8_counts_as_1_0_8() -> None:
    em, client = _session({"emode": "1.0.8.dev3", "protocol": 1}, "done")
    em.call("EM_x", values=np.zeros(2))
    assert "__ndarray__" in client.sent[1]["values"]


def test_an_older_emode_is_sent_a_real_array_as_a_list() -> None:
    em, client = _session({"emode": "1.0.7", "protocol": 1}, "done")
    em.call("EM_x", values=np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert client.sent[1]["values"] == [[1.0, 2.0], [3.0, 4.0]]


def test_an_emode_older_than_1_0_7_is_sent_a_real_array_as_a_list() -> None:
    em, client = _session(EModeError("'version' is not a supported EMode function"), "done")
    em.call("EM_x", values=np.array([1.0, 2.0]))
    assert client.sent[1]["values"] == [1.0, 2.0]


def test_a_complex_array_is_refused_before_an_older_emode_gets_it() -> None:
    em, client = _session({"emode": "1.0.7", "protocol": 1}, "done")
    with pytest.raises(ArgumentError, match=r"complex.*1\.0\.8") as raised:
        em.call("EM_x", values=np.array([1 + 1j]))
    assert raised.value.argument == "values"
    assert [m["function"] for m in client.sent] == ["EM_version"]


def test_a_call_without_arrays_does_not_ask_the_version() -> None:
    em, client = _session("done")
    em.call("EM_x", values=[1.0, 2.0])
    assert [m["function"] for m in client.sent] == ["EM_x"]


def test_the_first_release_that_reads_ndarray() -> None:
    assert compat.reads_ndarray("1.0.8") and compat.reads_ndarray("1.1.0")
    assert not compat.reads_ndarray("1.0.7") and not compat.reads_ndarray(None)
