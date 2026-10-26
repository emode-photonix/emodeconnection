"""How ``EModeClient.recv`` reads a reply's protocol version.

The solver stamps every dictionary reply with ``"__version__": 1``. A reply
that is not a dictionary (a scalar, a list, an array or ``None``) carries no
stamp, and an unstamped reply is a plain value that ``recv`` returns as is.
A stamp other than 1 is a protocol this client does not understand, so
``recv`` refuses it rather than returning the raw dictionary.

The solver is never started: ``recv`` reads from a stand-in for the socket
that hands back one recorded message.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from emodeconnection.emodeclient import EModeClient
from emodeconnection.smatrix import SMatrix
from emodeconnection.types import ArgumentError, EModeError

# Replies as an EMode 1.0.6 server writes them, recorded byte for byte.
_SMATRIX_1_0_6 = (
    '{"port_sizes": [1, 1], "data": {"__ndarray__": '
    '"MzMzMzMz4z8AAAAAAAAAAAAAAAAAAAAAmpmZmZmZ6T8AAAAAAAAAAJqZmZmZmek/MzMzMzMz4z8AAAAAAAAAAA==", '
    '"dtype": "complex128", "shape": [2, 2]}, "port_names": ["left", "right"], '
    '"wavelengths": [1550.0, 1550.0], "__data_type__": "SMatrix", "__version__": 1}'
)
_DICT_1_0_6 = (
    '{"n_eff": {"__ndarray__": "zczMzMzMAEBmZmZmZmb+Pw==", "dtype": "float64", "shape": [2]}, '
    '"__version__": 1}'
)
_ARRAY_1_0_6 = '{"__ndarray__": "AAAAAAAA8D8AAAAAAAAAQA==", "dtype": "float64", "shape": [2]}'
_ERROR_1_0_6 = (
    '{"__data_type__": "ArgumentError", "msg": "must be positive", '
    '"function": "EM_init", "argument": "x_resolution", "__version__": 1}'
)


def _client(message: str) -> EModeClient:
    """A client whose next message from the socket is `message`."""
    client = EModeClient.__new__(EModeClient)
    client._recv_msg = lambda: message.encode("utf-8")  # type: ignore[method-assign]
    return client


def test_an_unknown_protocol_version_is_refused() -> None:
    reply = json.dumps({"n_eff": 2.1, "__version__": 2})
    with pytest.raises(EModeError, match="protocol version 2"):
        _client(reply).recv()


def test_an_unknown_version_on_a_tagged_reply_is_refused() -> None:
    reply = json.dumps({"__data_type__": "SMatrix", "port_sizes": [1], "__version__": 3})
    with pytest.raises(EModeError, match="protocol version 3"):
        _client(reply).recv()


@pytest.mark.parametrize("value", [3.5, 7, "TE", True, [1.0, 2.0], ["a", "b"], None])
def test_an_unstamped_plain_value_passes_through(value: Any) -> None:
    assert _client(json.dumps(value)).recv() == value


def test_an_unstamped_array_passes_through() -> None:
    received = _client(_ARRAY_1_0_6).recv()
    assert isinstance(received, np.ndarray)
    np.testing.assert_array_equal(received, [1.0, 2.0])


def test_a_1_0_6_smatrix_reply_is_accepted() -> None:
    received = _client(_SMATRIX_1_0_6).recv()
    assert isinstance(received, SMatrix)
    assert received.port_names == ("left", "right")
    assert received.mode_power is None
    np.testing.assert_allclose(received.data, [[0.6, 0.8j], [0.8j, 0.6]])
    # Without mode_power every mode counts as carrying unit power.
    assert received.unitarity_error() < 1e-12
    assert received.unitarity_error(from_ports="left") < 1e-12


def test_a_1_0_6_dict_reply_is_accepted() -> None:
    received = _client(_DICT_1_0_6).recv()
    assert set(received) == {"n_eff"}
    np.testing.assert_allclose(received["n_eff"], [2.1, 1.9])


def test_a_1_0_6_error_reply_is_raised() -> None:
    with pytest.raises(ArgumentError) as info:
        _client(_ERROR_1_0_6).recv()
    assert info.value.argument == "x_resolution"
