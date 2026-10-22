"""``Confinement``, the result of ``confinement()``.

``confinement()`` maps each shape name to the confinement factor of each mode,
keyed by mode index. JSON object keys are always strings, so a plain dict
reached the client keyed by ``'0'``, ``'1'``, and ``result['core'][0]`` raised
``KeyError``. ``Confinement`` carries the mode indices through validation,
which turns them back into integers.

On the wire it is a flat object, the shape names at the top level beside the
``__data_type__`` tag, so a client that does not know the type still reads
``result['core']['0']`` as it always did.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from emodeconnection import Confinement, WavelengthDict
from emodeconnection.emodeclient import EModeClient
from emodeconnection.types import _TYPE_REGISTRY, reconstruct

_FACTORS = {"core": {0: 0.81, 1: 0.42}, "clad": {0: 0.19, 1: 0.58}}

# confinement() of a two-mode, two-shape profile, as an EMode 1.0.6 server
# writes it: a stamped plain dict whose mode indices JSON made strings.
_CONFINEMENT_1_0_6 = (
    '{"core": {"0": 0.81, "1": 0.42}, "clad": {"0": 0.19, "1": 0.58}, "__version__": 1}'
)


class _Recorded:
    """Stand-in for the socket: hands back one framed message."""

    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")

    def recv_msg(self) -> bytes:
        return self._payload


def _recv(payload: str) -> Any:
    client = EModeClient.__new__(EModeClient)
    client._recv_msg = _Recorded(payload).recv_msg  # type: ignore[method-assign]
    return client.recv()


def _over_the_wire(value: Any) -> Any:
    """What the solver's reply becomes on this side: JSON, then ``reconstruct``."""
    payload = json.loads(json.dumps(value.model_dump()))
    return reconstruct(payload)


def test_modes_are_indexed_by_integer_after_the_wire() -> None:
    got = _over_the_wire(Confinement(factors=_FACTORS))
    assert isinstance(got, Confinement)
    assert got["core"][0] == pytest.approx(0.81)
    assert got["clad"][1] == pytest.approx(0.58)
    assert set(got["core"]) == {0, 1}


def test_it_reads_like_the_dict_it_replaces() -> None:
    got = Confinement(factors=_FACTORS)
    assert isinstance(got, Mapping)
    assert list(got) == ["core", "clad"]
    assert len(got) == 2
    assert "core" in got
    assert dict(got) == _FACTORS
    assert got == _FACTORS
    assert dict(got.items())["clad"] == {0: 0.19, 1: 0.58}


def test_the_wire_form_puts_the_shapes_at_the_top_level() -> None:
    dumped = json.loads(json.dumps(Confinement(factors=_FACTORS).model_dump()))
    assert dumped == {
        "__data_type__": "Confinement",
        "core": {"0": 0.81, "1": 0.42},
        "clad": {"0": 0.19, "1": 0.58},
    }


def test_a_client_that_does_not_know_the_type_reads_it_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An emodeconnection from before ``Confinement`` leaves the tag as a plain dict."""
    payload = json.dumps(Confinement(factors=_FACTORS).model_dump())
    monkeypatch.delitem(_TYPE_REGISTRY, "Confinement")
    got = reconstruct(json.loads(payload))
    assert type(got) is dict
    assert got["core"]["0"] == pytest.approx(0.81)
    assert got["clad"]["1"] == pytest.approx(0.58)


def test_a_shape_named_like_the_field_is_still_a_shape() -> None:
    got = _over_the_wire(Confinement(factors={"factors": {0: 0.5}, "core": {0: 0.5}}))
    assert got["factors"][0] == pytest.approx(0.5)
    assert set(got) == {"factors", "core"}


def test_a_polygon_window_is_keyed_like_a_shape() -> None:
    """``confinement(vertices=...)`` reports the polygon under ``'vs'``."""
    got = _over_the_wire(Confinement(factors={"vs": {0: 0.9, 2: 0.1}}))
    assert got["vs"][2] == pytest.approx(0.1)


def test_one_confinement_per_wavelength_round_trips() -> None:
    per_wavelength = WavelengthDict(
        data={1550.0: Confinement(factors=_FACTORS), 1310.0: Confinement(factors=_FACTORS)}
    )
    got = _over_the_wire(per_wavelength)
    assert isinstance(got[1550], Confinement)
    assert got[1310]["core"][1] == pytest.approx(0.42)


def test_a_1_0_6_server_reply_is_still_a_plain_dict() -> None:
    got = _recv(_CONFINEMENT_1_0_6)
    assert type(got) is dict
    assert got["core"]["0"] == pytest.approx(0.81)


def test_a_new_server_reply_arrives_as_confinement() -> None:
    payload = json.dumps(Confinement(factors=_FACTORS).model_dump() | {"__version__": 1})
    got = _recv(payload)
    assert isinstance(got, Confinement)
    assert got["core"][0] == pytest.approx(0.81)


def test_it_is_read_only() -> None:
    got = Confinement(factors=_FACTORS)
    with pytest.raises(TypeError):
        got["core"] = {0: 1.0}  # type: ignore[index]
