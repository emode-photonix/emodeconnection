"""``ModeDict``, a result keyed by mode index.

A shape returned by ``get_shape()`` carries the confinement factor of each mode
under ``metadata['confinement']``. JSON object keys are always strings, so a
plain dict reached the client keyed by ``'0'``, ``'1'``, and
``shape['metadata']['confinement'][0]`` raised ``KeyError``. ``ModeDict`` carries
the mode indices through validation, which turns them back into integers.

On the wire it is a flat object, the mode indices at the top level beside the
``__data_type__`` tag, so a client that does not know the type still reads
``['0']`` as it always did.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from emodeconnection import ModeDict
from emodeconnection.types import _TYPE_REGISTRY, reconstruct

_FACTORS = {0: 0.81, 1: 0.42, 3: 0.05}

# get_shape('core') after confinement(), as an EMode 1.0.6 server writes it:
# the shape's metadata holds a plain dict whose mode indices JSON made strings.
_SHAPE_1_0_6 = {
    "__data_type__": "Shape",
    "name": "core",
    "metadata": {"confinement": {"0": 0.81, "1": 0.42, "3": 0.05}},
}


def _over_the_wire(value: Any) -> Any:
    """What the solver's reply becomes on this side: JSON, then ``reconstruct``."""
    payload = value.model_dump() if hasattr(value, "model_dump") else value
    return reconstruct(json.loads(json.dumps(payload)))


def test_modes_are_indexed_by_integer_after_the_wire() -> None:
    got = _over_the_wire(ModeDict(data=_FACTORS))
    assert isinstance(got, ModeDict)
    assert got[0] == pytest.approx(0.81)
    assert got[3] == pytest.approx(0.05)
    assert list(got) == [0, 1, 3]


def test_it_reads_like_the_dict_it_replaces() -> None:
    got = ModeDict(data=_FACTORS)
    assert isinstance(got, Mapping)
    assert len(got) == 3
    assert 1 in got
    assert 2 not in got
    assert dict(got) == _FACTORS
    assert got == _FACTORS
    assert got.get(2) is None


def test_the_wire_form_puts_the_modes_at_the_top_level() -> None:
    dumped = json.loads(json.dumps(ModeDict(data=_FACTORS).model_dump()))
    assert dumped == {"__data_type__": "ModeDict", "0": 0.81, "1": 0.42, "3": 0.05}


def test_inside_a_shape_s_metadata_it_is_rebuilt() -> None:
    shape = dict(_SHAPE_1_0_6, metadata={"confinement": ModeDict(data=_FACTORS).model_dump()})
    got = _over_the_wire(shape)
    assert got["metadata"]["confinement"][0] == pytest.approx(0.81)


def test_a_client_that_does_not_know_the_type_reads_it_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An emodeconnection from before ``ModeDict`` leaves the tag as a plain dict."""
    payload = json.dumps(ModeDict(data=_FACTORS).model_dump())
    monkeypatch.delitem(_TYPE_REGISTRY, "ModeDict")
    got = reconstruct(json.loads(payload))
    assert type(got) is dict
    assert got["0"] == pytest.approx(0.81)


def test_a_1_0_6_server_reply_is_still_a_plain_dict() -> None:
    got = _over_the_wire(_SHAPE_1_0_6)
    assert got["metadata"]["confinement"] == {"0": 0.81, "1": 0.42, "3": 0.05}


def test_it_is_read_only() -> None:
    got = ModeDict(data=_FACTORS)
    with pytest.raises(TypeError):
        got[0] = 1.0  # type: ignore[index]
