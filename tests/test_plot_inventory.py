"""``PlotInventory``: what an EMode session can plot, for the browser tab's Plot menu (GUI-M4a, emode#1702).

EMode answers ``EM_plot_inventory`` with names and keys only. The client reads
it to grey out the kinds the session cannot show and to say why.
"""
from __future__ import annotations

import json

import pytest

from emode import compat
from emode.emodeclient import obj_hook
from emode.plotting import KindEntry, PlotInventory
from emode.types import reconstruct, wire_dump


def _round_trip(value):
    from emode.types import serialize

    payload = json.dumps(serialize(wire_dump(value), ndarray=True))
    return reconstruct(json.loads(payload, object_hook=obj_hook))


def _inventory() -> PlotInventory:
    return PlotInventory(
        kinds=[
            KindEntry(kind="plot", available=True, summary="2 profiles"),
            KindEntry(kind="smatrix", available=False, reason="run EME() first"),
        ],
        profiles=["FDM", "wg"],
        device="preview",
        gds=False,
        sweep="width, 11 points: effective_index",
        pulse_response=False,
        materials=["Si", "SiO2"],
    )


def test_the_inventory_round_trips_the_wire():
    value = _inventory()
    back = _round_trip(value)
    assert type(back) is PlotInventory
    assert back == value
    assert all(type(k) is KindEntry for k in back.kinds)


def test_an_empty_inventory_has_nothing_to_hold():
    value = PlotInventory(kinds=[])
    assert value.profiles == [] and value.device is None
    assert value.sweep is None and not value.gds and not value.pulse_response
    assert _round_trip(value) == value


def test_a_kind_entry_has_no_reason_by_default():
    entry = KindEntry(kind="material", available=True)
    assert entry.reason is None and entry.summary == ""


def test_an_unknown_device_state_is_refused():
    with pytest.raises(ValueError):
        PlotInventory(kinds=[], device="halfway")


def test_the_two_types_are_versioned():
    for name in ("PlotInventory", "KindEntry"):
        assert compat.TYPE_SINCE[name] == "1.0.9"


def test_an_emode_without_the_inventory_is_refused_before_the_call():
    from plot_standin import VERSION_1_0_9, session

    from emode.types import EModeError
    from emode.ui.session import check_plot_support

    em, _ = session(version={**VERSION_1_0_9, "views": ["plot", "details"]})
    with pytest.raises(EModeError, match="plot inventory needs EMode 1.0.9"):
        check_plot_support(em, "inventory")
    em, _ = session(version={**VERSION_1_0_9, "views": ["plot", "inventory"]})
    check_plot_support(em, "inventory")
