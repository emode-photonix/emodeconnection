"""A port window can ask for its own boundary (emodeconnection#104)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from emode import compat
from emode.port import Bounds
from emode.types import reconstruct, wire_dump

BOUNDARY_FIELDS = (
    "boundary_condition",
    "pml_NSEW_bool",
    "num_pml_layers",
    "pml_thickness",
    "expansion_size",
    "expansion_resolution",
)


def test_every_boundary_field_defaults_to_the_sections_own() -> None:
    bounds = Bounds(window_width=1000, window_height=800)
    for name in BOUNDARY_FIELDS:
        assert getattr(bounds, name) is None


def test_a_window_without_a_boundary_crosses_the_wire_as_before() -> None:
    bounds = Bounds(window_width=1000, window_height=800)
    assert set(wire_dump(bounds)) == {"window_width", "window_height", "__data_type__"}


def test_a_window_with_a_pml_round_trips_and_stays_hashable() -> None:
    bounds = Bounds(
        window_width=1000,
        window_height=800,
        pml_NSEW_bool=(True, False, False, False),
        pml_thickness=(500.0, 0.0, 0.0, 0.0),
        boundary_condition="00",
    )
    back = reconstruct(wire_dump(bounds))
    assert back == bounds
    assert hash(back) == hash(bounds)


def test_a_bad_boundary_condition_is_refused() -> None:
    with pytest.raises(ValidationError):
        Bounds(window_width=1000, window_height=800, boundary_condition="0")


@pytest.mark.parametrize("name", BOUNDARY_FIELDS)
def test_each_boundary_field_needs_emode_1_0_9(name: str) -> None:
    assert compat.FIELD_SINCE[("Bounds", name)] == "1.0.9"
