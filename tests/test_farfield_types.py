"""The far-field wire types: ``Screen``, ``ScreenField`` and ``FarField``.

``farfield()`` answers with a ``FarField``: the far field of a port's (or a
profile's) mode as a function of direction, its divergence and beam quality,
and, when a ``Screen`` was given, the field on that screen.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
import pytest

from emode import compat
from emode.emodeclient import obj_hook
from emode.free_space import FarField, Screen, ScreenField
from emode.types import ArgumentError, reconstruct, wire_dump


class _WireEncoder(json.JSONEncoder):
    """The solver's encoding of an ndarray, as the wire contract defines it."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            data = base64.b64encode(np.ascontiguousarray(o).data).decode()
            return {"__ndarray__": data, "dtype": str(o.dtype), "shape": o.shape}
        return super().default(o)


def _round_trip(model: Any) -> Any:
    return reconstruct(json.loads(json.dumps(wire_dump(model), cls=_WireEncoder), object_hook=obj_hook))


def _gaussian_far_field(screen: ScreenField | None = None) -> FarField:
    """A paraxial Gaussian's far field, 1/e^2 half-angle 0.1 rad in x and 0.05 in y."""
    ux = np.linspace(-0.4, 0.4, 81)
    uy = np.linspace(-0.3, 0.3, 61)
    UX, UY = np.meshgrid(ux, uy, indexing="ij")
    intensity = np.exp(-2 * UX**2 / 0.1**2 - 2 * UY**2 / 0.05**2)
    field = np.sqrt(intensity).astype(complex)
    return FarField(
        wavelength=1550.0,
        index=1.0,
        ux=ux,
        uy=uy,
        fields={"Ex": field, "Ey": 0 * field, "Ez": -UX * field},
        intensity=intensity,
        radiated_power=0.999,
        divergence=(np.degrees(0.1), np.degrees(0.05)),
        m_squared=(1.0, 1.0),
        waist=(4934.0, 9868.0),
        waist_position=(0.0, 0.0),
        centroid=(0.0, 0.0),
        screen=screen,
    )


def _screen_field() -> ScreenField:
    screen = Screen(distance=1e5, width=2e4, height=1e4, points=(5, 3))
    s = np.linspace(-1e4, 1e4, 5)
    t = np.linspace(-5e3, 5e3, 3)
    S, T = np.meshgrid(s, t, indexing="ij")
    field = np.exp(-(S**2 + T**2) / 1e8).astype(complex)
    return ScreenField(
        screen=screen,
        s=s,
        t=t,
        x=S,
        y=T,
        z=np.full_like(S, 1e5),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        fields={name: field for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")},
        Sn=np.abs(field) ** 2,
        power_through_screen=0.98,
    )


def test_a_far_field_round_trips_over_the_wire() -> None:
    sent = _gaussian_far_field(_screen_field())
    received = _round_trip(sent)
    assert isinstance(received, FarField)
    np.testing.assert_array_equal(received.intensity, sent.intensity)
    np.testing.assert_array_equal(received.fields["Ez"], sent.fields["Ez"])
    assert received.m_squared == (1.0, 1.0)
    assert isinstance(received.screen, ScreenField)
    assert isinstance(received.screen.screen, Screen)
    np.testing.assert_array_equal(received.screen.fields["Hy"], sent.screen.fields["Hy"])
    assert received.screen.power_through_screen == pytest.approx(0.98)


def test_a_far_field_without_a_screen_round_trips() -> None:
    received = _round_trip(_gaussian_far_field())
    assert isinstance(received, FarField)
    assert received.screen is None
    assert received.port is None and received.mode == 0


def test_the_angles_are_the_arcsine_of_the_direction_cosines() -> None:
    ff = _gaussian_far_field()
    np.testing.assert_allclose(ff.theta_x, np.degrees(np.arcsin(ff.ux)))
    np.testing.assert_allclose(ff.theta_y, np.degrees(np.arcsin(ff.uy)))


def test_a_cut_reads_the_intensity_along_one_axis() -> None:
    ff = _gaussian_far_field()
    angle, intensity = ff.cut("x")
    np.testing.assert_allclose(angle, ff.theta_x)
    np.testing.assert_allclose(intensity, ff.intensity[:, 30])
    angle, intensity = ff.cut("y")
    np.testing.assert_allclose(intensity, ff.intensity[40, :])
    with pytest.raises(ArgumentError):
        ff.cut("z")


def test_the_plot_draws_the_map_and_the_cuts() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    fig = _gaussian_far_field(_screen_field()).plot()
    assert len([ax for ax in fig.axes if ax.get_label() != "<colorbar>"]) >= 3
    fig = _gaussian_far_field().plot()
    assert len(fig.axes) >= 2


def test_a_screen_takes_lists_from_json() -> None:
    screen = Screen(distance=1e5, width=1e4, height=1e4, center=[1.0, 2.0], tilt=[5, 0], points=[11, 21])
    assert screen.center == (1.0, 2.0)
    assert screen.tilt == (5.0, 0.0)
    assert screen.points == (11, 21)
    assert Screen(distance=1.0, width=1.0, height=1.0, points=31).points == (31, 31)


@pytest.mark.parametrize(
    "arguments, name",
    [
        ({"distance": 0.0}, "distance"),
        ({"distance": -1.0}, "distance"),
        ({"width": 0.0}, "width"),
        ({"height": -2.0}, "height"),
        ({"tilt": (90.0, 0.0)}, "tilt"),
        ({"tilt": (0.0, -95.0)}, "tilt"),
        ({"points": (1, 11)}, "points"),
    ],
)
def test_a_screen_refuses_a_bad_argument_by_name(arguments: dict, name: str) -> None:
    base = {"distance": 1e5, "width": 1e4, "height": 1e4}
    with pytest.raises(ArgumentError) as error:
        Screen(**{**base, **arguments})
    assert error.value.argument == name


def test_a_default_screen_sends_only_what_it_was_given() -> None:
    dumped = wire_dump(Screen(distance=1e5, width=1e4, height=1e4))
    assert set(dumped) == {"__data_type__", "distance", "width", "height"}


def test_the_new_types_name_the_emode_that_reads_them() -> None:
    for name in ("FarField", "Screen", "ScreenField"):
        assert compat.TYPE_SINCE[name] == "1.0.8"
