"""EMode's colormap family (``emode/colormaps.py``): its properties, its names, ``colormap=``.

The family's colors are a drop-in (``colormaps.delivered()``). These tests hold
whatever is delivered to what a plot needs: a sequential map that only gets
lighter, a diverging map as light at -x as at +x, a cyclic map whose ends
meet, line colors far enough apart to tell apart, and the same stops in the
browser (Plotly) and in a static image (matplotlib).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from emode import colormaps
from emode.types import ArgumentError
from emode.ui.figure import PlotOptions

from color_science import color_difference, hex_to_rgb, lightness

mpl = pytest.importorskip("matplotlib")

#: The smallest CIEDE2000 distance between two line colors. 20 is well past
#: the about 2 at which two flat patches look different, since a thin line
#: needs much more than a patch to be told apart.
MIN_LINE_DISTANCE = 20.0

#: L* may not step back by more than this between neighboring samples, which
#: is the rounding of a color to 8 bits.
LIGHTNESS_TOLERANCE = 0.05


def _sampled_lightness(name: str) -> np.ndarray:
    """L* of every entry of the registered matplotlib colormap's lookup table."""
    cmap = colormaps.get(name)
    return np.array([lightness(cmap(i)) for i in range(cmap.N)])


def _stop_lightness(kind: str) -> np.ndarray:
    return np.array([lightness(hex_to_rgb(c)) for _, c in colormaps.stops(kind)])


# ------------------------------------------------------------------ the family's properties


def test_the_sequential_map_gets_lighter_monotonically():
    for values in (_stop_lightness("sequential"), _sampled_lightness("emode-sequential")):
        steps = np.diff(values)
        assert np.all(steps > -LIGHTNESS_TOLERANCE), f"L* steps back by {-steps.min():.3f}"
        assert values[-1] - values[0] > 50, "the map spans too little lightness to read"


def test_the_diverging_map_is_symmetric_in_lightness_about_its_center():
    for values in (_stop_lightness("diverging"), _sampled_lightness("emode-diverging")):
        np.testing.assert_allclose(values, values[::-1], atol=0.5)
        # each half is monotonic, so a value's magnitude reads the same either side of zero
        half = values[: len(values) // 2 + 1]
        steps = np.diff(half)
        assert np.all(steps > -LIGHTNESS_TOLERANCE) or np.all(steps < LIGHTNESS_TOLERANCE)


def test_the_cyclic_maps_ends_meet():
    points = colormaps.stops("cyclic")
    assert points[0][1] == points[-1][1]
    cmap = colormaps.get("emode-cyclic")
    np.testing.assert_allclose(cmap(0.0), cmap(1.0), atol=1 / 255)
    # and it has no other jump: neighboring samples stay close all the way round
    samples = [mpl.colors.to_hex(cmap(f)) for f in np.linspace(0, 1, 257)]
    assert max(color_difference(a, b) for a, b in zip(samples, samples[1:])) < 5


def test_the_line_colors_are_distinguishable():
    lines = colormaps.line_palette()
    assert len(lines) >= 5
    worst = min(
        (color_difference(a, b), a, b) for a, b in itertools.combinations(lines, 2)
    )
    assert worst[0] >= MIN_LINE_DISTANCE, f"{worst[1]} and {worst[2]} are only {worst[0]:.1f} apart"


# ------------------------------------------------------------------ one set of stops, two renderers


@pytest.mark.parametrize("kind", colormaps.KINDS)
def test_matplotlib_and_plotly_draw_from_the_same_stops(kind):
    name = colormaps.MPL_NAMES[kind]
    scale = colormaps.plotly_colorscale(name)
    assert scale == [[p, c] for p, c in colormaps.stops(kind)]
    assert colormaps.plotly_colorscale(kind) == scale
    cmap = colormaps.get(name)
    # matplotlib reads a position from its 256-entry table, so a stop between
    # two entries is drawn as the nearer one, within one step of the table
    step = np.abs(np.diff(cmap(np.arange(cmap.N))[:, :3], axis=0)).max()
    for position, color in scale:
        np.testing.assert_allclose(cmap(position)[:3], hex_to_rgb(color), atol=step + 0.5 / 255)


def test_every_name_is_registered_with_matplotlib():
    names = colormaps.register()
    assert names == colormaps.NAMES == ("emode-sequential", "emode-diverging", "emode-cyclic")
    for name in names:
        assert mpl.colormaps[name].name == name
    assert colormaps.register() == names  # a second call is harmless


def test_get_gives_the_registered_colormap_by_kind_or_name():
    assert colormaps.get("sequential").name == "emode-sequential"
    assert colormaps.get("emode-cyclic").name == "emode-cyclic"
    assert colormaps.get("viridis").name == "viridis"


def test_the_browser_gets_every_kind_the_names_and_the_lines():
    stops = colormaps.all_stops()
    assert set(stops["kinds"]) == set(colormaps.KINDS)
    assert stops["names"] == dict(colormaps.MPL_NAMES)
    assert stops["lines"] == list(colormaps.line_palette())


# ------------------------------------------------------------------ the drop-in


def test_the_drop_in_is_one_function_and_the_others_derive_from_it(monkeypatch):
    """Replacing ``delivered()`` alone changes every kind, the browser's and matplotlib's."""
    gray = ("#000000", "#808080", "#ffffff")
    monkeypatch.setattr(colormaps, "delivered", lambda: colormaps.Family(sequential=gray))
    assert [c for _, c in colormaps.stops("sequential")] == list(gray)
    assert [c for _, c in colormaps.stops("diverging")] == ["#ffffff", "#808080", "#000000", "#808080", "#ffffff"]
    assert [c for _, c in colormaps.stops("cyclic")] == ["#000000", "#808080", "#ffffff", "#808080", "#000000"]
    assert colormaps.plotly_colorscale("emode-sequential")[1] == [0.5, "#808080"]
    # with no line colors delivered the placeholder palette stays
    assert colormaps.line_palette() == colormaps.Family.__dataclass_fields__["lines"].default


def test_a_delivered_diverging_or_cyclic_map_is_used_as_given(monkeypatch):
    family = colormaps.Family(
        sequential=("#000000", "#ffffff"),
        diverging=("#0000ff", "#ffffff", "#ff0000"),
        cyclic=("#ff0000", "#00ff00", "#0000ff", "#ff0000"),
        lines=("#111111", "#eeeeee"),
    )
    monkeypatch.setattr(colormaps, "delivered", lambda: family)
    assert [c for _, c in colormaps.stops("diverging")] == list(family.diverging)
    assert [c for _, c in colormaps.stops("cyclic")] == list(family.cyclic)
    assert colormaps.line_palette() == family.lines


def test_a_bad_delivery_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(colormaps, "delivered", lambda: colormaps.Family(sequential=("#000000", "red")))
    with pytest.raises(ValueError, match="'red'"):
        colormaps.stops("sequential")


# ------------------------------------------------------------------ colormap=


@pytest.mark.parametrize("name", colormaps.NAMES + colormaps.KINDS)
def test_colormap_takes_every_name_of_the_family(name):
    assert PlotOptions.from_arguments({"colormap": name}).colormap == name


def test_colormap_takes_every_colormap_matplotlib_has_registered():
    for name in list(mpl.colormaps):
        assert PlotOptions.from_arguments({"colormap": name}).colormap == name


def test_an_unknown_colormap_is_refused_naming_the_choices():
    with pytest.raises(ArgumentError) as info:
        PlotOptions.from_arguments({"colormap": "plasmaa"})
    message = str(info.value)
    assert "'plasmaa'" in message
    for name in colormaps.NAMES:
        assert name in message
    assert "matplotlib" in message
    assert info.value.argument == "colormap"


def test_a_colormap_that_is_not_a_string_is_refused():
    with pytest.raises(ArgumentError, match="colormap"):
        PlotOptions.from_arguments({"colormap": 3})
