"""The contour marks across the colorbar show, at their levels, and mark only the lines drawn (#178, #186).

The plot draws a short line across the colorbar at each contour level (#170).
Plotly's colorbar has no such lines. Until #186 they were layout shapes, which
Plotly draws in the main plot's SVG, while it draws every colorbar in the layer
above it (``_toppaper``'s ``infolayer``). The bar's opaque gradient covered the
marks, so they never showed, wherever they were placed. Now each mark is a tick
of a second, clear colorbar laid exactly over the shown one: a trace after the
field's, whose bar has a transparent colorscale and the bar's own geometry, and
whose ticks run inside across the bar's whole thickness, one at each level.
Plotly draws a later trace's colorbar above an earlier one's.

`_bar` below is Plotly 3.1's own geometry (``components/colorbar/draw.js``: the
axis domain ``[vFrac + ypad / h, vFrac + lenFrac - ypad / h]`` and the fill's
left edge ``x * w + xpad``), so the marks lie where the shown bar draws each
level only if the clear bar's geometry is the shown bar's. The browser check is
`test_colorbar_render.py`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from emode.plotting import Raster, ScaleInfo
from emode.ui.figure import FigureSpec, PlotOptions, compose
from emode.ui.renderers.plotly import to_plotly

from plot_fixtures import xy_view

#: The levels EMode contours a field at (``plots/utils.py`` lpos and lneg).
POSITIVE = [0.001, 0.01, *np.round(np.linspace(0.1, 0.9, 9), 6), 0.99]
SIGNED = sorted(POSITIVE + [-v for v in POSITIVE])
#: A log plot's levels, in dB.
DB = [-40.0, -30.0, -20.0, -10.0, -3.0]

#: The plot area the geometry is checked on, in pixels.
W, H = 600.0, 400.0

#: Plotly's colorbar defaults (colorbar/attributes.js and defaults.js).
DEFAULTS = {
    "xpad": 10,
    "ypad": 10,
    "len": 1.0,
    "y": 0.5,
    "x": 1.02,
    "xanchor": "left",
    "yanchor": "middle",
    "thickness": 30,
    "lenmode": "fraction",
    "thicknessmode": "pixels",
}

#: What must match between the shown bar and the clear one laid over it.
GEOMETRY = (
    "x",
    "xanchor",
    "xref",
    "y",
    "yanchor",
    "yref",
    "len",
    "lenmode",
    "thickness",
    "thicknessmode",
    "xpad",
    "ypad",
    "orientation",
)


def _view(peak: float = 1.0, signed: bool = True) -> Any:
    """An x-y field whose largest magnitude is `peak`, contoured at EMode's levels."""
    x = np.linspace(-1000.0, 1000.0, 81)
    y = np.linspace(-500.0, 700.0, 61)
    X, Y = np.meshgrid(x, y)
    z = np.exp(-((X / 300) ** 2) - ((Y - 110) / 150) ** 2)
    if signed:
        z = z * np.sign(X + 1)
    z = (peak * z / np.abs(z).max()).astype(np.float32)
    levels = np.array(SIGNED if signed else POSITIVE)
    base = xy_view()
    raster = Raster(
        x=x,
        y=y,
        z=z,
        kind="diverging" if signed else "sequential",
        contour_levels=levels,
    )
    scale = base.scale.model_copy(
        update={
            "kind": "signed" if signed else "positive",
            "vmin": -1.0 if signed else 0.0,
        }
    )
    return base.model_copy(update={"rasters": [raster], "scale": scale})


def _log_view() -> Any:
    """The same field in dB, from -50 to 0, contoured at `DB`."""
    base = _view(signed=False)
    z = np.asarray(base.rasters[0].z, np.float64)
    db = (10 * np.log10(np.maximum(z, 1e-5))).astype(np.float32)
    raster = base.rasters[0].model_copy(
        update={"z": db, "contour_levels": np.array(DB)}
    )
    return base.model_copy(
        update={
            "rasters": [raster],
            "scale": ScaleInfo(kind="db", vmin=-50.0, vmax=0.0),
        }
    )


def _bar(colorbar: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Where Plotly draws the colored bar: (its x span in pixels from the paper's left, its y domain)."""
    cb = {**DEFAULTS, **colorbar}
    thick = float(cb["thickness"])
    length = float(cb["len"])
    v = float(cb["y"]) - {"top": 1.0, "middle": 0.5}.get(cb["yanchor"], 0.0) * length
    domain = (v + cb["ypad"] / H, v + length - cb["ypad"] / H)
    assert cb["xanchor"] == "left"
    left = float(cb["x"]) * W + cb["xpad"]
    return (left, left + thick), domain


def _shown_bars(
    figure: Dict[str, Any],
) -> List[Tuple[int, Dict[str, Any], Tuple[float, float]]]:
    """Each shown colorbar: (its trace's index, the colorbar, its value range)."""
    out = []
    for i, t in enumerate(figure["data"]):
        if t.get("meta") == "colorbar_levels":
            continue
        if t.get("showscale") and "colorbar" in t:
            out.append((i, t["colorbar"], (t["zmin"], t["zmax"])))
        marker = t.get("marker") or {}
        if marker.get("showscale") and "colorbar" in marker:
            out.append((i, marker["colorbar"], (marker["cmin"], marker["cmax"])))
    return out


def _overlays(figure: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    return [
        (i, t)
        for i, t in enumerate(figure["data"])
        if t.get("meta") == "colorbar_levels"
    ]


def _check_marks(figure: Dict[str, Any], levels: List[float]) -> None:
    bars = _shown_bars(figure)
    overlays = _overlays(figure)
    assert bars and len(overlays) == len(bars), (
        "one clear bar of marks over each shown bar"
    )
    # no layout shape marks a level: a shape is drawn under the colorbar, so never seen
    assert not [
        s
        for s in figure["layout"].get("shapes", [])
        if s.get("name") == "colorbar_level"
    ]
    for (i, bar, (lo, hi)), (j, overlay) in zip(bars, overlays):
        assert j > i, "the clear bar is drawn after, so above, the shown one"
        marker = overlay["marker"]
        clear = marker["colorbar"]
        assert marker["showscale"] is True
        assert (marker["cmin"], marker["cmax"]) == (lo, hi)
        assert all(
            str(c).replace(" ", "").endswith(",0)") for _, c in marker["colorscale"]
        ), "the clear bar is transparent"
        for key in GEOMETRY:
            assert clear.get(key, DEFAULTS.get(key)) == bar.get(
                key, DEFAULTS.get(key)
            ), key
        assert _bar(clear) == _bar(bar)
        assert clear["tickvals"] == pytest.approx(levels)
        assert clear["ticks"] == "inside"
        assert clear["ticklen"] == bar.get("thickness", DEFAULTS["thickness"]), (
            "each mark spans the bar"
        )
        assert clear["showticklabels"] is False
        assert clear["outlinewidth"] == 0
        assert not (clear.get("title") or {}).get("text")
        # nothing of the clear trace is drawn on the axes
        assert (
            overlay["x"] == [None]
            and overlay["y"] == [None]
            and overlay["hoverinfo"] == "skip"
        )
        assert overlay.get("showlegend") is False


def test_each_mark_is_a_tick_of_a_clear_bar_over_the_shown_one():
    _check_marks(to_plotly(compose(_view())), SIGNED)


def test_a_positive_field_is_marked_on_its_bar():
    _check_marks(to_plotly(compose(_view(signed=False))), POSITIVE[:-1])


def test_a_log_field_is_marked_in_db():
    spec = compose(_log_view())
    assert spec.colorbar is not None and spec.colorbar.vmin == -50.0
    _check_marks(to_plotly(spec), DB)


def test_each_panels_marks_lie_on_its_own_bar():
    one = compose(_view())
    two = FigureSpec(axes=one.axes, panels=(one, one), panel_grid=(2, 1))
    figure = to_plotly(two)
    _check_marks(figure, SIGNED)
    (_, first, _), (_, second, _) = _shown_bars(figure)
    assert first["y"] != second["y"], "each panel's bar sits beside its own panel"


def test_the_bar_marks_only_the_levels_the_field_reaches():
    # an EME x-y cut normalized to the launched field peaks well below 1: the
    # levels above its peak have no line on the field, so none on the bar
    view = _view(peak=0.143, signed=False)
    spec = compose(view)
    assert spec.colorbar is not None
    reached = [v for v in POSITIVE if v < 0.143]
    assert len(spec.colorbar.lines) == len(reached) and np.allclose(
        spec.colorbar.lines, reached
    )
    _check_marks(to_plotly(spec), reached)


def test_the_colorbar_stays_unmarked_when_contours_are_off():
    spec = compose(_view(), PlotOptions(contours=False))
    assert spec.colorbar is not None and spec.colorbar.lines == ()
    assert not _overlays(to_plotly(spec))
