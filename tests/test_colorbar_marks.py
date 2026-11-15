"""The contour marks across the colorbar sit at their levels, and mark only the lines drawn (#178).

The plot draws a short line across the colorbar at each contour level (#170).
Plotly's colorbar has no such lines, so they are layout shapes placed in paper
fractions. Plotly draws the bar itself ``xpad`` pixels in from its anchor and
``ypad`` pixels short of each end of its length, and a colorbar anchored on its
right is shifted by the width of its whole box, tick labels included, which
only the browser knows. `_bar` below is Plotly 3.1's own geometry
(``components/colorbar/draw.js``: the axis domain ``[vFrac + ypad / h,
vFrac + lenFrac - ypad / h]`` and the fill's left edge ``x * w + xpad``), so a
mark is where the bar shows its value only if the figure sets what the marks
assume.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from emode.plotting import Raster
from emode.ui.figure import FigureSpec, PlotOptions, compose
from emode.ui.renderers.plotly import to_plotly

from plot_fixtures import xy_view

#: The levels EMode contours a field at (``plots/utils.py`` lpos and lneg).
POSITIVE = [0.001, 0.01, *np.round(np.linspace(0.1, 0.9, 9), 6), 0.99]
SIGNED = sorted(POSITIVE + [-v for v in POSITIVE])

#: The plot area the geometry is checked on, in pixels.
W, H = 600.0, 400.0

#: Plotly's colorbar defaults (colorbar/attributes.js and defaults.js).
DEFAULTS = {"xpad": 10, "ypad": 10, "len": 1.0, "y": 0.5, "x": 1.02, "xanchor": "left", "yanchor": "middle"}


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
    raster = Raster(x=x, y=y, z=z, kind="diverging" if signed else "sequential", contour_levels=levels)
    scale = base.scale.model_copy(update={"kind": "signed" if signed else "positive", "vmin": -1.0 if signed else 0.0})
    return base.model_copy(update={"rasters": [raster], "scale": scale})


def _bar(colorbar: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Where Plotly draws the colored bar: (its x span in pixels from the paper's left, its y domain)."""
    cb = {**DEFAULTS, **colorbar}
    thick = float(cb.get("thickness", 30))
    length = float(cb["len"])
    v = float(cb["y"]) - {"top": 1.0, "middle": 0.5}.get(cb["yanchor"], 0.0) * length
    domain = (v + cb["ypad"] / H, v + length - cb["ypad"] / H)
    assert cb["xanchor"] == "left", (
        "a colorbar anchored on its right moves with its tick labels' width, so no mark can follow it"
    )
    left = float(cb["x"]) * W + cb["xpad"]
    return (left, left + thick), domain


def _marks(figure: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in figure["layout"].get("shapes", []) if s.get("name") == "colorbar_level"]


def _check_marks(figure: Dict[str, Any], vmin: float, vmax: float, levels: List[float]) -> None:
    bars = [t["colorbar"] for t in figure["data"] if "colorbar" in t]
    marks = _marks(figure)
    assert len(marks) == len(bars) * len(levels)
    for k, bar in enumerate(bars):
        (x0, x1), (d0, d1) = _bar(bar)
        mine = marks[k * len(levels) : (k + 1) * len(levels)]
        for level, mark in zip(levels, mine):
            want = d0 + (level - vmin) / (vmax - vmin) * (d1 - d0)
            assert abs(mark["y0"] - want) * H < 0.5, f"the mark of {level} is {(mark['y0'] - want) * H:.1f} px off"
            got = (mark["xanchor"] * W + mark["x0"], mark["xanchor"] * W + mark["x1"])
            assert abs(got[0] - x0) < 0.5 and abs(got[1] - x1) < 0.5, f"the mark spans {got} px, the bar {(x0, x1)} px"


def test_each_mark_lies_where_the_bar_shows_its_level():
    figure = to_plotly(compose(_view()))
    _check_marks(figure, -1.0, 1.0, SIGNED)


def test_each_panels_marks_lie_on_its_own_bar():
    one = compose(_view())
    two = FigureSpec(axes=one.axes, panels=(one, one), panel_grid=(2, 1))
    _check_marks(to_plotly(two), -1.0, 1.0, SIGNED)


def test_the_bar_marks_only_the_levels_the_field_reaches():
    # an EME x-y cut normalized to the launched field peaks well below 1: the
    # levels above its peak have no line on the field, so none on the bar
    view = _view(peak=0.143, signed=False)
    spec = compose(view)
    assert spec.colorbar is not None
    reached = [v for v in POSITIVE if v < 0.143]
    assert len(spec.colorbar.lines) == len(reached) and np.allclose(spec.colorbar.lines, reached)
    _check_marks(to_plotly(spec), 0.0, 1.0, reached)


def test_the_colorbar_stays_off_when_contours_are_off():
    spec = compose(_view(), PlotOptions(contours=False))
    assert spec.colorbar is not None and spec.colorbar.lines == ()
    assert not _marks(to_plotly(spec))
