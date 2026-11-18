"""A window region that carries a color is drawn filled at about half opacity.

With the window display on it is filled in that color, with it off it is not
drawn. A region without a color stays an outline.
"""
from __future__ import annotations

import pytest

from emode.plotting import Region
from emode.ui.figure import PlotOptions, compose
from emode.ui.renderers.plotly import to_plotly

from plot_fixtures import xy_view, zx_view

pytest.importorskip("matplotlib")
from matplotlib.colors import to_rgba  # noqa: E402

from emode.ui.renderers import mpl as mpl_renderer  # noqa: E402


def _view():
    view = zx_view()
    frame = Region(
        kind="frame", x=[-800, 800, 800, -800], y=[-300, -300, 500, 500], color="#808080"
    )
    return view.model_copy(update={"window": [*view.window, frame]})


def _frame(spec):
    (fill,) = [f for f in spec.fills if f.kind == "frame"]
    return fill


def test_a_colored_window_region_is_a_half_opaque_fill_with_the_window_display_on():
    fill = _frame(compose(_view(), PlotOptions(window_display=True)))
    assert fill.color == "#808080"
    assert fill.alpha == pytest.approx(0.5)


def test_it_is_not_drawn_with_the_window_display_off():
    assert not [f for f in compose(_view(), PlotOptions()).fills if f.kind == "frame"]


def test_a_region_without_a_color_stays_an_outline():
    spec = compose(xy_view(), PlotOptions(window_display=True))
    (window,) = [f for f in spec.fills if f.kind == "window"]
    assert window.color is None and window.alpha == 1.0


def test_matplotlib_fills_the_patch_at_that_opacity():
    ax = mpl_renderer.to_figure(compose(_view(), PlotOptions(window_display=True))).axes[0]
    (patch,) = [p for p in ax.patches if p.get_gid() == "frame"]
    assert tuple(patch.get_facecolor()) == pytest.approx(to_rgba("#808080", 0.5))
    ax = mpl_renderer.to_figure(compose(xy_view(), PlotOptions(window_display=True))).axes[0]
    (plain,) = [p for p in ax.patches if p.get_gid() == "window"]
    assert plain.get_facecolor()[3] == 0.0


def test_plotly_fills_the_trace_at_that_opacity():
    figure = to_plotly(compose(_view(), PlotOptions(window_display=True)))
    (trace,) = [t for t in figure["data"] if t.get("name") == "frame"]
    assert trace["fillcolor"] == "rgba(128,128,128,0.5)"
    figure = to_plotly(compose(xy_view(), PlotOptions(window_display=True)))
    (plain,) = [t for t in figure["data"] if t.get("name") == "window"]
    assert plain["fillcolor"] == "rgba(0,0,0,0)"


def test_with_the_display_off_neither_renderer_draws_it():
    spec = compose(_view(), PlotOptions())
    assert not [p for p in mpl_renderer.to_figure(spec).axes[0].patches if p.get_gid() == "frame"]
    assert not [t for t in to_plotly(spec)["data"] if t.get("name") == "frame"]
