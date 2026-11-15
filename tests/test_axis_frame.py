"""Axis lines and tick marks (#187, #190).

Eric, 2026-11-15: a trend (a line plot) with one y axis draws no right axis
line, and no plot draws a top axis line. Grid lines stay. Tick marks go outside
the axes, small but visible: 4 px long and 1 px wide, on every plot. A field
plot keeps its right edge, a line without ticks, since its image fills the box.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from emode.figures import Figure
from emode.ui.figure import compose
from emode.ui.renderers.plotly import to_plotly
from emode.ui.sweep_figure import sweep_panel

from plot_fixtures import smatrix_view, xy_view
from test_sweep_plots import length_sweep


def _trend() -> Figure:
    fig = Figure()
    fig.panel(0).line([1.0, 2.0, 3.0], [0.2, 0.9, 0.4], style="o-", label="T")
    return fig


def _twin() -> Figure:
    fig = Figure()
    p = fig.panel(0)
    p.line([1.0, 2.0, 3.0], [0.2, 0.9, 0.4])
    p.twin().line([1.0, 2.0, 3.0], [5.0, 6.0, 7.0])
    return fig


TRENDS = {"figure": _trend, "sweep": lambda: sweep_panel(length_sweep())}
EVERY = {**TRENDS, "twin": _twin, "field": xy_view, "smatrix": smatrix_view}


def _layout(make: Any) -> Dict[str, Any]:
    return to_plotly(compose(make()))["layout"]


@pytest.mark.parametrize("name", sorted(TRENDS))
def test_a_trend_with_one_y_axis_draws_neither_a_top_nor_a_right_line(name):
    layout = _layout(TRENDS[name])
    assert layout["xaxis"]["mirror"] is False
    assert layout["yaxis"]["mirror"] is False
    assert layout["xaxis"]["showline"] and layout["yaxis"]["showline"]
    # the grid stays
    assert layout["xaxis"]["showgrid"] and layout["yaxis"]["showgrid"]


def test_a_twin_axis_trend_keeps_its_right_axis_and_no_top_line():
    layout = _layout(_twin)
    assert layout["xaxis"]["mirror"] is False
    assert layout["yaxis"]["mirror"] is False
    right = [
        v
        for k, v in layout.items()
        if k.startswith("yaxis") and v.get("side") == "right"
    ]
    assert right and right[0]["showline"]


@pytest.mark.parametrize("name", ["field", "smatrix"])
def test_an_image_plot_keeps_its_right_edge_without_ticks_and_no_top_line(name):
    layout = _layout(EVERY[name])
    assert layout["xaxis"]["mirror"] is False
    assert layout["yaxis"]["mirror"] is True


@pytest.mark.parametrize("name", sorted(EVERY))
def test_every_tick_mark_is_outside_4_px_long_and_1_px_wide(name):
    layout = _layout(EVERY[name])
    axes = [v for k, v in layout.items() if k[1:5] == "axis" and v.get("visible", True)]
    assert axes
    for axis in axes:
        assert axis["ticks"] == "outside"
        assert axis["ticklen"] == 4 and axis["tickwidth"] == 1


def test_the_matplotlib_export_draws_the_same_frame():
    pytest.importorskip("matplotlib")
    from emode.ui.renderers import mpl as mpl_renderer

    fig = mpl_renderer.to_figure(compose(_trend()))
    ax = fig.axes[0]
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    assert ax.xaxis.get_tick_params()["direction"] == "out"
    field = mpl_renderer.to_figure(compose(xy_view())).axes[0]
    assert not field.spines["top"].get_visible() and field.spines["right"].get_visible()
    assert np.isfinite(ax.get_ylim()).all()
