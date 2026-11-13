"""Each plot's default aspect ratio, and the tab sizing the plot to its width (emodeconnection#160).

Eric's defaults: the S-matrix 1:1, an x-y cross section at the window's true
ratio, every other plot at the golden ratio, wider than tall. ``aspect_ratio``
still wins. The plot box's height over its width is ``AxesSpec.box_aspect``,
which both renderers use, and the tab takes the plot's height from its width and
that ratio rather than filling a tall phone frame.
"""

from __future__ import annotations

import json
import math

import pytest

from emode.ui.figure import GOLDEN, PlotOptions, compose, figure_aspect
from emode.ui.renderers import mpl
from emode.ui.server import UIServer
from emode.ui.views import ViewRegistry

from js_harness import needs_jsc, run_app
from plot_fixtures import layout_view, material_view, schematic, smatrix_view, xy_view, zx_view
from test_ui_server import request


def test_golden_is_the_golden_ratio():
    assert GOLDEN == pytest.approx((1 + math.sqrt(5)) / 2)


def test_the_smatrix_is_square():
    assert compose(smatrix_view()).axes.box_aspect == pytest.approx(1.0)


def test_an_xy_cross_section_has_the_windows_true_ratio():
    view = xy_view()
    (x0, x1), (y0, y1) = view.axes.xlim, view.axes.ylim
    assert compose(view).axes.box_aspect == pytest.approx((y1 - y0) / (x1 - x0))


@pytest.mark.parametrize("make", [zx_view, layout_view, material_view, schematic], ids=lambda f: f.__name__)
def test_every_other_plot_is_golden_and_wider_than_tall(make):
    assert compose(make()).axes.box_aspect == pytest.approx(1 / GOLDEN)


@pytest.mark.parametrize("make", [zx_view, xy_view, smatrix_view, material_view], ids=lambda f: f.__name__)
def test_an_explicit_aspect_ratio_wins(make):
    assert compose(make(), PlotOptions(aspect_ratio=0.25)).axes.box_aspect == pytest.approx(0.25)


def test_real_is_the_limits_own_ratio():
    view = zx_view()
    spec = compose(view, PlotOptions(aspect_ratio="real"))
    (x0, x1), (y0, y1) = spec.axes.xrange, spec.axes.yrange
    assert spec.axes.box_aspect == pytest.approx(abs(y1 - y0) / abs(x1 - x0))


@pytest.mark.parametrize(
    "make, ratio",
    [(smatrix_view, 1.0), (zx_view, 1 / GOLDEN), (layout_view, 1 / GOLDEN), (schematic, 1 / GOLDEN)],
    ids=["smatrix", "zx", "layout", "schematic"],
)
def test_matplotlib_draws_the_same_box(make, ratio):
    fig = mpl.to_figure(compose(make()))
    ax = fig.axes[0]
    fig.canvas.draw()
    box = ax.get_position()
    w, h = fig.get_size_inches()
    assert (box.height * h) / (box.width * w) == pytest.approx(ratio, rel=0.02)


def test_the_xy_box_in_matplotlib_is_the_windows_ratio():
    view = xy_view()
    (x0, x1), (y0, y1) = view.axes.xlim, view.axes.ylim
    fig = mpl.to_figure(compose(view))
    fig.canvas.draw()
    box = fig.axes[0].get_position()
    w, h = fig.get_size_inches()
    assert (box.height * h) / (box.width * w) == pytest.approx((y1 - y0) / (x1 - x0), rel=0.02)


def test_figure_aspect_reads_the_box():
    assert figure_aspect(compose(smatrix_view())) == pytest.approx(1.0)
    assert figure_aspect(compose(zx_view())) == pytest.approx(1 / GOLDEN)


def test_the_tab_reply_carries_the_aspect():
    registry = ViewRegistry()
    view = registry.add(zx_view(), {"plane": "z-x"}, PlotOptions(), (1200, 750))
    server = UIServer(registry, lambda v, a, p: zx_view()).start()
    try:
        _, _, body = request(server, "POST", f"/api/views/{view.id}/data", body={"controls": {}})
        assert json.loads(body)["aspect"] == pytest.approx(1 / GOLDEN)
    finally:
        server.stop()


@needs_jsc
def test_the_tab_sets_the_plots_height_from_its_width():
    got = run_app(
        """
var fig = document.getElementById("figure");
fig.clientWidth = 400;
var layouts = [];
Plotly.react = function (el, data, layout) { layouts.push(layout); };
buildControls = function () {};
draw({id: "p1", aspect: 0.5, figure: {data: [], layout: {margin: {l: 60, r: 20, t: 40, b: 50}}}});
return {height: layouts[0].height, style: fig.style.height};
"""
    )
    # the plot box is 400 - 60 - 20 = 320 wide, so 160 tall, plus 40 + 50 of margins
    assert got["height"] == 250
    assert got["style"] == "250px"
