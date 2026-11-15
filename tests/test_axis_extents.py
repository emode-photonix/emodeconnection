"""Axis extent controls, and non-negative quantities ranged from 0 (#191).

Eric, 2026-11-15: a min and a max for each axis, blank for automatic. A
quantity that cannot be negative (|S|, |S|², a power or a transmission, an
intensity) starts its automatic range at 0 and never auto-ranges below it.
Which quantities count is decided from the data's kind, the component a sweep
draws (``abs``, ``abs^2``) and whether its result is a power, never from the
label's text. The extents are the figure options ``xmin``, ``xmax``, ``ymin``
and ``ymax``.
"""

from __future__ import annotations

import numpy as np
import pytest

from emode.figures import Figure
from emode.ui.figure import PlotOptions, compose
from emode.types import ArgumentError
from emode.ui.renderers import mpl
from emode.ui.renderers.plotly import to_plotly
from emode.ui.sweep_figure import sweep_panel
from emode.ui.views import _select

from js_harness import needs_jsc, run_app
from plot_fixtures import xy_view
from test_sweep_plots import length_sweep


def test_each_extent_overrides_its_end_and_blank_is_automatic():
    auto = compose(xy_view())
    spec = compose(xy_view(), PlotOptions(xmin=-500.0, ymax=300.0))
    assert spec.axes.xrange == (-500.0, auto.axes.xrange[1])
    assert spec.axes.yrange == (auto.axes.yrange[0], 300.0)


def test_the_extents_apply_to_each_panel_of_a_figure():
    fig = Figure()
    fig.panel(0).line([1.0, 2.0, 3.0], [0.1, 0.5, 0.9])
    spec = compose(fig, PlotOptions(xmin=1.5, ymin=0.2, ymax=0.8))
    (panel,) = spec.panels
    assert panel.axes.xrange[0] == 1.5 and panel.axes.yrange == (0.2, 0.8)


@pytest.mark.parametrize("component", ["abs^2"])
def test_a_magnitude_of_s_starts_its_range_at_zero(component):
    spec = compose(_select(sweep_panel(length_sweep()), {"component": component}))
    assert spec.axes.y_nonnegative is True
    assert spec.axes.yrange[0] == 0.0
    yaxis = to_plotly(spec)["layout"]["yaxis"]
    assert yaxis["autorangeoptions"]["minallowed"] == 0
    assert yaxis["rangemode"] == "nonnegative"


@pytest.mark.parametrize("component", ["real", "imag", "phase", "dB"])
def test_a_signed_part_is_ranged_from_its_data(component):
    spec = compose(_select(sweep_panel(length_sweep()), {"component": component}))
    assert spec.axes.y_nonnegative is False
    assert "autorangeoptions" not in to_plotly(spec)["layout"]["yaxis"]


def test_a_set_minimum_wins_over_the_zero_floor():
    spec = compose(sweep_panel(length_sweep()), PlotOptions(ymin=0.5))
    assert spec.axes.yrange[0] == 0.5


def test_a_figure_panel_over_a_sweep_inherits_the_floor():
    fig = Figure()
    fig.panel(0).sweep(length_sweep())
    (panel,) = compose(fig).panels
    assert panel.axes.y_nonnegative is True and panel.axes.yrange[0] == 0.0


BAR = """
var posted = [];
show = function (id, controls) { posted.push(controls || {}); };
state.current = "p1";
var box = document.getElementById("controls");
buildControls({id: "p1", kind: "plot", menu: {}, controls: {xmin: -500}, figure: {data: [], layout: {}}});
function box_(name) { return box.querySelectorAll("input").filter(function (el) {
  return el.parentNode && __text(el.parentNode).indexOf(name) === 0; })[0]; }
function type(input, v) { input.value = v; input.dispatch("change", {target: input}); }
"""


@needs_jsc
def test_the_bar_has_four_extent_boxes_showing_what_is_set():
    got = run_app(
        BAR
        + 'return ["x min", "x max", "y min", "y max"].map(function (n) { var b = box_(n); return b ? b.value : null; });'
    )
    assert got == ["-500", "", "", ""]


@needs_jsc
def test_an_extent_posts_a_number_and_a_blank_posts_automatic():
    got = run_app(
        BAR + 'type(box_("y min"), "0.2"); type(box_("x min"), ""); return posted;'
    )
    assert got == [{"ymin": 0.2}, {"xmin": None}]


@needs_jsc
def test_with_auto_render_off_the_extents_wait_for_render():
    got = run_app(
        BAR
        + """
setAutoRender(false);
type(box_("y min"), "0.2"); type(box_("y max"), "0.9");
var before = posted.length; renderNow();
return {before: before, posted: posted};
"""
    )
    assert got == {"before": 0, "posted": [{"ymin": 0.2, "ymax": 0.9}]}


def _plain_figure(**kwargs):
    fig = Figure()
    p = fig.panel(0)
    p.line([1.0, 2.0, 3.0], [0.2, 0.5, 0.9])
    p.set(**kwargs)
    return fig


def test_a_script_marks_its_y_axis_non_negative():
    """A plain line carries no kind, so the script says it (#193)."""
    (panel,) = compose(_plain_figure(ynonnegative=True)).panels
    assert panel.axes.y_nonnegative is True and panel.axes.yrange[0] == 0.0
    yaxis = to_plotly(panel)["layout"]["yaxis"]
    assert yaxis["rangemode"] == "nonnegative" and yaxis["autorangeoptions"]["minallowed"] == 0


def test_an_unmarked_plain_line_is_not_floored():
    (panel,) = compose(_plain_figure()).panels
    assert panel.axes.y_nonnegative is False and panel.axes.yrange[0] > 0.0 - 1e-9
    assert "autorangeoptions" not in to_plotly(panel)["layout"]["yaxis"]


def test_the_export_gets_the_same_floor():
    pytest.importorskip("matplotlib")
    fig = mpl.to_figure(compose(_plain_figure(ynonnegative=True)))
    (ax,) = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    assert ax.get_ylim()[0] == 0.0


def test_the_twin_axis_can_be_marked_too():
    fig = Figure()
    p = fig.panel(0)
    p.line([1.0, 2.0], [-1.0, 1.0])
    p.twin().line([1.0, 2.0], [0.3, 0.8]).set(ynonnegative=True)
    spec = compose(fig)
    (panel,) = spec.panels
    (ax, twin) = [a for a in mpl.to_figure(spec).axes if a.get_label() != "<colorbar>"][:2]
    assert twin.get_ylim()[0] == 0.0 and ax.get_ylim()[0] < 0.0
    assert panel.axes.y2_nonnegative is True and panel.axes.y2range[0] == 0.0
    assert panel.axes.y_nonnegative is False
    layout = to_plotly(panel)["layout"]
    assert layout["yaxis2"]["rangemode"] == "nonnegative"
    assert "rangemode" not in layout["yaxis"]


@pytest.mark.parametrize("value", ["yes", 2, [True]])
def test_an_unknown_value_is_an_argument_error(value):
    with pytest.raises(ArgumentError):
        Figure().panel(0).set(ynonnegative=value)
    with pytest.raises(ArgumentError):
        Figure().panel(0).twin().set(ynonnegative=value)
