"""The tab's aspect ratio control (#189).

Eric, 2026-11-15: a dropdown of sqrt(2) (every plot's default but the
S-matrix's, which is 1:1), golden (1.618), silver (1 + sqrt(2) = 2.414), 1:1
and Custom, a width:height number from 0.001 to 1000. The bounds only keep a
figure from degenerating, so a value at either end draws, with the figure's
pixels clamped to sane limits, and a value past them is refused with a message
and posts nothing. The choice posts the existing ``aspect_ratio`` option, which
is the plot box's height over its width.
"""

from __future__ import annotations

import math

import pytest

from emode.figures import Figure
from emode.types import ArgumentError
from emode.ui.figure import PlotOptions, compose, figure_aspect
from emode.ui.renderers.plotly import to_plotly

from js_harness import needs_jsc, run_app
from plot_fixtures import xy_view

GOLDEN = (1 + math.sqrt(5)) / 2
SILVER = 1 + math.sqrt(2)

BAR = """
var posted = [];
show = function (id, controls) { posted.push(controls || {}); };
state.current = "p1";
var messages = [];
showMessage = function (t) { messages.push(t || ""); };
function bar(kind, controls) {
  var box = document.getElementById("controls");
  buildControls({id: "p1", kind: kind, menu: {}, controls: controls || {}, figure: {data: [], layout: {}}});
  var s = box.querySelectorAll("select").filter(function (el) {
    return el.parentNode && __text(el.parentNode).indexOf("Aspect") === 0; })[0];
  var custom = box.querySelectorAll("input").filter(function (el) {
    return el.parentNode && __text(el.parentNode).indexOf("width:height") >= 0; })[0];
  return {select: s, custom: custom};
}
function pick(s, v) { s.value = v; s.dispatch("change", {target: s}); }
function type(input, v) { input.value = v; input.dispatch("change", {target: input}); }
"""


@needs_jsc
def test_the_menu_and_its_defaults():
    got = run_app(
        BAR
        + """
var plot = bar("plot"), sm = bar("smatrix");
return {options: plot.select.children.map(function (o) { return o.value; }),
        plot: plot.select.value, smatrix: sm.select.value};
"""
    )
    assert got["options"] == ["√2", "golden", "silver", "1:1", "Custom"]
    assert got == {**got, "plot": "√2", "smatrix": "1:1"}


@needs_jsc
@pytest.mark.parametrize(
    "name, ratio",
    [("golden", GOLDEN), ("silver", SILVER), ("1:1", 1.0), ("√2", math.sqrt(2))],
)
def test_a_named_ratio_posts_the_box_height_over_width(name, ratio):
    got = run_app(
        BAR + f'var b = bar("plot"); pick(b.select, "{name}"); return posted;'
    )
    assert len(got) == 1 and got[0]["aspect_ratio"] == pytest.approx(1 / ratio)


@needs_jsc
@pytest.mark.parametrize(
    "text, ratio", [("2", 2.0), ("0.001", 0.001), ("1000", 1000.0)]
)
def test_a_custom_ratio_in_bounds_posts_it(text, ratio):
    got = run_app(
        BAR
        + f'var b = bar("plot"); pick(b.select, "Custom"); type(b.custom, "{text}"); return posted;'
    )
    assert got[-1]["aspect_ratio"] == pytest.approx(1 / ratio)


@needs_jsc
@pytest.mark.parametrize("text", ["0.0009", "1001", "0", "-1", "wide"])
def test_a_custom_ratio_out_of_bounds_is_refused_with_a_message(text):
    got = run_app(
        BAR
        + f'var b = bar("plot"); pick(b.select, "Custom"); type(b.custom, "{text}"); return {{posted: posted, messages: messages}};'
    )
    assert not [p for p in got["posted"] if "aspect_ratio" in p]
    assert any("0.001" in m and "1000" in m for m in got["messages"])


@needs_jsc
def test_the_bar_shows_a_custom_ratio_it_was_given():
    got = run_app(
        BAR
        + 'var b = bar("plot", {aspect_ratio: 0.25}); return {sel: b.select.value, text: b.custom.value};'
    )
    assert got["sel"] == "Custom" and float(got["text"]) == pytest.approx(4.0)


@needs_jsc
@pytest.mark.parametrize("aspect", [1000.0, 0.001])
def test_an_extreme_ratio_draws_a_figure_of_sane_size(aspect):
    got = run_app(
        f"""
var calls = [];
Plotly.react = function (el, data, layout) {{ calls.push(layout); }};
draw({{id: "p1", kind: "plot", aspect: {aspect}, menu: {{}}, controls: {{}},
  figure: {{data: [], layout: {{margin: {{l: 60, r: 20, t: 40, b: 50}}}}}}}});
return calls[0];
"""
    )
    m = got["margin"]
    box_h = got["height"] - m["t"] - m["b"]
    assert 40 <= box_h and got["height"] <= 6000
    assert got["width"] >= 100


@pytest.mark.parametrize("ratio", [0.001, 1000.0])
def test_options_at_the_bounds_compose_and_render(ratio):
    options = PlotOptions.from_arguments({"aspect_ratio": 1 / ratio})
    spec = compose(xy_view(), options)
    assert spec.axes.box_aspect == pytest.approx(1 / ratio)
    to_plotly(spec)


@pytest.mark.parametrize("ratio", [0.0009, 1001.0])
def test_options_past_the_bounds_are_refused(ratio):
    with pytest.raises(ArgumentError, match="aspect_ratio"):
        PlotOptions.from_arguments({"aspect_ratio": 1 / ratio})


def test_a_figure_of_panels_takes_the_ratio_too():
    fig = Figure()
    fig.panel(0).line([1.0, 2.0], [1.0, 2.0])
    assert figure_aspect(
        compose(fig, PlotOptions(aspect_ratio=1 / GOLDEN))
    ) == pytest.approx(1 / GOLDEN)
