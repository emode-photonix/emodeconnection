"""The tab's plot size control: a cap on the plot's width, chosen per browser.

Medium (960 px) is the default. Small is 640 px, Large 1280 px and Fit width is the
stage's whole width. No mode is wider than the stage. The choice is client-side only:
it never posts ``aspect_ratio``, it refits the shown plot and asks for the data again
with the capped width, so the server samples the pixels that are shown.
"""

from __future__ import annotations

import math

import pytest

from js_harness import needs_jsc, run_app

PRELUDE = """
var stageEl = document.getElementById("stage");
var figEl = document.getElementById("figure");
function stage(w) { stageEl.clientWidth = w; figEl.clientWidth = w; }
function reply(extra) {
  return Object.assign({id: "p1", kind: "plot", aspect: Math.SQRT2, menu: {}, controls: {},
    figure: {data: [], layout: {margin: {l: 0, r: 0, t: 0, b: 0}}}}, extra || {});
}
function width() { return figureWidth(figEl); }
function pick(label, v) {
  var s = document.getElementById("controls").querySelectorAll("select").filter(function (el) {
    return el.parentNode && __text(el.parentNode).indexOf(label) === 0; })[0];
  s.value = v; s.dispatch("change", {target: s});
}
"""

MODES = [("Small", 640), ("Medium", 960), ("Large", 1280), ("Fit width", 2000)]


def _run(body: str, setup: str = "") -> object:
    return run_app(setup + PRELUDE + body)


@needs_jsc
def test_the_default_is_medium():
    assert _run("stage(2000); return width();") == 960


@needs_jsc
@pytest.mark.parametrize("name, expected", MODES)
def test_each_mode_caps_a_wide_stage(name, expected):
    got = _run(f'stage(2000); setPlotSize("{name}"); return width();')
    assert got == expected


@needs_jsc
@pytest.mark.parametrize("name, _", MODES)
def test_no_mode_is_wider_than_the_stage(name, _):
    assert _run(f'stage(500); setPlotSize("{name}"); return width();') == 500


@needs_jsc
def test_the_height_follows_the_aspect_from_the_capped_width():
    got = _run(
        "stage(2000); var s = sizeFigure(reply()); return [s.width, s.height];"
    )
    assert got[0] == 960
    assert got[1] == round(960 * math.sqrt(2))


@needs_jsc
def test_the_choice_is_stored_and_restored():
    setup = """
var stored = {};
localStorage = { getItem: function (k) { return k in stored ? stored[k] : null; },
                     setItem: function (k, v) { stored[k] = String(v); } };
"""
    got = _run('stage(2000); setPlotSize("Large"); return stored;', setup)
    assert list(got.values()) == ["Large"]
    key = next(iter(got))
    restored = _run(
        f'stage(2000); return width();',
        setup.replace("var stored = {};", f'var stored = {{"{key}": "Small"}};'),
    )
    assert restored == 640


@needs_jsc
def test_blocked_storage_falls_back_to_medium_and_breaks_nothing():
    setup = """
localStorage = { getItem: function () { throw new Error("blocked"); },
                     setItem: function () { throw new Error("blocked"); } };
"""
    got = _run('stage(2000); var a = width(); setPlotSize("Small"); return [a, width()];', setup)
    assert got == [960, 640]


@needs_jsc
def test_a_stored_nonsense_value_is_medium():
    setup = 'localStorage = { getItem: function () { return "huge"; }, setItem: function () {} };'
    assert _run("stage(2000); return width();", setup) == 960


@needs_jsc
def test_changing_the_size_refits_and_requests_the_data_with_the_capped_width():
    got = _run(
        """
stage(2000);
state.reply = reply(); state.current = "p1";
var relayouts = [];
Plotly.relayout = function (el, layout) { relayouts.push(layout.width); };
__posts.length = 0;
setPlotSize("Small");
var data = __posts.filter(function (p) { return p[0].indexOf("/data") >= 0; })[0];
return {relayouts: relayouts, width: data[1].width, pixels: data[1].pixels,
        aspectPosted: __posts.some(function (p) { return p[1] && "aspect_ratio" in p[1].controls; })};
"""
    )
    assert got["relayouts"] == [640]
    assert got["width"] == 640
    assert got["pixels"][0] == 640
    assert got["aspectPosted"] is False


@needs_jsc
def test_the_control_is_in_the_plot_views_bar():
    got = _run(
        """
stage(2000);
buildControls(reply());
var s = document.getElementById("controls").querySelectorAll("select").filter(function (el) {
  return el.parentNode && __text(el.parentNode).indexOf("Size") === 0; })[0];
return {found: !!s, value: s && s.value,
        options: s && s.children.map(function (o) { return o.value; })};
"""
    )
    assert got == {
        "found": True,
        "value": "Medium",
        "options": ["Small", "Medium", "Large", "Fit width"],
    }
