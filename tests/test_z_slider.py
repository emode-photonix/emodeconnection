"""The EME plot's z position is a real slider, with the text box beside it (emodeconnection#161).

On the x-y plane the slider is the z of the cut, on a z panel ``MenuState.position``
is the z of the x-y cut the panel marks. Each draws a range input from the menu's
``min`` to its ``max`` at its value, next to the text box for an exact value.
Dragging moves the text box only, and letting go posts the position once.
"""

from __future__ import annotations

import re
from pathlib import Path

from js_harness import needs_jsc, run_app

APP_JS = Path(__file__).resolve().parents[1] / "emode" / "ui" / "static" / "app.js"

XY = """
var box = document.createElement("div");
var changes = [];
change = function (c) { changes.push(c); };
buildPlotControls(box, {menu: {slider: {label: "z:", min: 0, max: 30, unit: "μm",
  value: "12.5 μm", options: ["0 μm", "Entrance", "Exit"]}}, controls: {}});
var range = box.querySelector("input[type=range]");
var text = box.querySelector("input[type=text]");
"""

ZX = """
var box = document.createElement("div");
var changes = [];
change = function (c) { changes.push(c); };
buildPlotControls(box, {menu: {position: {label: "z:", min: 0, max: 30, unit: "μm",
  value: "7.5 μm", options: []}}, controls: {}});
var range = box.querySelector("input[type=range]");
var text = box.querySelector("input[type=text]");
"""


@needs_jsc
def test_the_xy_z_slider_is_a_range_input_with_the_menus_limits_and_value():
    got = run_app(XY + "return range && {min: range.min, max: range.max, value: range.value, text: text.value};")
    assert got is not None, "no range input for the z slider"
    assert float(got["min"]) == 0.0
    assert float(got["max"]) == 30.0
    assert float(got["value"]) == 12.5
    assert got["text"] == "12.5 μm"


@needs_jsc
def test_dragging_moves_the_text_box_and_letting_go_posts_the_z_once():
    got = run_app(
        XY
        + """
range.value = "20";
range.dispatch("input", {target: range});
var during = changes.length, shown = text.value;
range.dispatch("change", {target: range});
return {during: during, shown: shown, changes: changes};
"""
    )
    assert got["during"] == 0, "a drag posted before it ended"
    assert got["shown"].startswith("20")
    assert got["changes"] == [{"slice_position": 20, "z_position": 20}]


@needs_jsc
def test_the_text_box_still_takes_an_exact_value_or_a_named_entry():
    got = run_app(
        XY
        + """
text.value = "Exit";
text.dispatch("change", {target: text});
text.value = "3.25";
text.dispatch("change", {target: text});
return changes;
"""
    )
    assert got == [{"slice_position": "Exit"}, {"slice_position": "3.25", "z_position": 3.25}]


@needs_jsc
def test_the_z_panels_cut_position_is_a_range_input_that_posts_z_position():
    got = run_app(
        ZX
        + """
var before = {min: range.min, max: range.max, value: range.value};
range.value = "29";
range.dispatch("change", {target: range});
return {before: before, changes: changes};
"""
    )
    assert float(got["before"]["min"]) == 0.0
    assert float(got["before"]["max"]) == 30.0
    assert float(got["before"]["value"]) == 7.5
    assert got["changes"] == [{"z_position": 29}]


@needs_jsc
def test_a_transverse_slider_in_nm_is_a_range_too():
    got = run_app(
        """
var box = document.createElement("div");
var changes = [];
change = function (c) { changes.push(c); };
buildPlotControls(box, {menu: {slider: {label: "y:", min: -1500, max: 1500, unit: "nm",
  value: "sum", options: ["sum", "mean"]}}, controls: {}});
var range = box.querySelector("input[type=range]");
range.value = "100";
range.dispatch("change", {target: range});
return {min: range.min, max: range.max, changes: changes};
"""
    )
    assert float(got["min"]) == -1500.0 and float(got["max"]) == 1500.0
    assert got["changes"] == [{"slice_position": 100}]


def test_the_script_draws_a_range_input_and_posts_on_change_only():
    js = APP_JS.read_text()
    start = js.find("function zSlider(")
    assert start >= 0, "no zSlider() in app.js"
    body = js[start : js.index("\n}\n", start)]
    assert re.search(r'type\s*=\s*"range"', body)
    assert '"change"' in body
