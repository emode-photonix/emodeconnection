"""A click on a z panel's field moves the x-y cut to the clicked z (#227).

The "Cut z" slider is the only control that moved the cut. A click on the field
of a z-x or z-y panel that offers a cut (a ``position`` control) now does the
same, with the slider's own change. Clicks that already mean something (a
schematic component, a panel of the All modes grid) keep their meaning, the x-y
panel below the z panel is unchanged, and a swipe sends nothing.
"""

from __future__ import annotations

from js_harness import needs_jsc, run_app

#: Draw a reply, with the figure's handlers kept, and click its first point.
PRELUDE = """
var handlers = {};
var fig = document.getElementById("figure");
fig.on = function (name, f) { handlers[name] = f; };
var changes = [], renders = [], opened = [];
change = function (c) { changes.push(c); };
renderNow = function (c) { renders.push(c); };
openDetails = function (c) { opened.push(c); };
function reply(plane, title, extra) {
  return {id: "p1", kind: "plot", controls: {},
    menu: {position: {label: "z:", min: 0, max: 30, unit: "μm", value: "7.5 μm", options: []},
      selected: {plane: plane}},
    figure: {data: [{type: "scatter", meta: "slice", x: [7.5, 7.5], y: [-1, 1]}],
      layout: {xaxis: {title: {text: title}}}}};
}
function click(r, point, ev) {
  draw(r);
  handlers.plotly_click(Object.assign({points: [point]}, ev || {}));
}
var field = function (x, axis) { return {x: x, y: 0.1, data: {type: "heatmap", xaxis: axis}}; };
"""


def _run(body: str) -> object:
    return run_app(PRELUDE + body)


@needs_jsc
def test_a_click_on_a_z_x_panel_moves_the_cut_as_the_slider_does():
    got = _run('click(reply("z-x", "z (μm)"), field(12.25, "x")); return changes;')
    assert got == [{"z_position": 12.25}]


@needs_jsc
def test_a_click_on_a_z_y_panel_moves_the_cut_as_the_slider_does():
    got = _run('click(reply("z-y", "z (μm)"), field(3, "x")); return changes;')
    assert got == [{"z_position": 3}]


@needs_jsc
def test_a_z_axis_in_nm_is_converted_to_the_cuts_unit():
    got = _run('click(reply("z-x", "z (nm)"), field(12250, "x")); return changes;')
    assert got == [{"z_position": 12.25}]


@needs_jsc
def test_a_custom_axis_label_still_converts_by_the_marker():
    got = _run(
        'var r = reply("z-x", "length"); r.figure.data[0].x = [7500, 7500];'
        ' click(r, field(12250, "x")); return changes;'
    )
    assert got == [{"z_position": 12.25}]


@needs_jsc
def test_a_click_outside_the_cuts_range_goes_to_its_end():
    got = _run('click(reply("z-x", "z (μm)"), field(45, "x")); return changes;')
    assert got == [{"z_position": 30}]


@needs_jsc
def test_the_click_uses_the_current_reply_not_an_earlier_one():
    got = _run(
        """
draw(reply("z-x", "z (μm)"));
var later = reply("z-x", "z (nm)");
draw(later);
handlers.plotly_click({points: [field(5000, "x")]});
return changes;
"""
    )
    assert got == [{"z_position": 5}]


@needs_jsc
def test_a_click_on_the_x_y_panel_below_does_not_move_the_cut():
    got = _run('click(reply("z-x", "z (μm)"), field(0.2, "x2")); return {c: changes, r: renders};')
    assert got == {"c": [], "r": []}


@needs_jsc
def test_a_click_on_the_x_y_plane_does_not_move_the_cut():
    got = _run('click(reply("x-y", "x (μm)"), field(0.2)); return changes;')
    assert got == []


@needs_jsc
def test_a_z_panel_without_a_cut_control_does_nothing():
    got = _run('var r = reply("z-x", "z (μm)"); delete r.menu.position; click(r, field(2, "x")); return changes;')
    assert got == []


@needs_jsc
def test_a_component_still_opens_its_details_and_moves_nothing():
    got = _run(
        'click(reply("z-x", "z (μm)"), {x: 2, data: {meta: "component:section:a", xaxis: "x"}});'
        " return {c: changes, o: opened};"
    )
    assert got == {"c": [], "o": ["section:a"]}


@needs_jsc
def test_a_mode_panel_still_switches_mode_and_moves_nothing():
    got = _run(
        'click(reply("z-x", "z (μm)"), {x: 2, data: {meta: "mode:1, TM", xaxis: "x"}});'
        " return {c: changes, r: renders};"
    )
    assert got == {"c": [], "r": [{"mode": "1, TM"}]}


@needs_jsc
def test_a_swipe_sends_nothing():
    got = _run('swiped = true; click(reply("z-x", "z (μm)"), field(12, "x")); return {c: changes, r: renders};')
    assert got == {"c": [], "r": []}
