"""Auto render and the Render button (#183).

Eric, 2026-11-15: every control change redrew the plot at once. The control
bar now has an "Auto render" checkbox, on by default and remembered per browser
(``localStorage``, behind try/catch, so a private window or blocked storage
just starts with it on), and a "Render" button. With auto render off, control
changes are collected and posted together when Render is pressed. The button is
disabled while nothing waits. A click on the plot itself (a panel of the All
modes grid, a section arrow) renders at once, with whatever waits.
"""

from __future__ import annotations

from js_harness import needs_jsc, run_app

#: The tab showing a field plot, its requests recorded, after `prelude`.
SHOWN = """
var posted = [];
show = function (id, controls) { posted.push(controls || {}); };
state.current = "p1";
var reply = {id: "p1", kind: "plot", menu: {data: ["Ex", "Ey"], modes: ["0, TE", "1, TM"],
  selected: {data: "Ex", mode: "0, TE"}}, controls: {}, figure: {data: [], layout: {}}};
var box = document.getElementById("controls");
buildControls(reply);
function find(tag, text) {
  return box.querySelectorAll(tag).filter(function (el) {
    return __text(el.parentNode || el).indexOf(text) >= 0 || __text(el).indexOf(text) >= 0; })[0];
}
var auto = box.querySelectorAll("input[type=checkbox]").filter(function (c) {
  return __text(c.parentNode).indexOf("Auto render") >= 0; })[0];
var render = box.querySelectorAll("button").filter(function (b) { return __text(b) === "Render"; })[0];
var mode = find("select", "Mode");
var data = find("select", "Data");
function pick(s, v) { s.value = v; s.dispatch("change", {target: s}); }
function untick() { auto.checked = false; auto.dispatch("change", {target: auto}); }
"""


def _app(prelude: str, test_js: str) -> object:
    return run_app(prelude + SHOWN + test_js)


@needs_jsc
def test_the_bar_has_auto_render_on_by_default_and_a_disabled_render_button():
    got = _app(
        "",
        "return {auto: !!auto && auto.checked, render: !!render, disabled: render && render.disabled};",
    )
    assert got == {"auto": True, "render": True, "disabled": True}


@needs_jsc
def test_with_auto_render_on_a_change_renders_at_once():
    got = _app("", 'pick(mode, "1, TM"); return posted;')
    assert got == [{"mode": "1, TM"}]


@needs_jsc
def test_with_auto_render_off_changes_wait_for_render_and_go_together():
    got = _app(
        "",
        """
untick();
pick(mode, "1, TM");
pick(data, "Ey");
var waiting = {posted: posted.length, disabled: render.disabled, marked: render.classList.contains("pending")};
render.click();
return {waiting: waiting, posted: posted, after: render.disabled};
""",
    )
    assert got["waiting"] == {"posted": 0, "disabled": False, "marked": True}
    assert got["posted"] == [{"mode": "1, TM", "component": "Ey"}]
    assert got["after"] is True


@needs_jsc
def test_turning_auto_render_back_on_renders_what_waits():
    got = _app(
        "",
        """
untick();
pick(mode, "1, TM");
auto.checked = true; auto.dispatch("change", {target: auto});
return posted;
""",
    )
    assert got == [{"mode": "1, TM"}]


@needs_jsc
def test_a_click_on_the_plot_renders_at_once_with_what_waits():
    got = _app(
        "",
        """
untick();
pick(data, "Ey");
renderNow({mode: "1, TM"});
return posted;
""",
    )
    assert got == [{"component": "Ey", "mode": "1, TM"}]


@needs_jsc
def test_the_choice_is_remembered_per_browser():
    got = _app(
        """
var stored = {};
localStorage = { getItem: function (k) { return k in stored ? stored[k] : null; },
                 setItem: function (k, v) { stored[k] = String(v); } };
""",
        "untick(); return stored;",
    )
    assert got == {"emode.autoRender": "off"}


@needs_jsc
def test_a_remembered_off_starts_unticked():
    got = _app(
        'localStorage = { getItem: function (k) { return k === "emode.autoRender" ? "off" : null; }, setItem: function () {} };',
        'pick(mode, "1, TM"); return {auto: auto.checked, posted: posted.length};',
    )
    assert got == {"auto": False, "posted": 0}


@needs_jsc
def test_blocked_storage_starts_with_auto_render_on():
    got = _app(
        """
localStorage = { getItem: function () { throw new Error("blocked"); },
                 setItem: function () { throw new Error("blocked"); } };
""",
        'var on = auto.checked; untick(); pick(mode, "1, TM"); render.click(); return {on: on, posted: posted};',
    )
    assert got == {"on": True, "posted": [{"mode": "1, TM"}]}


@needs_jsc
def test_changes_waiting_on_one_plot_are_dropped_when_another_opens():
    got = _app(
        "",
        """
untick();
pick(mode, "1, TM");
forgetPending("p2");
return {disabled: render.disabled, waiting: pendingControls()};
""",
    )
    assert got == {"disabled": True, "waiting": {}}
