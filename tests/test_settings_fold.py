"""A phone's control bar folds into Settings (emode#1701 review, Eric 2026-11-15).

On a phone the per-kind controls took about 300 px (eight rows for an EME device) above
the plot. Under the tab's 800 px breakpoint they now sit in a "Settings" section, closed
by default and remembered per browser (``localStorage`` behind try/catch, like Auto
render). The Render button with Auto render, and Export, stay outside it. Wide screens
are unchanged.
"""

from __future__ import annotations

import pytest

from chrome_cdp import Page
from js_harness import needs_jsc, run_app
from test_colorbar_render import CHROME
from test_phone_controls import _cut_server

#: The tab showing a plot with a few per-kind controls, on a narrow or a wide screen.
SHOWN = """
var reply = {id: "p1", kind: "plot", menu: {data: ["Ex", "Ey"], modes: ["0, TE", "1, TM"],
  selected: {data: "Ex", mode: "0, TE"}}, controls: {}, figure: {data: [], layout: {}}};
state.current = "p1";
var box = document.getElementById("controls");
function build() { buildControls(reply); }
function fold() { return box.children.filter(function (c) {
  return c.tagName === "DETAILS" && c.classList.contains("settings"); })[0]; }
function names(el) { return el.querySelectorAll("select").length; }
"""


def _app(prelude: str, test_js: str) -> object:
    return run_app(SHOWN + prelude + test_js)


@needs_jsc
def test_narrow_the_per_kind_controls_sit_in_a_closed_settings_fold():
    got = _app(
        "narrowQuery.matches = true; build();",
        """
var f = fold();
var outside = box.children.filter(function (c) { return c !== f; });
return {fold: !!f, open: f && f.open, inside: f ? names(f) : -1,
  selectsOutside: outside.reduce(function (n, c) { return n + names(c) + (c.tagName === "LABEL" ? 1 : 0); }, 0),
  text: f ? __text(f.children[0]) : "",
  auto: __text(box).indexOf("Auto render") >= 0 && __text(f).indexOf("Auto render") < 0,
  render: __text(f).indexOf("Render") < 0 && __text(box).indexOf("Render") >= 0,
  exportOutside: __text(f).indexOf("Export") < 0 && __text(box).indexOf("Export") >= 0};
""",
    )
    assert got["fold"] and got["open"] is False
    assert got["text"] == "Settings"
    assert got["inside"] >= 3  # Data, Mode, Aspect
    assert got["selectsOutside"] == 0
    assert got["auto"] and got["render"] and got["exportOutside"]


@needs_jsc
def test_wide_there_is_no_settings_fold():
    got = _app("narrowQuery.matches = false; build();", "return {fold: !!fold(), selects: box.querySelectorAll('select').length};")
    assert got["fold"] is False
    assert got["selects"] >= 3


@needs_jsc
def test_opening_the_fold_is_remembered_per_browser():
    got = _app(
        """
var stored = {};
localStorage = { getItem: function (k) { return k in stored ? stored[k] : null; },
                 setItem: function (k, v) { stored[k] = String(v); } };
narrowQuery.matches = true; build();
""",
        """
var f = fold(); f.open = true; f.dispatch("toggle");
return stored;
""",
    )
    assert got == {"emode.settingsOpen": "open"}


@needs_jsc
def test_a_remembered_open_fold_starts_open():
    got = _app(
        """
localStorage = { getItem: function (k) { return k === "emode.settingsOpen" ? "open" : null; }, setItem: function () {} };
narrowQuery.matches = true; build();
""",
        "return fold().open;",
    )
    assert got is True


@needs_jsc
def test_blocked_storage_starts_closed_and_does_not_throw():
    got = _app(
        """
localStorage = { getItem: function () { throw new Error("blocked"); },
                 setItem: function () { throw new Error("blocked"); } };
narrowQuery.matches = true; build();
""",
        'var f = fold(); f.open = true; f.dispatch("toggle"); return f.open;',
    )
    assert got is True


# ------------------------------------------------------------------ in a real browser

PROBE = """(function () {
  var c = document.getElementById("controls");
  var f = document.getElementById("figure");
  var d = c.querySelector("details.settings");
  var fig = f.getBoundingClientRect(), bar = c.getBoundingClientRect();
  var inside = d ? d.querySelectorAll("select, input[type=text]").length : 0;
  var all = c.querySelectorAll("select, input[type=text]").length;
  var shown = d ? Array.prototype.every.call(d.querySelectorAll("select"),
    function (s) { return s.getBoundingClientRect().height > 0; }) : null;
  return JSON.stringify({fold: !!d, open: d ? d.open : null, inside: inside, all: all,
    shown: shown, gap: fig.top - bar.top, barTop: bar.top});
})()"""


def _page(size):
    server, _ = _cut_server()
    page = Page(
        CHROME, f"http://127.0.0.1:{server.port}/?token={server.token}#/plots/p1", *size
    )
    page.wait_for("document.querySelectorAll('#controls select').length > 1")
    return server, page


@pytest.mark.skipif(CHROME is None, reason="needs a headless Chrome")
def test_a_phone_shows_the_plot_near_the_top_and_settings_holds_every_control():
    import json

    server, page = _page((390, 844))
    try:
        closed = json.loads(page.eval(PROBE))
        page.eval("document.querySelector('#controls details.settings').open = true")
        opened = json.loads(page.eval(PROBE))
    finally:
        page.close()
        server.stop()
    assert closed["fold"] and closed["open"] is False
    assert closed["gap"] <= 150, closed
    assert opened["open"] is True and opened["shown"] is True
    assert opened["inside"] == opened["all"] and opened["all"] >= 3


@pytest.mark.skipif(CHROME is None, reason="needs a headless Chrome")
def test_a_desktop_has_no_settings_fold():
    import json

    server, page = _page((1280, 900))
    try:
        got = json.loads(page.eval(PROBE))
    finally:
        page.close()
        server.stop()
    assert got["fold"] is False and got["all"] >= 3
