"""Run the tab's script against a small stand-in page, with macOS's own JavaScript engine.

No browser runs in the tests, and node is not a test dependency, so a test that
needs app.js to *do* something (build a control, post a value) evaluates the
script here: osascript's JavaScriptCore, a stand-in ``document`` with just the
calls app.js makes, and stubs for ``fetch``, ``Plotly`` and the timers. The test
then runs its own lines in the same scope and reads back a JSON value.

Where osascript is missing (Linux, Windows) these tests skip. The static checks
on app.js's text cover those platforms.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "emode" / "ui" / "static" / "app.js"

needs_jsc = pytest.mark.skipif(shutil.which("osascript") is None, reason="needs macOS's osascript")

#: The stand-in page: elements that keep their children, attributes, listeners and style.
DOM = r"""
var __posts = [];
var __timers = [];
function __el(tag) {
  var el = {
    tagName: String(tag).toUpperCase(), children: [], attributes: {}, listeners: {},
    style: {}, dataset: {}, hidden: false, textContent: "", value: "", className: "",
    id: "", clientWidth: 400, clientHeight: 600, parentNode: null,
    classList: {
      _s: {}, add: function (c) { this._s[c] = true; }, remove: function (c) { delete this._s[c]; },
      toggle: function (c, on) { if (on === undefined) on = !this._s[c]; if (on) this._s[c] = true; else delete this._s[c]; return on; },
      contains: function (c) { return !!this._s[c]; }
    },
    append: function () { for (var i = 0; i < arguments.length; i++) this.appendChild(arguments[i]); },
    appendChild: function (c) {
      if (typeof c === "string") { var t = __el("#text"); t.textContent = c; c = t; }
      c.parentNode = this; this.children.push(c); return c;
    },
    replaceChildren: function () { this.children = []; this.append.apply(this, arguments); },
    remove: function () {},
    setAttribute: function (k, v) { this.attributes[k] = String(v); if (k === "id") this.id = String(v); },
    getAttribute: function (k) { return k in this.attributes ? this.attributes[k] : null; },
    removeAttribute: function (k) { delete this.attributes[k]; },
    addEventListener: function (t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    dispatch: function (t, ev) { (this.listeners[t] || []).forEach(function (f) { f(ev || { target: el }); }); },
    focus: function () {}, click: function () { this.dispatch("click"); },
    closest: function () { return null; },
    querySelector: function (sel) { var all = __find(this, sel); return all.length ? all[0] : null; },
    querySelectorAll: function (sel) { return __find(this, sel); },
    on: function () {}, removeAllListeners: function () {}
  };
  return el;
}
function __matches(el, sel) {
  var m = sel.match(/^([a-z]*)(?:\[type=([a-z]+)\])?(?:\.([\w-]+))?$/i);
  if (!m) return false;
  if (m[1] && el.tagName !== m[1].toUpperCase()) return false;
  if (m[2] && el.type !== m[2]) return false;
  if (m[3] && !el.classList.contains(m[3]) && (" " + el.className + " ").indexOf(" " + m[3] + " ") < 0) return false;
  return true;
}
function __find(root, sel) {
  var out = [];
  (function walk(n) { n.children.forEach(function (c) { if (__matches(c, sel)) out.push(c); walk(c); }); })(root);
  return out;
}
function __text(el) {
  return el.textContent + el.children.map(__text).join("");
}
var __ids = {};
var document = {
  body: __el("body"), title: "", visibilityState: "visible",
  getElementById: function (id) { if (!__ids[id]) { __ids[id] = __el("div"); __ids[id].id = id; } return __ids[id]; },
  createElement: function (t) { return __el(t); },
  addEventListener: function () {},
  querySelectorAll: function () { return []; }
};
var window = {
  devicePixelRatio: 1, innerWidth: 400, innerHeight: 800,
  matchMedia: function () { return { matches: true, addEventListener: function () {} }; },
  addEventListener: function () {}
};
var location = { search: "?token=t", hash: "" };
var localStorage = { getItem: function () { return null; }, setItem: function () {} };
function URLSearchParams(s) { this.get = function (k) { return k === "token" ? "t" : null; }; }
function setTimeout(f, ms) { __timers.push([f, ms]); return __timers.length; }
function clearTimeout() {}
function setInterval() { return 0; }
function clearInterval() {}
function fetch(path, options) {
  __posts.push([path, options && options.body ? JSON.parse(options.body) : null]);
  return { then: function () { return { then: function () { return this; }, catch: function () { return this; } }; } };
}
var Plotly = { react: function () {}, Plots: { resize: function () {} }, relayout: function () {}, restyle: function () {} };
var console = { log: function () {} };
"""


def run_app(test_js: str) -> Any:
    """Evaluate app.js on the stand-in page, then `test_js`, whose last expression is JSON-dumped."""
    source = DOM + "\n" + APP_JS.read_text() + "\n;JSON.stringify((function () {\n" + test_js + "\n})());"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(source)
        path = handle.name
    try:
        done = subprocess.run(
            ["osascript", "-l", "JavaScript", path], capture_output=True, text=True, timeout=60
        )
    finally:
        Path(path).unlink(missing_ok=True)
    if done.returncode != 0:
        raise AssertionError(f"app.js failed on the stand-in page: {done.stderr.strip()}")
    return json.loads(done.stdout.strip())
