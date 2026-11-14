"""The tab exports PNG and SVG itself through plotly.js, PDF through matplotlib (emodeconnection#164).

Eric, 2026-11-13: the browser tab draws PNG and SVG with the Plotly it already
has, so no matplotlib runs for them. PDF goes on through the client's export
route, which draws with matplotlib, as ``plot(file_name=...)`` does from a
script. Save writes the tab's own image bytes for PNG and SVG.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from emode.ui.figure import PlotOptions
from emode.ui.renderers import mpl
from emode.ui.server import UIServer
from emode.ui.views import ViewRegistry

from js_harness import needs_jsc, run_app
from plot_fixtures import zx_view
from test_ui_server import request

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = ViewRegistry()
    registry.add(zx_view(), {"plane": "z-x"}, PlotOptions(), (1200, 750))
    s = UIServer(registry, lambda v, a, p: zx_view()).start()
    yield s
    s.stop()


def _no_matplotlib(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("matplotlib drew a figure the tab had drawn")

    monkeypatch.setattr(mpl, "render", refuse)


@pytest.mark.parametrize(
    "file_type, data, url",
    [
        ("png", PNG, "data:image/png;base64," + base64.b64encode(PNG).decode()),
        ("svg", SVG, "data:image/svg+xml;base64," + base64.b64encode(SVG).decode()),
        (
            "svg",
            SVG,
            "data:image/svg+xml,"
            + __import__("urllib.parse").parse.quote(SVG.decode()),
        ),
    ],
    ids=["png", "svg-base64", "svg-url-encoded"],
)
def test_save_writes_the_tabs_own_image_without_matplotlib(
    server, tmp_path, monkeypatch, file_type, data, url
):
    _no_matplotlib(monkeypatch)
    _, _, body = request(
        server,
        "POST",
        "/api/views/p1/save",
        body={"file_name": "mine", "file_type": file_type, "image": url},
    )
    reply = json.loads(body)
    path = Path(reply["path"])
    assert path == (tmp_path / "EM_figures" / f"mine.{file_type}").resolve()
    assert path.read_bytes() == data


def test_save_refuses_an_image_of_another_type(server, monkeypatch):
    _no_matplotlib(monkeypatch)
    url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    _, _, body = request(
        server,
        "POST",
        "/api/views/p1/save",
        body={"file_name": "x", "file_type": "svg", "image": url},
    )
    assert "error" in json.loads(body)


def test_save_refuses_an_image_for_a_pdf(server, monkeypatch):
    _no_matplotlib(monkeypatch)
    url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    _, _, body = request(
        server,
        "POST",
        "/api/views/p1/save",
        body={"file_name": "x", "file_type": "pdf", "image": url},
    )
    assert "error" in json.loads(body)


def test_the_export_route_still_serves_a_pdf(server):
    status, headers, body = request(
        server,
        "POST",
        "/api/views/p1/export",
        body={"file_name": "f", "file_type": "pdf"},
    )
    assert status == 200
    assert headers["content-type"] == "application/pdf"
    assert body.startswith(b"%PDF")


EXPORT = """
var calls = [];
Plotly.downloadImage = function (el, o) { calls.push(["download", o]); };
Plotly.toImage = function (el, o) {
  calls.push(["toImage", o]);
  return { then: function (f) { f("data:image/png;base64,AAAA"); return { catch: function () {} }; } };
};
var posted = [];
api = function (path, body) { posted.push([path, body]); return { then: function () { return this; }, catch: function () {} }; };
downloadServer = function (id, format, options) { posted.push(["server " + format, options]); };
var panel = exportPanel({id: "p1", controls: {figure_size: [80, 50], dpi: 300}});
var format = panel.querySelectorAll("select")[0];
var buttons = panel.querySelectorAll("button").filter(function (b) {
  return b.textContent === "Download" || b.textContent === "Save"; });
function press(value, name) {
  format.value = value;
  buttons.filter(function (b) { return b.textContent === name; })[0].click();
}
"""


@needs_jsc
def test_the_format_menu_offers_png_and_svg_from_the_browser_and_pdf():
    got = run_app(
        EXPORT
        + "return format.children.map(function (o) { return [o.value, o.textContent]; });"
    )
    assert [v for v, _ in got] == ["png", "svg", "pdf"]
    assert all("matplotlib" not in text for v, text in got if v != "pdf")


@needs_jsc
def test_png_and_svg_download_through_plotly_at_the_size_and_dpi():
    got = run_app(
        EXPORT
        + 'press("png", "Download"); press("svg", "Download"); return {calls: calls, posted: posted};'
    )
    (_, png), (_, svg) = got["calls"]
    assert png["format"] == "png" and svg["format"] == "svg"
    assert png["width"] == round(80 * 96 / 25.4) and png["height"] == round(
        50 * 96 / 25.4
    )
    assert png["scale"] == pytest.approx(300 / 96)
    assert svg["scale"] == 1
    assert got["posted"] == []


@needs_jsc
def test_pdf_downloads_through_the_export_route():
    got = run_app(
        EXPORT + 'press("pdf", "Download"); return {calls: calls, posted: posted};'
    )
    assert got["calls"] == []
    ((what, options),) = got["posted"]
    assert what == "server pdf"


@needs_jsc
def test_save_of_a_png_posts_plotlys_image():
    got = run_app(
        EXPORT + 'press("png", "Save"); return {calls: calls, posted: posted};'
    )
    assert got["calls"][0][0] == "toImage"
    ((path, body),) = got["posted"]
    assert path == "views/p1/save"
    assert body["file_type"] == "png"
    assert body["image"].startswith("data:image/png")


@needs_jsc
def test_save_of_a_pdf_posts_no_image():
    got = run_app(
        EXPORT + 'press("pdf", "Save"); return {calls: calls, posted: posted};'
    )
    assert got["calls"] == []
    ((path, body),) = got["posted"]
    assert (
        path == "views/p1/save" and body["file_type"] == "pdf" and "image" not in body
    )


NO_PDF = "Saving as PDF needs matplotlib. Install it with: pip install matplotlib. PNG and SVG work without it."


@pytest.fixture
def no_matplotlib(monkeypatch):
    import builtins
    import sys

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError(f"No module named {name!r}")
        return real(name, *args, **kwargs)

    for name in [
        m for m in sys.modules if m == "matplotlib" or m.startswith("matplotlib.")
    ]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(builtins, "__import__", refuse)


@pytest.mark.parametrize("action", ["export", "save"])
def test_a_pdf_without_matplotlib_answers_the_warning(server, no_matplotlib, action):
    status, headers, body = request(
        server,
        "POST",
        f"/api/views/p1/{action}",
        body={"file_name": "f", "file_type": "pdf"},
    )
    assert status == 200
    assert headers["content-type"] == "application/json"
    assert json.loads(body)["error"] == NO_PDF


@pytest.mark.parametrize(
    "file_type, url",
    [
        ("png", "data:image/png;base64," + base64.b64encode(PNG).decode()),
        ("svg", "data:image/svg+xml;base64," + base64.b64encode(SVG).decode()),
    ],
)
def test_png_and_svg_save_without_matplotlib(
    server, tmp_path, no_matplotlib, file_type, url
):
    _, _, body = request(
        server,
        "POST",
        "/api/views/p1/save",
        body={"file_name": "ok", "file_type": file_type, "image": url},
    )
    assert Path(json.loads(body)["path"]).is_file()


@needs_jsc
def test_the_pdf_option_is_labeled_pdf():
    got = run_app(
        EXPORT + "return format.children.map(function (o) { return o.textContent; });"
    )
    assert got == ["PNG", "SVG", "PDF"]


@needs_jsc
def test_the_tab_shows_the_routes_error():
    got = run_app(
        """
var shown = [];
showMessage = function (t) { shown.push(t); };
saved({error: "Saving as PDF needs matplotlib."});
return shown;
"""
    )
    assert got == ["Saving as PDF needs matplotlib."]
