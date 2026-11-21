"""A phone's first plot keeps every control (emode#1701 review).

Eric, 2026-11-15, on his phone: "The workbench view needs to show all options
and settings. Currently I see none." Below 800 px the stage is a column and the
plot's element has no height until its first reply sizes it (#185), so the tab
asked EMode for a plot ``pixels`` tall of 0. An EME device's x-y cut (#180)
came back one row tall. Plotly cannot draw a one-row heatmap smoothed
(``zsmooth: 'best'``) across a wide range: ``createImageData`` throws, and the
tab's ``draw()`` stopped before it built the control bar. These tests hold each
link of that chain: the tab never asks for a zero height, the server never
passes one on, a one-row raster is drawn unsmoothed, a figure Plotly refuses
still gets its controls, and in a real browser at phone width the bar is there.
"""

from __future__ import annotations

import http.client
import json
import re
import subprocess
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from emode.plotting import Raster
from emode.ui.figure import PlotOptions, compose
from emode.ui.renderers.plotly import to_plotly
from emode.ui.server import UIServer
from emode.ui.views import ViewRegistry

from js_harness import needs_jsc, run_app
from plot_fixtures import xy_view
from test_colorbar_render import CHROME
from test_cut_panel import ARGUMENTS, cut_view


# ------------------------------------------------------------------ the tab's size request


@needs_jsc
def test_the_tab_never_asks_for_a_plot_zero_pixels_tall():
    got = run_app(
        """
var fig = document.getElementById("figure");
fig.clientWidth = 374; fig.clientHeight = 0;
document.getElementById("stage").clientWidth = 374;
return pixels();
"""
    )
    width, height = got
    assert width == 374
    # the plot box's default aspect, sqrt(1/2), until a reply gives its own
    assert height == round(374 * 2**-0.5)


@needs_jsc
def test_a_figure_plotly_refuses_still_gets_its_controls():
    got = run_app(
        """
Plotly.react = function () {
  throw new Error("Failed to execute 'createImageData': The source height is zero or not a number.");
};
var threw = null;
try {
  draw({id: "p1", kind: "plot", title: "z-x Ex abs", aspect: 0.7,
    menu: {data: ["Ex", "Ey"], components: ["abs", "real"], selected: {data: "Ex", component: "abs"}},
    controls: {}, extra: {menus: []},
    figure: {data: [], layout: {margin: {l: 60, r: 20, t: 40, b: 50}}}});
} catch (e) { threw = String(e); }
var box = document.getElementById("controls");
var message = document.getElementById("message");
return {threw: threw, controls: box.children.length, selects: box.querySelectorAll("select").length,
  text: __text(box), message: message.hidden ? "" : message.textContent};
"""
    )
    assert got["threw"] is None
    assert (
        got["selects"] >= 2 and "Aspect" in got["text"] and "Auto render" in got["text"]
    )
    assert "createImageData" in got["message"]


# ------------------------------------------------------------------ the server


def _cut_server() -> Tuple[UIServer, List[Tuple[str, Tuple[int, int]]]]:
    """A z-x plot with its x-y cut, fetched from a stand-in EMode that records each panel's pixels."""
    registry = ViewRegistry()
    # 'auto' z sampling follows the tab's width, so a narrower tab fetches the plot again
    registry.add(
        cut_view(), {**ARGUMENTS, "resolution": "auto"}, PlotOptions(), (1200, 750)
    )
    asked: List[Tuple[str, Tuple[int, int]]] = []

    def fetch(view: Any, args: Dict[str, Any], pixels: Tuple[int, int]) -> Any:
        asked.append((str(args.get("plane")), tuple(int(p) for p in pixels)))
        return one_row_where_asked(pixels) if args.get("plane") == "x-y" else cut_view()

    return UIServer(registry, fetch).start(), asked


def one_row_where_asked(pixels: Tuple[int, int]) -> Any:
    """The x-y field sampled at the panel's pixels, as EMode samples it: one row for a panel 0 tall."""
    base = xy_view()
    rows = max(1, int(pixels[1]) // 40)
    x = np.linspace(-900.0, 900.0, 40)
    y = np.linspace(-400.0, 600.0, rows + 1)[:rows] + 12.5
    z = np.exp(-((x[None, :] / 300) ** 2) - ((y[:, None] - 110) / 150) ** 2)
    return base.model_copy(
        update={
            "rasters": [
                Raster(x=x, y=y, z=z, kind="diverging", interpolation="bilinear")
            ]
        }
    )


def _post(server: UIServer, pixels: List[int]) -> Dict[str, Any]:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    body = json.dumps({"controls": {}, "pixels": pixels, "width": 374}).encode()
    headers = {
        "Host": f"127.0.0.1:{server.port}",
        "X-EMode-Token": server.token,
        "Content-Type": "application/json",
    }
    conn.request("POST", "/api/views/p1/data", body=body, headers=headers)
    reply = json.loads(conn.getresponse().read())
    conn.close()
    return reply


@pytest.mark.parametrize("pixels", [[374, 0], [374, -5]])
def test_the_server_never_passes_a_zero_height_on_to_emode(pixels):
    server, asked = _cut_server()
    try:
        reply = _post(server, pixels)
    finally:
        server.stop()
    assert "error" not in reply, reply
    assert asked, "the narrower tab should have fetched the plot again"
    assert all(h > 0 for _, (_, h) in asked), asked


# ------------------------------------------------------------------ the renderer


def test_a_one_row_raster_is_drawn_unsmoothed():
    view = one_row_where_asked((374, 0))
    assert np.asarray(view.rasters[0].z).shape[0] == 1
    traces = [t for t in to_plotly(compose(view))["data"] if t["type"] == "heatmap"]
    assert traces and all(t["zsmooth"] is False for t in traces)


def test_a_raster_of_two_rows_or_more_stays_smoothed():
    view = one_row_where_asked((374, 400))
    traces = [t for t in to_plotly(compose(view))["data"] if t["type"] == "heatmap"]
    assert traces and all(t["zsmooth"] == "best" for t in traces)


# ------------------------------------------------------------------ in a real browser

#: The controls every plot of this kind shows: its own, then the figure's.
EXPECTED = ("Data:", "Aspect:", "Axes", "Auto render", "Render", "Export")


def _dom(server: UIServer, width: int, height: int) -> str:
    assert CHROME is not None
    url = f"http://127.0.0.1:{server.port}/?token={server.token}#/plots/p1"
    done = subprocess.run(
        [
            CHROME,
            "--headless",
            "--no-sandbox",  # the Linux runner forbids the sandbox's user namespaces
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--window-size={width},{height}",
            "--virtual-time-budget=8000",
            "--dump-dom",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    return done.stdout


def _controls(dom: str) -> str:
    m = re.search(r'<div id="controls"[^>]*>(.*?)</div>\s*<div id="message"', dom, re.S)
    if m is None:
        m = re.search(r'id="controls"[^>]*>(.*)', dom, re.S)
    assert m is not None, "the page has no control bar"
    return re.sub(r"<[^>]+>", " ", m.group(1))


@pytest.mark.skipif(CHROME is None, reason="needs a headless Chrome")
@pytest.mark.parametrize("size", [(390, 844), (1280, 900)], ids=["phone", "desktop"])
def test_a_real_browser_shows_every_control(size):
    server, asked = _cut_server()
    try:
        dom = _dom(server, *size)
    finally:
        server.stop()
    text = _controls(dom)
    missing = [name for name in EXPECTED if name not in text]
    assert not missing, f"missing {missing} at {size}: {text[:400]!r}"
    assert all(h > 0 for _, (_, h) in asked), asked
