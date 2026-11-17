"""A z-x or z-y plot of an EME device draws the x-y field at its cut below it (#180).

Eric, 2026-11-15: "What is the cut z supposed to do? Can you make it plot the
x-y field as a secondary plot below the z-x/y plot?" The Cut z control of an
EME device's z panel (``MenuState.position``) marks the z of the x-y cut. The
tab now also shows the device's x-y field at that z as a second panel below,
with the same component, normalization, mode, port and colormap, fetched again
when the cut moves. Scripts' handles and the headless path draw the z panel alone.
"""

from __future__ import annotations

import http.client
import json
from typing import Any, Dict, List, Tuple

import numpy as np

from emode.plotting import Polyline, Slider
from emode.ui.figure import PLOT_RATIO, PlotOptions, compose
from emode.ui.server import UIServer
from emode.ui.views import ViewRegistry

from plot_fixtures import xy_view, zx_view

#: The z panel's data arguments, as plot() and the tab leave them.
ARGUMENTS = {
    "plane": "z-x",
    "component": "Ex",
    "plot_function": "abs",
    "normalization": "component",
    "mode": "1, TM",
    "port": "left",
    "view": "simulation",
    "slice_position": "sum",
    "section": "s1",
    "resolution": 50,
    "z_position": 1.5,
}


def cut_view(z: float = 1.5) -> Any:
    """A z-x panel of an EME device with its x-y cut at `z` um, as EMode returns it."""
    base = zx_view()
    position = Slider(label="z:", min=0.0, max=3.0, unit="μm", value=f"{z:g} μm")
    marker = Polyline(kind="slice", x=[z * 1000, z * 1000], y=[-1200.0, 1200.0])
    return base.model_copy(update={"menu": base.menu.model_copy(update={"position": position}), "slice_marker": marker})


def _server(first: Any = None, arguments: Dict[str, Any] = ARGUMENTS) -> Tuple[UIServer, List[Dict[str, Any]]]:
    registry = ViewRegistry()
    registry.add(first if first is not None else cut_view(), dict(arguments), PlotOptions(), (1200, 750))
    fetched: List[Dict[str, Any]] = []

    def fetch(view: Any, args: Dict[str, Any], pixels: Tuple[int, int]) -> Any:
        fetched.append(dict(args))
        if args.get("plane") == "x-y":
            return xy_view()
        return cut_view(float(args.get("z_position") or 0.0))

    return UIServer(registry, fetch).start(), fetched


def _post(server: UIServer, controls: Dict[str, Any], view: str = "p1") -> Dict[str, Any]:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    body = json.dumps({"controls": controls, "pixels": [1170, 720], "width": 390}).encode()
    headers = {"Host": f"127.0.0.1:{server.port}", "X-EMode-Token": server.token, "Content-Type": "application/json"}
    conn.request("POST", f"/api/views/{view}/data", body=body, headers=headers)
    reply = json.loads(conn.getresponse().read())
    conn.close()
    assert "error" not in reply, reply
    return reply


def _xy_fetches(fetched: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [a for a in fetched if a.get("plane") == "x-y"]


def _domains(figure: Dict[str, Any]) -> List[Tuple[List[float], List[float]]]:
    layout = figure["layout"]
    out = []
    for suffix in ("", "2", "3"):
        if f"xaxis{suffix}" in layout:
            out.append((layout[f"xaxis{suffix}"].get("domain", [0, 1]), layout[f"yaxis{suffix}"].get("domain", [0, 1])))
    return out


def test_the_cut_draws_the_x_y_field_below_the_z_panel():
    server, fetched = _server()
    try:
        reply = _post(server, {})
    finally:
        server.stop()
    (xy,) = _xy_fetches(fetched)
    assert xy["z_position"] == 1.5
    for name in ("component", "plot_function", "normalization", "mode", "port", "view"):
        assert xy[name] == ARGUMENTS[name], name
    # the z panel's transverse position, section zoom and z sampling mean nothing on x-y
    for name in ("slice_position", "section", "resolution"):
        assert xy.get(name) is None, name
    (top_x, top_y), (bottom_x, bottom_y) = _domains(reply["figure"])
    assert min(top_y) >= max(bottom_y), "the x-y panel is not below the z panel"
    assert top_x == bottom_x, "the two panels are not one column"
    assert len([t for t in reply["figure"]["data"] if t["type"] == "heatmap" and t.get("xaxis") == "x2"]) == 1


def test_each_panel_has_the_default_aspect():
    server, _ = _server()
    try:
        reply = _post(server, {})
    finally:
        server.stop()
    assert abs(reply["aspect"] - 2 / PLOT_RATIO) < 1e-9


def test_the_menu_and_controls_are_the_z_panels():
    server, _ = _server()
    try:
        reply = _post(server, {})
    finally:
        server.stop()
    assert reply["menu"]["selected"]["plane"] == "z-x"
    assert reply["menu"]["position"]["value"] == "1.5 μm"
    assert reply["controls"]["plane"] == "z-x" and reply["controls"]["z_position"] == 1.5


def test_moving_the_cut_fetches_the_x_y_field_there():
    server, fetched = _server()
    try:
        _post(server, {})
        _post(server, {"z_position": 2.25})
    finally:
        server.stop()
    assert [a["z_position"] for a in _xy_fetches(fetched)] == [1.5, 2.25]


def test_a_figure_option_draws_both_again_without_asking_emode():
    server, fetched = _server()
    try:
        _post(server, {})
        before = len(fetched)
        reply = _post(server, {"colormap": "gray"})
    finally:
        server.stop()
    assert len(fetched) == before
    scales = [json.dumps(t["colorscale"]) for t in reply["figure"]["data"] if t["type"] == "heatmap"]
    assert len(scales) >= 2 and len(set(scales)) == 1, "the two panels are not on one colormap"


def test_a_z_panel_with_no_cut_is_drawn_alone():
    server, fetched = _server(first=zx_view())
    try:
        reply = _post(server, {})
    finally:
        server.stop()
    assert not _xy_fetches(fetched)
    assert len(_domains(reply["figure"])) == 1


def test_an_x_y_plot_has_no_second_panel():
    server, fetched = _server(first=xy_view(), arguments={"plane": "x-y"})
    try:
        reply = _post(server, {})
    finally:
        server.stop()
    assert not fetched
    assert len(_domains(reply["figure"])) == 1


def test_a_scripts_figure_of_the_z_panel_is_unchanged():
    spec = compose(cut_view())
    assert not spec.panels and spec.rasters
    assert np.isclose(spec.axes.box_aspect, 1 / PLOT_RATIO)
