"""The tab's View menu (emodeconnection#146) and an EME device's z position (emode#1625).

An EME device's plot offers its views, 'layout', 'simulation', 'schematic'
and 'gds' when it has GDS sections, in ``MenuState.views``. Choosing layout or
simulation refetches the view with ``view=``, and choosing the schematic or
the GDS view opens it as a plot of its own. A mode solve's plot offers none.

On a z panel, ``MenuState.position`` is the z of the x-y cut, which
``z_position`` sets and ``PlotView.slice_marker`` marks on the panel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emode.plotting import MenuState, PlotView, Polyline, Slider
from emode.ui.figure import DATA_ARGUMENTS, PlotOptions, compose
from emode.ui.renderers.plotly import to_plotly
from emode.ui.server import UIServer
from emode.ui.views import ViewRegistry

from plot_fixtures import schematic, xy_view, zx_view
from test_plot_view_types import _over_the_wire
from test_ui_server import request

APP_JS = Path(__file__).resolve().parents[1] / "emode" / "ui" / "static" / "app.js"
EME_VIEWS = ["layout", "simulation", "schematic"]


def eme_zx() -> PlotView:
    view = zx_view()
    menu = view.menu.model_copy(
        update={
            "views": EME_VIEWS,
            "position": Slider(label="z:", min=0.0, max=3.0, unit="μm", value="1.5 μm"),
            "selected": {**view.menu.selected, "view": "simulation"},
        }
    )
    marker = Polyline(kind="slice", x=[1500.0, 1500.0], y=[-1200.0, 1200.0])
    return view.model_copy(update={"menu": menu, "slice_marker": marker})


def _server(data: PlotView):
    registry = ViewRegistry()
    view = registry.add(data, {"plane": "z-x"}, PlotOptions(), (1200, 750))
    fetched = []

    def fetch(v, arguments, pixels):
        fetched.append((v.kind, dict(arguments)))
        if arguments.get("view") == "schematic":
            return schematic()
        return eme_zx()

    server = UIServer(registry, fetch).start()
    return server, view, fetched


def test_menu_state_carries_views_and_the_position_across_the_wire() -> None:
    back = _over_the_wire(eme_zx()).menu
    assert back.views == EME_VIEWS
    assert back.position.value == "1.5 μm"
    assert MenuState().views == [] and MenuState().position is None


def test_the_reply_offers_the_eme_views() -> None:
    server, view, _ = _server(eme_zx())
    try:
        status, _, body = request(
            server, "POST", f"/api/views/{view.id}/data", body={"controls": {}}
        )
        reply = json.loads(body)
        assert reply["menu"]["views"] == EME_VIEWS
        assert reply["menu"]["selected"]["view"] == "simulation"
        assert reply["menu"]["position"]["value"] == "1.5 μm"
    finally:
        server.stop()


def test_a_mode_solves_plot_offers_no_view_menu() -> None:
    server, view, _ = _server(xy_view())
    try:
        _, _, body = request(
            server, "POST", f"/api/views/{view.id}/data", body={"controls": {}}
        )
        assert json.loads(body)["menu"]["views"] == []
    finally:
        server.stop()


def test_choosing_layout_refetches_with_view() -> None:
    server, view, fetched = _server(eme_zx())
    try:
        request(
            server,
            "POST",
            f"/api/views/{view.id}/data",
            body={"controls": {"view": "layout"}},
        )
        assert fetched[-1] == ("plot", {"plane": "z-x", "view": "layout"})
    finally:
        server.stop()


def test_choosing_the_schematic_opens_it_as_its_own_plot() -> None:
    server, view, fetched = _server(eme_zx())
    try:
        status, _, body = request(
            server, "POST", f"/api/views/{view.id}/open", body={"view": "schematic"}
        )
        assert status == 200
        opened = json.loads(body)["id"]
        assert fetched[-1] == ("schematic", {"plane": "z-x", "view": "schematic"})
        new = server.registry.get(opened)
        assert new is not None and new.kind == "schematic"
        assert server.registry.get(view.id) is view
    finally:
        server.stop()


def test_open_refuses_a_view_the_menu_does_not_offer() -> None:
    server, view, fetched = _server(eme_zx())
    try:
        _, _, body = request(
            server, "POST", f"/api/views/{view.id}/open", body={"view": "gds"}
        )
        assert "error" in json.loads(body)
        assert fetched == []
    finally:
        server.stop()


def test_z_position_is_a_data_argument_that_refetches() -> None:
    assert "z_position" in DATA_ARGUMENTS
    server, view, fetched = _server(eme_zx())
    try:
        request(
            server,
            "POST",
            f"/api/views/{view.id}/data",
            body={"controls": {"z_position": 2.0}},
        )
        assert fetched[-1][1]["z_position"] == 2.0
    finally:
        server.stop()


def test_the_slice_marker_is_drawn_in_both_renderers() -> None:
    spec = compose(eme_zx())
    (marker,) = [line for line in spec.lines if line.kind == "slice"]
    assert np.allclose(marker.x, [1.5, 1.5])  # z in um
    figure = to_plotly(spec)
    assert [t for t in figure["data"] if t.get("meta") == "slice"]


def test_no_marker_without_one() -> None:
    assert not [line for line in compose(zx_view()).lines if line.kind == "slice"]


def test_the_app_builds_the_view_menu_and_the_z_position() -> None:
    js = APP_JS.read_text()
    assert "m.views" in js
    assert "/open" in js
    assert "m.position" in js and "z_position" in js
