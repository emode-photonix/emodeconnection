"""The tab's routes for a GDS layer view: its data and its layer panel, behind the token.

The layer panel restyles the figure in the browser and posts the new state to
``/api/views/<id>/layers``, which keeps it with the view for the session (so a
saved file and a later visit draw the same) and answers the panel's rows. A
KLayout ``.lyp`` file the tab read goes the same way as text.
"""
from __future__ import annotations

import pytest

from emode.ui.figure import PlotOptions
from emode.ui.gds import GDSPanel
from emode.ui.server import UIServer
from emode.ui.views import ViewRegistry

from gds_fixtures import LYP, gds_view
from test_ui_server import request


@pytest.fixture
def server():
    registry = ViewRegistry()
    registry.add(GDSPanel.from_view(gds_view()), {"view": "gds"}, PlotOptions(), (1200, 750))
    fetched = []

    def fetch(view, arguments, pixels):
        fetched.append(arguments)
        return gds_view()

    s = UIServer(registry, fetch).start()
    s.fetched = fetched  # type: ignore[attr-defined]
    yield s
    s.stop()


def _json(reply):
    import json

    status, _, body = reply
    assert status == 200
    return json.loads(body)


def test_the_data_route_answers_the_layers_figure_and_the_panels_rows(server):
    reply = _json(request(server, "POST", "/api/views/g1/data", body={"controls": {}}))
    assert reply["kind"] == "gds"
    assert [r["key"] for r in reply["extra"]["layers"]] == ["1/0", "2/0", "2/5"]
    assert reply["extra"]["shapes"] is False
    metas = [t.get("meta") for t in reply["figure"]["data"]]
    assert "layer:2/5" in metas
    assert not server.fetched


def test_the_layers_route_keeps_the_panel_for_the_session_without_fetching(server):
    reply = _json(
        request(server, "POST", "/api/views/g1/layers", body={"layers": {"2/0": {"visible": False}}, "shapes": True})
    )
    assert [r["visible"] for r in reply["layers"]] == [True, False, True]
    assert reply["shapes"] is True
    view = server.registry.get("g1")
    assert not view.data.style("2/0").visible and view.data.shapes
    # a later visit and a saved file draw the same
    again = _json(request(server, "POST", "/api/views/g1/data", body={"controls": {}}))
    assert [r["visible"] for r in again["extra"]["layers"]] == [True, False, True]
    assert not server.fetched


def test_solo_then_show_all_through_the_route(server):
    solo = _json(request(server, "POST", "/api/views/g1/layers", body={"solo": "1/0"}))
    assert [r["visible"] for r in solo["layers"]] == [True, False, False]
    every = _json(request(server, "POST", "/api/views/g1/layers", body={"show_all": True}))
    assert all(r["visible"] for r in every["layers"])


def test_a_layer_properties_file_is_read_from_the_text_the_tab_posts(server):
    reply = _json(request(server, "POST", "/api/views/g1/layers", body={"layer_properties": LYP}))
    assert reply["layers"][0]["label"] == "WG (1/0)"
    assert reply["layers"][0]["color"] == "#ff8000"
    assert reply["layers"][1]["visible"] is False


def test_a_bad_change_is_an_error_in_the_reply_and_changes_nothing(server):
    reply = _json(request(server, "POST", "/api/views/g1/layers", body={"layers": {"1/0": {"hatch": "*"}}}))
    assert "hatch" in reply["error"]
    assert server.registry.get("g1").data.style("1/0").hatch == ""


def test_the_layers_route_of_a_view_that_is_not_gds_is_refused(server):
    from plot_fixtures import zx_view

    server.registry.add(zx_view(), {"plane": "z-x"}, PlotOptions(), (1200, 750))
    reply = _json(request(server, "POST", "/api/views/p2/layers", body={"shapes": True}))
    assert "GDS" in reply["error"]


@pytest.mark.parametrize("token", [None, "wrong"])
def test_the_layers_route_refuses_a_missing_or_wrong_token(server, token):
    status, _, body = request(server, "POST", "/api/views/g1/layers", token=token, body={"shapes": True})
    assert (status, body) == (403, b"")
    assert not server.registry.get("g1").data.shapes


def test_the_layers_route_refuses_a_foreign_origin(server):
    status, _, _ = request(
        server, "POST", "/api/views/g1/layers", origin="http://evil.example", body={"shapes": True}
    )
    assert status == 403
    assert not server.registry.get("g1").data.shapes


def test_a_saved_file_draws_the_panels_layers(server, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    request(server, "POST", "/api/views/g1/layers", body={"layers": {"1/0": {"hatch": "x"}}})
    reply = _json(request(server, "POST", "/api/views/g1/save", body={"file_name": "g", "file_type": "svg"}))
    assert reply["path"].endswith("g.svg")
    assert "<svg" in (tmp_path / "EM_figures" / "g.svg").read_text()
