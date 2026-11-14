"""Click a schematic component to see its settings and parameters (emodeconnection#159).

The schematic's components carry ids: ``Block.id`` (``'section:<name>'``), an
external ``Stub.id`` (``'port:<name>'``) and ``Schematic.connection_ids`` (one
``'junction:<a>.<port>|<b>.<port>'`` per connection). The tab posts an id to the
view's ``details`` route, the client asks EMode (``EM_component_details``) and
answers a ``ComponentDetails``: its title and its rows, each a group, a name and
a value. The page shows them in a closable panel that works on a phone, by tap,
by keyboard and with a screen reader.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from emode.plotting import Block, ComponentDetails, DetailRow, Schematic, Stub
from emode.types import EModeError
from emode.ui.figure import PlotOptions, compose
from emode.ui.renderers.plotly import to_plotly
from emode.ui.server import UIServer
from emode.ui.views import ViewRegistry

from js_harness import needs_jsc, run_app
from plot_fixtures import schematic, zx_view
from test_plot_view_types import _over_the_wire
from test_ui_server import request

UI = Path(__file__).resolve().parents[1] / "emode" / "ui"


def schematic_with_ids() -> Schematic:
    view = schematic()
    blocks = [b.model_copy(update={"id": f"section:{b.name}"}) for b in view.blocks]
    stubs = [
        s.model_copy(
            update={
                "id": f"port:{s.label}" if s.external else "junction:a.right|b.left"
            }
        )
        for s in view.stubs
    ]
    return view.model_copy(
        update={
            "blocks": blocks,
            "stubs": stubs,
            "connection_ids": ["junction:a.right|b.left"],
        }
    )


def details(component: str) -> ComponentDetails:
    return ComponentDetails(
        component=component,
        kind="section",
        title="a, straight",
        rows=[
            DetailRow(group="Section", name="length", value="1000 nm"),
            DetailRow(group="Profile _default", name="wavelength", value="1550 nm"),
        ],
    )


def test_the_ids_and_the_details_cross_the_wire():
    back = _over_the_wire(schematic_with_ids())
    assert [b.id for b in back.blocks] == ["section:a", "section:b"]
    assert back.stubs[0].id == "port:a.left"
    assert back.connection_ids == ["junction:a.right|b.left"]
    d = _over_the_wire(details("section:a"))
    assert d.rows[1] == DetailRow(
        group="Profile _default", name="wavelength", value="1550 nm"
    )
    # an older EMode sends none
    assert Block(name="a", label="a", x=0, y=0, width=1, height=1).id is None
    assert Stub(section="a", side="left", x0=0, y0=0, x1=1, y1=0).id is None


def test_the_figure_tags_each_component_for_a_click():
    figure = to_plotly(compose(schematic_with_ids()))
    metas = [t.get("meta") for t in figure["data"]]
    assert "component:section:a" in metas and "component:section:b" in metas
    assert "component:port:a.left" in metas
    assert "component:junction:a.right|b.left" in metas
    block = next(t for t in figure["data"] if t.get("meta") == "component:section:a")
    assert block["hoveron"] == "fills" and block["hoverinfo"] != "skip"


def _server(data, answer=None):
    registry = ViewRegistry()
    view = registry.add(data, {"view": "schematic"}, PlotOptions(), (1200, 750))
    asked = []

    def fetch_details(component):
        asked.append(component)
        if answer is not None:
            return answer(component)
        raise EModeError(f"no component {component!r}")

    server = UIServer(registry, lambda v, a, p: data, details=fetch_details).start()
    return server, view, asked


def test_the_details_route_answers_the_components_rows():
    server, view, asked = _server(
        schematic_with_ids(), lambda c: details(c).model_dump(mode="json")
    )
    try:
        status, _, body = request(
            server,
            "POST",
            f"/api/views/{view.id}/details",
            body={"component": "section:a"},
        )
        reply = json.loads(body)
        assert status == 200 and asked == ["section:a"]
        assert reply["title"] == "a, straight"
        row = reply["rows"][0]
        assert (row["group"], row["name"], row["value"]) == ("Section", "length", "1000 nm")
    finally:
        server.stop()


def test_the_details_route_refuses_an_id_the_schematic_does_not_carry():
    server, view, asked = _server(schematic_with_ids(), details)
    try:
        _, _, body = request(
            server,
            "POST",
            f"/api/views/{view.id}/details",
            body={"component": "section:zz"},
        )
        assert "error" in json.loads(body)
        assert asked == []
    finally:
        server.stop()


def test_the_details_route_answers_emodes_error():
    server, view, _ = _server(schematic_with_ids())
    try:
        _, _, body = request(
            server,
            "POST",
            f"/api/views/{view.id}/details",
            body={"component": "section:a"},
        )
        assert "no component" in json.loads(body)["error"]
    finally:
        server.stop()


def test_a_plot_view_has_no_components():
    server, view, asked = _server(zx_view(), details)
    try:
        _, _, body = request(
            server,
            "POST",
            f"/api/views/{view.id}/details",
            body={"component": "section:a"},
        )
        assert "error" in json.loads(body) and asked == []
    finally:
        server.stop()


def test_the_page_carries_the_panel_as_a_labeled_dialog_with_a_close_button():
    html = (UI / "pages" / "app.html").read_text()
    panel = re.search(r"<aside\b[^>]*id=\"details\"[^>]*>", html)
    assert panel, "no details panel"
    assert 'role="dialog"' in panel.group(0)
    assert 'aria-labelledby="details-title"' in panel.group(0)
    assert 'id="details-title"' in html
    close = re.search(r"<button\b[^>]*id=\"details-close\"[^>]*>", html)
    assert (
        close and 'type="button"' in close.group(0) and "aria-label=" in close.group(0)
    )
    css = (UI / "static" / "emode.css").read_text()
    narrow = css[css.index("@media (max-width: 800px)") :]
    assert "#details" in narrow


SCHEMATIC_REPLY = """
var reply = {id: "s1", kind: "schematic", aspect: 0.6, figure: {data: [], layout: {}},
  menu: {ports: ["a.left", "b.right"], selected: {plane: "schematic"}}, controls: {},
  extra: {components: [["section:a", "a"], ["section:b", "b"], ["port:a.left", "port a.left"],
                       ["junction:a.right|b.left", "a.right to b.left"]]}};
"""


@needs_jsc
def test_the_schematic_lists_its_components_in_a_menu_that_opens_the_details():
    got = run_app(
        SCHEMATIC_REPLY
        + """
var asked = [];
openDetails = function (id) { asked.push(id); };
state.reply = reply;
var box = document.getElementById("controls");
buildControls(reply);
var menu = box.querySelectorAll("select").filter(function (s) {
  return s.getAttribute("aria-label") === "Component details"; })[0];
var options = menu.children.map(function (o) { return o.value; });
menu.value = "port:a.left";
menu.dispatch("change", {target: menu});
return {options: options, asked: asked};
"""
    )
    assert got["options"][1:] == [
        "section:a",
        "section:b",
        "port:a.left",
        "junction:a.right|b.left",
    ]
    assert got["asked"] == ["port:a.left"]


@needs_jsc
def test_the_panel_shows_the_rows_by_group_and_closes():
    got = run_app(
        """
showDetails({component: "section:a", kind: "section", title: "a, straight", rows: [
  {group: "Section", name: "length", value: "1000 nm"},
  {group: "Section", name: "kind", value: "straight"},
  {group: "Profile _default", name: "wavelength", value: "1550 nm"}]});
var panel = document.getElementById("details");
var open = !panel.hidden;
var title = document.getElementById("details-title").textContent;
var body = __text(document.getElementById("details-body"));
var heads = document.getElementById("details-body").querySelectorAll("th").length;
closeDetails();
return {open: open, title: title, body: body, closed: panel.hidden};
"""
    )
    assert got["open"] and got["closed"]
    assert got["title"] == "a, straight"
    for text in ("Section", "length", "1000 nm", "Profile _default", "1550 nm"):
        assert text in got["body"]


def test_the_script_opens_the_details_from_a_click_on_the_figure():
    js = (UI / "static" / "app.js").read_text()
    assert "plotly_click" in js
    assert '"component:"' in js
    assert '/details"' in js or "/details'" in js or '"/details"' in js


def test_the_session_asks_emode_for_the_details():
    from plot_standin import session

    from emode.ui import session as ui

    em, client = session(version={**ui_version(), "views": ["plot", "details"]})
    client.answers["EM_component_details"] = lambda request: details(
        request["component"]
    )
    got = ui.fetch_details(em, "section:a")
    assert got["title"] == "a, straight"
    (sent,) = client.calls("EM_component_details")
    assert sent["component"] == "section:a"


def test_an_emode_without_the_details_is_told_to_update():
    from plot_standin import session

    from emode.ui import session as ui

    em, client = session()
    with pytest.raises(EModeError, match="1.0.9"):
        ui.fetch_details(em, "section:a")
    assert client.calls("EM_component_details") == []


def ui_version():
    from plot_standin import VERSION_1_0_9

    return dict(VERSION_1_0_9)
