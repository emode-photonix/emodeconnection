"""Saved views: the plot tab's presets, listed in its sidebar and kept in the simulation file (GUI-M4c, emode#1703).

A saved view is a recipe (``SavedView``: ``name``, ``kind``, ``arguments``,
``options``, ``dataset``, ``tags``, ``created``), never data. A script's
``plot()`` saves one automatically, named by its title and deduplicated by
recipe, so a script run twice lists its plots once. The workbench saves its
state with **Save** (back to the view it came from) or **Save as** (a new one),
and the sidebar renames and removes them. Each is posted to EMode
(``EM_save_view``, ``EM_delete_saved_view``), which keeps them in the ``.eph``,
and a session lists the file's views (``EM_saved_views``) when its tab starts.

Eric's decisions (emode#1698, 2026-11-15): saved views are presets, kept in
the ``.eph``, names first (tags in GUI-M4d), a script's view keeps its own
sidebar entry with an "Edit in workbench" action, and Save sits beside Render
and Export, outside the phone's Settings fold.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from emode import compat
from emode.nonlinear import CircuitResponse
from emode.plotting import SavedView
from emode.types import ArgumentError, EModeError
from emode.ui import session as ui

from js_harness import needs_jsc, run_app
from plot_fixtures import farfield
from test_plot_inventory import _round_trip
from test_workbench import _entry, _http, _server, bench, inventory
from test_workbench import desktop  # noqa: F401  (autouse fixture)
from test_workbench import em as _workbench_em  # noqa: F401  (fixture)

VIEWS = ["plot", "smatrix", "material", "session", "gds", "details", "inventory", "saved"]


def _recipe(name: str, **changes: Any) -> SavedView:
    recipe: Dict[str, Any] = {
        "name": name,
        "kind": "plot",
        "arguments": {"profile": "wg", "plane": "x-y"},
        "options": {},
        "created": "2026-11-15T10:00:00",
    }
    recipe.update(changes)
    return SavedView(**recipe)


class FileViews:
    """What EMode keeps under ``_saved_views``: the stand-in's answers to the three calls."""

    def __init__(self, views: Optional[List[SavedView]] = None) -> None:
        self.views: List[SavedView] = list(views or [])

    def listed(self, request: Dict[str, Any]) -> List[SavedView]:
        return list(self.views)

    def save(self, request: Dict[str, Any]) -> List[SavedView]:
        view = request["view"]
        names = [v.name for v in self.views]
        if view.name in names:
            self.views[names.index(view.name)] = view
        else:
            self.views.append(view)
        return list(self.views)

    def delete(self, request: Dict[str, Any]) -> List[SavedView]:
        names = [v.name for v in self.views]
        if request["name"] not in names:
            raise ArgumentError(f"no saved view {request['name']!r}", "delete_saved_view", "name")
        del self.views[names.index(request["name"])]
        return list(self.views)


def response() -> CircuitResponse:
    return CircuitResponse(
        port_sizes=(2, 2),
        port_names=("left", "right"),
        wavelengths=(1550.0, 1550.0),
        input=np.array([1e-3**0.5, 0.0, 0.0, 0.0], complex),
        output=np.array([0.0, 0.0, (0.9e-3) ** 0.5, (0.1e-3) ** 0.5], complex),
        converged=True,
        iterations=1,
        residual=0.0,
        linear=True,
    )


@pytest.fixture
def file_views():
    return FileViews()


@pytest.fixture
def em(_workbench_em, file_views):  # noqa: F811
    em = _workbench_em
    em.client.answers.update(
        {
            "EM_version": {"emode": "1.0.9", "protocol": 1, "plot": "view", "views": VIEWS},
            "EM_saved_views": file_views.listed,
            "EM_save_view": file_views.save,
            "EM_delete_saved_view": file_views.delete,
            "EM_close": None,
        }
    )
    return em


def _calls(em: Any, function: str) -> List[Dict[str, Any]]:
    return em.client.calls(function)


def _sidebar(em: Any) -> List[Dict[str, Any]]:
    status, views = _http(_server(em), "GET", "/api/views")
    assert status == 200
    return views[1:]  # after the pinned workbench


def _act(em: Any, **body: Any) -> Dict[str, Any]:
    status, reply = _http(_server(em), "POST", "/api/saved", body)
    assert status == 200
    return reply


# ------------------------------------------------------------------ the wire type


def test_a_saved_view_round_trips_the_wire():
    view = _recipe("TE", options={"colormap": "magma"}, dataset="profile:wg", tags=["paper"])
    assert _round_trip(view) == view


def test_the_saved_view_is_versioned_at_1_0_9():
    assert compat.TYPE_SINCE["SavedView"] == "1.0.9"


def test_an_emode_without_saved_views_is_refused_by_the_check():
    class Stand:
        def _emode_info(self) -> Dict[str, Any]:
            return {"emode": "1.0.9", "protocol": 1, "plot": "view", "views": ["plot", "inventory"]}

    with pytest.raises(EModeError, match="Saved views"):
        ui.check_plot_support(Stand(), "saved")  # type: ignore[arg-type]


def test_the_saved_view_calls_do_not_count_as_changing_the_session():
    from emode.emode import changes_session

    for function in ("EM_saved_views", "EM_save_view", "EM_delete_saved_view"):
        assert not changes_session(function), function


# ------------------------------------------------------------------ a script's plot()


def test_plot_adds_a_saved_entry_named_by_its_title_and_posts_its_recipe(em):
    handle = em.plot(plane="z-x", component="Ex", colormap="magma")
    (entry,) = _sidebar(em)
    assert entry["id"] == handle.id and entry["title"] == "z-x Ex abs"
    assert entry["script"] is True and entry["session_only"] is False
    (posted,) = _calls(em, "EM_save_view")
    view = posted["view"]
    assert isinstance(view, SavedView)
    assert view.name == "z-x Ex abs" and view.kind == "plot"
    assert view.arguments == {"plane": "z-x", "component": "Ex"}
    assert view.options == {"colormap": "magma"}  # only what differs from the defaults
    assert view.created


def test_the_same_plot_again_adds_no_entry_reselects_it_and_posts_nothing(em):
    first = em.plot(plane="z-x", component="Ex")
    server = _server(em)
    q = server.subscribe()
    again = em.plot(plane="z-x", component="Ex")
    assert again.id == first.id
    assert len(_sidebar(em)) == 1
    assert len(_calls(em, "EM_save_view")) == 1
    event, data = q.get(timeout=1)
    assert event == "view-added" and data["id"] == first.id


def test_a_named_plot_is_saved_under_its_name_and_a_second_replaces_it(em):
    first = em.plot(plane="z-x", component="Ex", name="TE")
    second = em.plot(plane="z-x", component="Ey", name="TE")
    (entry,) = _sidebar(em)
    assert entry["title"] == "TE" and second.id == first.id
    posted = [c["view"] for c in _calls(em, "EM_save_view")]
    assert [v.name for v in posted] == ["TE", "TE"]
    assert posted[-1].arguments["component"] == "Ey"
    assert em._views.get(first.id).arguments["component"] == "Ey"


def test_two_recipes_with_one_title_are_numbered(em):
    em.plot(plane="z-x", component="Ex", slice_position=100.0)
    em.plot(plane="z-x", component="Ex", slice_position=200.0)
    assert [e["title"] for e in _sidebar(em)] == ["z-x Ex abs", "z-x Ex abs (2)"]


def test_every_entry_point_takes_a_name(em):
    em.plot_S_matrix(name="through")
    em.material_explorer(material="Si", name="silicon")
    titles = [e["title"] for e in _sidebar(em)]
    assert titles == ["through", "silicon"]
    kinds = {c["view"].name: c["view"].kind for c in _calls(em, "EM_save_view")}
    assert kinds == {"through": "smatrix", "silicon": "material"}


def test_a_client_held_view_is_listed_for_the_session_and_never_posted(em):
    farfield().show(em=em, name="far")
    from emode.figures import Figure

    Figure(title="mine").plot(em=em)
    entries = _sidebar(em)
    assert [e["title"] for e in entries] == ["far", "mine"]
    assert all(e["session_only"] for e in entries)
    assert _calls(em, "EM_save_view") == []


def test_noninteractive_lists_the_plot_and_posts_nothing(em, monkeypatch):
    monkeypatch.setenv("EMODE_NONINTERACTIVE", "1")
    em.plot(plane="z-x", component="Ex")
    assert [e.name for e in em.saved_views()] == ["z-x Ex abs"]
    assert _calls(em, "EM_save_view") == []
    assert _calls(em, "EM_saved_views") == []


def test_headless_registers_nothing(em, monkeypatch):
    monkeypatch.setenv("EMODE_HEADLESS", "1")
    em.plot(plane="z-x", component="Ex")
    assert em.saved_views() == []
    assert _calls(em, "EM_save_view") == []


def test_an_emode_without_saved_views_still_plots_and_lists(em):
    em.client.answers["EM_version"] = {"emode": "1.0.9", "protocol": 1, "plot": "view", "views": VIEWS[:-1]}
    em.plot(plane="z-x", component="Ex")
    assert [e["title"] for e in _sidebar(em)] == ["z-x Ex abs"]
    assert _calls(em, "EM_save_view") == [] and _calls(em, "EM_saved_views") == []


# ------------------------------------------------------------------ the file's views


def test_the_files_views_are_listed_before_any_plot_in_file_order(em, file_views):
    file_views.views = [_recipe("b"), _recipe("a", kind="smatrix", arguments={}, dataset="smatrix")]
    em.browse()
    entries = _sidebar(em)
    assert [e["title"] for e in entries] == ["b", "a"]
    assert all(not e["script"] and e["opens"] == "bench" for e in entries)
    assert len(_calls(em, "EM_saved_views")) == 1
    _sidebar(em)
    assert len(_calls(em, "EM_saved_views")) == 1  # asked once a session


def test_saved_views_returns_saved_views_whose_show_replays_and_delete_posts(em, file_views):
    file_views.views = [_recipe("wg x-y", arguments={"profile": "wg", "plane": "x-y", "component": "Ey"})]
    (view,) = em.saved_views()
    assert isinstance(view, SavedView) and view.name == "wg x-y"
    plots = len(_calls(em, "EM_plot"))
    handle = view.show(em=em)
    sent = _calls(em, "EM_plot")[plots:]
    assert len(sent) == 1 and sent[0]["component"] == "Ey" and sent[0]["profile"] == "wg"
    assert handle.id in {e["id"] for e in _sidebar(em)}
    assert len(_sidebar(em)) == 1  # the replay is the same recipe, listed once
    assert _calls(em, "EM_save_view") == []
    view.delete(em=em)
    assert [c["name"] for c in _calls(em, "EM_delete_saved_view")] == ["wg x-y"]
    assert _sidebar(em) == []


def test_a_script_run_twice_lists_its_plot_once(em, file_views):
    """The second run opens the file the first one saved: its plot() finds the recipe already there."""
    file_views.views = [_recipe("z-x Ex abs", arguments={"plane": "z-x", "component": "Ex"})]
    em.plot(plane="z-x", component="Ex")
    entries = _sidebar(em)
    assert [e["title"] for e in entries] == ["z-x Ex abs"] and entries[0]["script"]
    assert _calls(em, "EM_save_view") == []


# ------------------------------------------------------------------ opening a saved view on the workbench


def test_opening_a_saved_entry_fetches_with_its_arguments_and_options(em, file_views):
    recipe = _recipe(
        "Ey", arguments={"profile": "wg", "plane": "x-y", "component": "Ey"}, options={"colormap": "magma"}
    )
    file_views.views = [recipe]
    (entry,) = _sidebar(em)
    plots = len(_calls(em, "EM_plot"))
    reply = bench(em, {"saved": entry["id"]})
    assert "error" not in reply, reply
    (sent,) = _calls(em, "EM_plot")[plots:]
    assert sent["component"] == "Ey" and sent["profile"] == "wg" and sent["plane"] == "x-y"
    assert reply["bench"]["dataset"] == "profile:wg"
    assert reply["bench"]["origin"] == {"id": entry["id"], "name": "Ey"}
    assert reply["bench"]["modified"] is False
    assert reply["controls"]["colormap"] == "magma"
    # the figure is a plot() with the same arguments
    handle = em.plot(profile="wg", plane="x-y", component="Ey", colormap="magma")
    from emode.ui.renderers.plotly import to_plotly

    assert json.dumps(reply["figure"]["data"]) == json.dumps(
        json.loads(json.dumps(to_plotly(handle.figure, (1170, 720))["data"], default=_jsonable))
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def test_a_saved_view_of_analysis_data_opens_by_its_dataset(em, file_views):
    file_views.views = [_recipe("modes", kind="sweep", arguments={"result": "TE_fraction"}, dataset="modes:FDM")]
    (entry,) = _sidebar(em)
    reply = bench(em, {"saved": entry["id"]})
    assert "error" not in reply, reply
    assert reply["bench"]["dataset"] == "modes:FDM" and reply["kind"] == "sweep"
    menus = {m["name"]: m for m in reply["extra"]["menus"]}
    assert menus["result"]["selected"] == "TE_fraction"


def test_a_saved_view_of_data_the_session_no_longer_holds_says_why(em, file_views):
    file_views.views = [_recipe("gone", kind="sweep", arguments={}, dataset="pulse")]
    (entry,) = _sidebar(em)
    reply = bench(em, {"saved": entry["id"]})
    assert reply["error"] == "run a pulsed EME() first"
    assert len(_sidebar(em)) == 1  # the entry stays for the user to fix or remove


# ------------------------------------------------------------------ Edit in workbench


def test_edit_in_workbench_loads_a_script_views_settings(em):
    handle = em.plot(profile="wg", plane="x-y", component="Ey", colormap="magma")
    plots = len(_calls(em, "EM_plot"))
    reply = bench(em, {"saved": handle.id})
    assert "error" not in reply, reply
    assert len(_calls(em, "EM_plot")) == plots  # the script's data, not fetched again
    assert reply["bench"]["dataset"] == "profile:wg"
    assert reply["bench"]["origin"] == {"id": handle.id, "name": "x-y Ex real"}
    assert reply["controls"]["component"] == "Ey" and reply["controls"]["colormap"] == "magma"
    assert reply["bench"]["modified"] is False
    # the script's own view is left as it was
    assert em._views.get(handle.id).arguments["component"] == "Ey"


def test_a_client_held_view_cannot_be_edited_on_the_workbench(em):
    farfield().show(em=em, name="far")
    (entry,) = _sidebar(em)
    reply = bench(em, {"saved": entry["id"]})
    assert "workbench" in reply["error"]


# ------------------------------------------------------------------ Save, Save as, Rename, Remove


def test_save_writes_the_bench_back_to_the_view_it_came_from(em):
    handle = em.plot(profile="wg", plane="x-y", component="Ex")
    bench(em, {"saved": handle.id})
    changed = bench(em, {"component": "Ey"})
    assert changed["bench"]["modified"] is True
    reply = _act(em, action="save")
    assert reply.get("error") is None, reply
    posted = _calls(em, "EM_save_view")[-1]["view"]
    assert posted.name == "x-y Ex real" and posted.arguments["component"] == "Ey"
    assert posted.dataset == "profile:wg"
    assert bench(em)["bench"]["modified"] is False
    # the entry shows what was saved
    assert em._views.get(handle.id).arguments["component"] == "Ey"
    assert len(_sidebar(em)) == 1


def test_save_with_no_saved_view_behind_the_bench_asks_for_a_name(em):
    bench(em, {"dataset": "smatrix"})
    reply = _act(em, action="save")
    assert "Save as" in reply["error"]
    assert _calls(em, "EM_save_view") == []


def test_save_as_posts_under_the_new_name_and_adds_an_entry(em):
    bench(em, {"dataset": "smatrix"})
    bench(em, {"plot_function": "angle"})
    reply = _act(em, action="save_as", name="phase")
    assert reply.get("error") is None, reply
    posted = _calls(em, "EM_save_view")[-1]["view"]
    assert posted.name == "phase" and posted.kind == "smatrix" and posted.dataset == "smatrix"
    assert posted.arguments["plot_function"] == "angle"
    entries = _sidebar(em)
    assert [e["title"] for e in entries] == ["phase"] and entries[0]["opens"] == "bench"
    assert reply["id"] == entries[0]["id"]
    after = bench(em)
    assert after["bench"]["origin"]["name"] == "phase" and after["bench"]["modified"] is False


def test_save_as_refuses_a_name_another_view_has(em):
    em.plot(plane="z-x", component="Ex", name="TE")
    bench(em, {"dataset": "smatrix"})
    reply = _act(em, action="save_as", name="TE")
    assert "TE" in reply["error"]
    assert len(_calls(em, "EM_save_view")) == 1


def test_save_as_refuses_an_empty_name(em):
    bench(em, {"dataset": "smatrix"})
    assert "name" in _act(em, action="save_as", name="  ")["error"]


def test_rename_posts_a_delete_and_a_save(em):
    handle = em.plot(plane="z-x", component="Ex")
    reply = _act(em, action="rename", id=handle.id, name="TE")
    assert reply.get("error") is None, reply
    assert [c["name"] for c in _calls(em, "EM_delete_saved_view")] == ["z-x Ex abs"]
    assert _calls(em, "EM_save_view")[-1]["view"].name == "TE"
    assert [e["title"] for e in _sidebar(em)] == ["TE"]


def test_remove_posts_the_delete_and_drops_the_entry(em):
    handle = em.plot(plane="z-x", component="Ex")
    reply = _act(em, action="remove", id=handle.id)
    assert reply.get("error") is None, reply
    assert [c["name"] for c in _calls(em, "EM_delete_saved_view")] == ["z-x Ex abs"]
    assert _sidebar(em) == []
    assert em._views.all_fetched()  # close() does not wait for a view no one can open


def test_a_session_only_entry_is_renamed_and_removed_without_posting(em):
    farfield().show(em=em, name="far")
    (entry,) = _sidebar(em)
    _act(em, action="rename", id=entry["id"], name="far 2")
    _act(em, action="remove", id=entry["id"])
    assert _calls(em, "EM_delete_saved_view") == [] and _calls(em, "EM_save_view") == []


def test_saving_tells_the_tab(em):
    server = _server(em)
    q = server.subscribe()
    bench(em, {"dataset": "smatrix"})
    _act(em, action="save_as", name="phase")
    events = []
    while not q.empty():
        events.append(q.get_nowait()[0])
    assert "saved-changed" in events


@pytest.mark.parametrize("headers", [{"X-EMode-Token": None}, {"Origin": "http://evil.example"}])
def test_the_saved_route_refuses_another_page(em, headers):
    status, _ = _http(_server(em), "POST", "/api/saved", {"action": "remove", "id": "p1"}, headers=headers)
    assert status == 403


# ------------------------------------------------------------------ after close()


def test_after_close_a_cached_entry_opens_and_an_uncached_one_says_the_session_ended(em, file_views, monkeypatch):
    monkeypatch.setenv("EMODE_UI_EXIT_WAIT", "0")
    file_views.views = [_recipe("from the file")]
    handle = em.plot(plane="z-x", component="Ex")
    entries = {e["title"]: e["id"] for e in _sidebar(em)}
    server = _server(em)
    em.close()
    server.start()  # the tab stays a viewer
    status, cached = _http(server, "POST", f"/api/views/{handle.id}/data", {"controls": {}, "pixels": [1170, 720]})
    assert status == 200 and "error" not in cached, cached
    reply = bench(em, {"saved": entries["from the file"]})
    assert "session has ended" in reply["error"]
    server.stop()


# ------------------------------------------------------------------ the operating point (Eric, 2026-11-15)


def _with_response(em: Any) -> None:
    trend = _entry("response", "trend", "response", "Operating point", summary="linear, 2 ports")
    base = inventory()
    em.client.answers["EM_plot_inventory"] = lambda request: base.model_copy(
        update={"datasets": [*base.datasets, trend]}
    )
    get = em.client.answers["EM_get"]
    em.client.answers["EM_get"] = lambda request: response() if request.get("key") == "response" else get(request)


def test_the_operating_point_is_listed_and_drawn_on_the_bench(em):
    _with_response(em)
    trend = {d["id"]: d for d in bench(em, {"category": "trend"})["bench"]["datasets"]}
    assert trend["response"]["label"] == "Operating point" and trend["response"]["available"]
    reply = bench(em, {"dataset": "response"})
    assert "error" not in reply, reply
    assert reply["kind"] == "sweep"
    assert [c["key"] for c in _calls(em, "EM_get")][-1] == "response"
    traces = [t for t in reply["figure"]["data"] if t.get("meta") == "trace"]
    assert traces
    menus = {m["name"]: m for m in reply["extra"]["menus"]}
    assert set(menus["result"]["options"]) >= {"output_power", "input_power"}


def test_the_operating_point_of_a_chi2_device_is_power_along_z():
    from emode.nonlinear import Chi2Trajectory

    z = np.linspace(0.0, 1e6, 5)
    pump = np.sqrt(np.linspace(1e-3, 0.9e-3, 5))[None, :].astype(complex)
    signal = np.sqrt(np.linspace(0.0, 0.1e-3, 5))[None, :].astype(complex)
    trajectory = Chi2Trajectory(section="pp", z=z, wavelengths=(1550.0, 775.0), amplitudes=(pump, signal))
    value = response().model_copy(update={"trajectories": (trajectory,), "linear": False})
    trend = ui.response_trend(value)
    assert trend["key"] == "z"
    np.testing.assert_allclose(trend["values"], z)
    np.testing.assert_allclose(trend["power at 775 nm"], np.linspace(0.0, 0.1e-3, 5))
    np.testing.assert_allclose(trend["power at 1550 nm"], np.linspace(1e-3, 0.9e-3, 5))


# ------------------------------------------------------------------ the tab's script

BENCH_REPLY = """
var reply = {id: "bench", kind: "plot", title: "x-y Ey abs", figure: {data: [], layout: {}}, aspect: 0.7,
  menu: {data: ["Ex", "Ey"], selected: {data: "Ey"}}, controls: {}, extra: {},
  bench: {categories: ["field", "trend"], category: "field", dataset: "profile:wg",
    datasets: [{id: "profile:wg", label: "wg", available: true, reason: null, summary: ""}],
    origin: ORIGIN, modified: MODIFIED}};
state.current = "bench";
narrowQuery.matches = NARROW;
buildControls(reply);
var box = document.getElementById("controls");
var fold = box.children.filter(function (c) { return c.tagName === "DETAILS" && c.className === "settings"; })[0];
var save = box.querySelectorAll("button").filter(function (b) { return b.id === "save-view"; })[0];
"""


def _bench_js(test_js: str, origin: str = "null", modified: str = "false", narrow: str = "true") -> Any:
    prelude = BENCH_REPLY.replace("ORIGIN", origin).replace("MODIFIED", modified).replace("NARROW", narrow)
    return run_app(prelude + test_js)


@needs_jsc
def test_the_save_button_sits_beside_render_and_export_outside_the_fold():
    got = _bench_js(
        """
return {save: !!save, text: save ? save.textContent : "", inFold: fold ? __text(fold).indexOf("Save view") >= 0 : null,
  order: box.children.map(function (c) { return c.className || c.tagName; })};
"""
    )
    assert got["save"] and got["text"] == "Save view"
    assert got["inFold"] is False
    order = got["order"]
    assert order.index("save-view-bar") < order.index("export")


@needs_jsc
def test_the_bar_says_modified_from_the_saved_view():
    got = _bench_js(
        "return __text(box.querySelector('.modified'));",
        origin='{id: "p1", name: "TE at 1550"}',
        modified="true",
    )
    assert got == "modified from TE at 1550"


@needs_jsc
def test_the_bar_names_an_unmodified_saved_view_without_the_mark():
    got = _bench_js(
        "var m = box.querySelector('.modified'); return m ? m.hidden : null;",
        origin='{id: "p1", name: "TE at 1550"}',
    )
    assert got is True


@needs_jsc
def test_the_save_sheet_offers_save_only_when_the_bench_came_from_a_saved_view():
    fresh = _bench_js(
        """
save.click();
var sheet = document.getElementById("save-sheet");
return {hidden: sheet.hidden, save: document.getElementById("sheet-save").hidden,
  saveAs: document.getElementById("sheet-save-as").hidden};
"""
    )
    assert fresh == {"hidden": False, "save": True, "saveAs": False}
    loaded = _bench_js(
        """
save.click();
return {save: document.getElementById("sheet-save").hidden, name: document.getElementById("sheet-name").value};
""",
        origin='{id: "p1", name: "TE at 1550"}',
        modified="true",
    )
    assert loaded == {"save": False, "name": "TE at 1550"}


@needs_jsc
def test_save_as_posts_the_name_and_save_posts_to_the_origin():
    got = _bench_js(
        """
save.click();
document.getElementById("sheet-name").value = "phase";
document.getElementById("sheet-save-as").click();
document.getElementById("sheet-save").click();
return __posts.filter(function (p) { return p[0] === "api/saved"; }).map(function (p) { return p[1]; });
""",
        origin='{id: "p1", name: "TE at 1550"}',
    )
    assert got == [{"action": "save_as", "name": "phase"}, {"action": "save"}]


@needs_jsc
def test_a_script_view_offers_edit_in_workbench():
    got = run_app(
        """
var reply = {id: "p1", kind: "plot", title: "z-x Ex abs", figure: {data: [], layout: {}}, aspect: 0.7,
  menu: {selected: {}}, controls: {}, extra: {}};
state.current = "p1";
state.views = [{id: "bench", kind: "bench"}, {id: "p1", kind: "plot", script: true, opens: "view", title: "z-x Ex abs"}];
narrowQuery.matches = true;
buildControls(reply);
var box = document.getElementById("controls");
var edit = box.querySelectorAll("button").filter(function (b) { return b.id === "edit-in-workbench"; })[0];
edit.click();
return {text: edit.textContent, posts: __posts};
"""
    )
    assert got["text"] == "Edit in workbench"
    posted = [body for path, body in got["posts"] if path == "api/views/bench/data"]
    assert posted and posted[-1]["controls"] == {"saved": "p1"}


@needs_jsc
def test_a_saved_entry_route_opens_its_recipe_on_the_bench():
    got = run_app(
        """
state.views = [{id: "bench", kind: "bench"}, {id: "v1", kind: "plot", opens: "bench", title: "TE"}];
location.hash = "#/plots/v1";
route();
return __posts;
"""
    )
    posted = [body for path, body in got if path.startswith("api/views/")]
    assert posted and posted[-1]["controls"] == {"saved": "v1"}
    assert [path for path, _ in got if path.startswith("api/views/")][-1] == "api/views/bench/data"


@needs_jsc
def test_the_sidebar_entry_menu_renames_and_removes():
    got = run_app(
        """
state.current = "bench";
var li = {id: "p1", kind: "plot", script: true, opens: "view", title: "z-x Ex abs", session_only: false};
var row = sidebarEntry(li);
var menu = row.querySelectorAll("button").filter(function (b) { return b.className === "entry-menu"; })[0];
menu.dispatch("click", {stopPropagation: function () {}, preventDefault: function () {}});
var actions = row.querySelectorAll("button").map(function (b) { return b.textContent; });
var remove = row.querySelectorAll("button").filter(function (b) { return b.textContent === "Remove"; })[0];
remove.dispatch("click", {stopPropagation: function () {}, preventDefault: function () {}});
return {actions: actions, posts: __posts};
"""
    )
    assert {"Edit in workbench", "Rename", "Remove"} <= set(got["actions"])
    posted = [body for path, body in got["posts"] if path == "api/saved"]
    assert posted == [{"action": "remove", "id": "p1"}]


@needs_jsc
def test_the_tab_reloads_the_list_when_the_saved_views_change():
    got = run_app(
        """
onEvent("saved-changed", {});
return __posts.map(function (p) { return p[0]; });
"""
    )
    assert "api/views" in got


# ------------------------------------------------------------------ in a real browser at phone width


_SAVE_PROBE = """(function () {
  var c = document.getElementById("controls");
  var d = c.querySelector("details.settings");
  var b = document.getElementById("save-view");
  var r = b ? b.getBoundingClientRect() : null;
  return JSON.stringify({found: !!b, inFold: b && d ? d.contains(b) : null, foldOpen: d ? d.open : null,
    width: r ? r.width : 0, height: r ? r.height : 0, right: r ? r.right : 0, top: r ? r.top : 0,
    viewport: window.innerWidth, viewportHeight: window.innerHeight,
    shown: b ? getComputedStyle(b).visibility === "visible" && getComputedStyle(b).display !== "none" : false});
})()"""


def test_save_is_visible_at_phone_width_outside_the_settings_fold(em):
    from test_workbench import _bench_page

    page = _bench_page(em, (390, 844))
    try:
        got = json.loads(page.eval(_SAVE_PROBE))
        page.eval("document.getElementById('save-view').click()")
        sheet = json.loads(
            page.eval(
                """JSON.stringify((function () { var s = document.getElementById("save-sheet");
                var r = s.getBoundingClientRect(); return {hidden: s.hidden, bottom: r.bottom, width: r.width,
                top: r.top}; })())"""
            )
        )
    finally:
        page.close()
    assert got["found"] and got["inFold"] is False and got["foldOpen"] is False
    assert got["shown"] and got["width"] >= 44 and got["height"] >= 24, got
    assert 0 <= got["top"] < got["viewportHeight"] and got["right"] <= got["viewport"], got
    assert sheet["hidden"] is False and sheet["width"] <= 390 and sheet["bottom"] <= 844 + 1, sheet


# ------------------------------------------------------------------ a dev EMode before the release bumps its version


DEV_1_0_8 = {"emode": "1.0.8", "protocol": 1, "plot": "view", "views": VIEWS}
RELEASE_1_0_8 = {"emode": "1.0.8", "protocol": 1}


def test_a_dev_emode_that_lists_saved_views_is_sent_them_whatever_its_version(em):
    """Every dev EMode says 1.0.8 until the release: it reads a SavedView once it lists 'saved'."""
    em.client.answers["EM_version"] = DEV_1_0_8
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        em.plot(plane="z-x", component="Ex", name="TE")
    (posted,) = _calls(em, "EM_save_view")
    assert posted["view"].name == "TE"
    assert [v.name for v in em.saved_views()] == ["TE"]


def test_a_release_emode_without_saved_views_is_refused_a_saved_view_clearly(em):
    em.client.answers["EM_version"] = RELEASE_1_0_8
    with pytest.raises(ArgumentError, match="SavedView needs EMode 1.0.9"):
        em.call("EM_save_view", view=_recipe("TE"))
    assert _calls(em, "EM_save_view") == []


def test_the_capability_stands_in_for_the_release_only_for_its_own_type():
    from emode.compat import check_emode_version

    needs = {"view": [("SavedView", "1.0.9")]}
    check_emode_version(needs, "1.0.8", "EM_save_view", capabilities={"saved"})
    with pytest.raises(ArgumentError):
        check_emode_version(needs, "1.0.8", "EM_save_view", capabilities={"inventory"})
    with pytest.raises(ArgumentError):
        check_emode_version({"bounds": [("Bounds.num_pml_layers", "1.0.9")]}, "1.0.8", "EM_settings", {"saved"})
