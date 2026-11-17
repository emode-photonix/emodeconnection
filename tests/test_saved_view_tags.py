"""Tags on saved views, the sidebar grouped by first tag with a filter, and the session page's Holds row (GUI-M4d, emode#1704).

A saved view's ``tags`` are optional labels. The Save sheet and the Rename
sheet take them, comma separated, and the sidebar groups the saved views by
their first tag (untagged views last) under a filter box that keeps the views
whose name or a tag contains the text typed. The tags travel in the
``SavedView`` recipe, so the simulation file keeps them.

The session page gains a "Holds" row, drawn from the plot inventory
(``PlotInventory``) rather than from ``SessionInfo``, which is unchanged
(decision 2 of the plot workbench note, emode#1698).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from emode.plotting import KindEntry, PlotInventory, SavedView
from emode.ui import session as ui
from emode.ui.server import holds_text

from js_harness import needs_jsc, run_app
from plot_fixtures import session_info
from test_saved_views import _act, _recipe, _sidebar
from test_saved_views import em as _saved_em  # noqa: F401  (fixture)
from test_saved_views import file_views  # noqa: F401  (fixture)
from test_workbench import _entry, _http, _server, bench, inventory
from test_workbench import desktop  # noqa: F401  (autouse fixture)
from test_workbench import em as _workbench_em  # noqa: F401  (fixture)


@pytest.fixture
def em(_saved_em):  # noqa: F811
    return _saved_em


def _posted(em: Any) -> List[SavedView]:
    return [c["view"] for c in em.client.calls("EM_save_view")]


# ------------------------------------------------------------------ tags on the Save and Rename sheets


def test_save_as_with_tags_posts_them_and_the_sidebar_lists_them(em):
    bench(em, {"dataset": "profile:wg"})
    reply = _act(em, action="save_as", name="TE at 1550", tags=["modes", "C band"])
    posted = _posted(em)[-1]
    assert posted.name == "TE at 1550" and posted.tags == ["modes", "C band"]
    entry = next(v for v in reply["views"] if v["id"] == reply["id"])
    assert entry["tags"] == ["modes", "C band"]
    assert next(v for v in _sidebar(em) if v["title"] == "TE at 1550")["tags"] == ["modes", "C band"]


def test_tags_typed_as_text_are_split_trimmed_and_deduplicated(em):
    bench(em, {"dataset": "profile:wg"})
    _act(em, action="save_as", name="TE", tags=" modes, C band ,,modes ")
    assert _posted(em)[-1].tags == ["modes", "C band"]


def test_save_keeps_the_saved_views_tags_unless_the_sheet_changes_them(em):
    bench(em, {"dataset": "profile:wg"})
    _act(em, action="save_as", name="TE", tags=["modes"])
    bench(em, {"data": "Ex"})
    _act(em, action="save")
    assert _posted(em)[-1].tags == ["modes"]
    _act(em, action="save", tags=["modes", "draft"])
    assert _posted(em)[-1].tags == ["modes", "draft"]


def test_the_benchs_origin_carries_its_tags_for_the_sheet(em):
    bench(em, {"dataset": "profile:wg"})
    _act(em, action="save_as", name="TE", tags=["modes"])
    reply = bench(em, {})
    assert reply["bench"]["origin"]["tags"] == ["modes"]


def test_rename_can_retag_and_keeping_the_name_is_not_a_clash(em):
    bench(em, {"dataset": "profile:wg"})
    entry_id = _act(em, action="save_as", name="TE")["id"]
    _act(em, action="rename", id=entry_id, name="TE", tags=["final"])
    assert _posted(em)[-1].name == "TE" and _posted(em)[-1].tags == ["final"]
    _act(em, action="rename", id=entry_id, name="TE 1550")
    # a rename that does not name tags keeps them
    assert _posted(em)[-1].name == "TE 1550" and _posted(em)[-1].tags == ["final"]
    assert [v["tags"] for v in _sidebar(em) if v["id"] == entry_id] == [["final"]]


def test_the_files_tagged_views_list_their_tags(em, file_views):
    file_views.views = [_recipe("TE", tags=["modes"]), _recipe("TM")]
    listed = {v["title"]: v["tags"] for v in _sidebar(em)}
    assert listed == {"TE": ["modes"], "TM": []}


def test_a_new_recipe_under_a_tagged_name_keeps_its_tags(em, file_views):
    file_views.views = [_recipe("TE", tags=["modes"])]
    em.plot(name="TE", plane="z-x")  # replaces the recipe of that name: posted, and still tagged
    assert _posted(em)[-1].name == "TE" and _posted(em)[-1].tags == ["modes"]


# ------------------------------------------------------------------ the session page's Holds row


def test_holds_names_what_the_inventory_lists():
    text = holds_text(inventory())
    assert text == "2 profiles, an EME device (solved), a sweep over width (3 points), a monitor"


def test_holds_reads_an_unsolved_device_a_gds_section_a_pulse_and_an_operating_point():
    value = PlotInventory(
        kinds=[],
        profiles=["FDM"],
        device="preview",
        gds=True,
        pulse_response=True,
        datasets=[
            _entry("response", "trend", "response", "Operating point"),
            _entry("monitor:a", "field", "monitor", "monitor a"),
            _entry("monitor:b", "field", "monitor", "monitor b"),
        ],
    )
    assert holds_text(value) == (
        "1 profile, an EME device (not solved), a GDS section, a pulse response, an operating point, 2 monitors"
    )


def test_holds_of_an_empty_session_says_so():
    assert holds_text(PlotInventory(kinds=[KindEntry(kind="material", available=True)])) == "nothing to plot yet"


def test_the_session_page_has_a_holds_row_and_session_info_is_unchanged(em):
    em.client.answers["EM_session_info"] = lambda request: session_info()
    status, page = _http(_server(em), "GET", "/api/session")
    assert status == 200
    assert page["holds"] == holds_text(inventory())
    assert "holds" not in page["emode"]
    assert set(page["emode"]) == set(session_info().model_dump(mode="json"))


def test_an_emode_without_the_inventory_leaves_the_holds_row_out(em):
    em.client.answers["EM_session_info"] = lambda request: session_info()
    em.client.answers["EM_version"]["views"] = [v for v in em.client.answers["EM_version"]["views"] if v != "inventory"]
    status, page = _http(_server(em), "GET", "/api/session")
    assert status == 200
    assert page["holds"] is None and page["emode"]["version"] == "1.0.9"
    assert not em.client.calls("EM_plot_inventory")


# ------------------------------------------------------------------ the tab's script

SIDEBAR = """
state.views = [
  {id: "bench", kind: "bench", title: "Workbench"},
  {id: "p1", kind: "plot", title: "z-x Ex abs", created: 1, script: true, tags: []},
  {id: "v2", kind: "plot", title: "TE at 1550", created: 2, tags: ["modes", "C band"]},
  {id: "v3", kind: "smatrix", title: "S-matrix left", created: 3, tags: ["ports"]},
  {id: "v4", kind: "plot", title: "TM at 1550", created: 4, tags: ["modes"]}
];
"""


def _sidebar_js(test_js: str) -> Any:
    return run_app(
        SIDEBAR
        + """
function rowsOf() {
  return document.getElementById("views").children.map(function (li) {
    return (li.className.indexOf("group") >= 0 ? "# " : "") + (li.className.indexOf("empty") >= 0 ? "! " : "") +
      (li.dataset.id || "") + (li.dataset.id ? "" : __text(li));
  });
}
"""
        + test_js
    )


@needs_jsc
def test_the_sidebar_groups_saved_views_by_first_tag_untagged_last():
    got = _sidebar_js("renderList(); return rowsOf();")
    assert got == ["bench", "# modes", "v4", "v2", "# ports", "v3", "# Untagged", "p1"]


@needs_jsc
def test_without_tags_the_sidebar_keeps_one_saved_views_heading():
    got = run_app(
        """
state.views = [{id: "bench", kind: "bench"}, {id: "p1", kind: "plot", title: "a", created: 1},
  {id: "p2", kind: "plot", title: "b", created: 2, tags: []}];
renderList();
return document.getElementById("views").children.map(function (li) { return li.dataset.id || __text(li); });
"""
    )
    assert got == ["bench", "Saved views", "p2", "p1"]


@needs_jsc
def test_an_entry_shows_its_tags():
    got = _sidebar_js(
        """
renderList();
var li = document.getElementById("views").children.filter(function (c) { return c.dataset.id === "v2"; })[0];
var tags = li.querySelector(".tags");
return tags ? __text(tags) : null;
"""
    )
    assert got == "modes, C band"


@needs_jsc
def test_the_filter_keeps_views_whose_name_or_tag_matches():
    got = _sidebar_js(
        """
var box = document.getElementById("views-filter");
box.value = "C BAND";
box.dispatch("input");
var byTag = rowsOf();
box.value = "s-matrix";
box.dispatch("input");
var byName = rowsOf();
box.value = "nothing like it";
box.dispatch("input");
var none = rowsOf();
box.value = "";
box.dispatch("input");
return {byTag: byTag, byName: byName, none: none, all: rowsOf().length};
"""
    )
    assert got["byTag"] == ["bench", "# modes", "v2"]
    assert got["byName"] == ["bench", "# ports", "v3"]
    assert got["none"] == ["bench", "! No saved view matches “nothing like it”"]
    assert got["all"] == 8


@needs_jsc
def test_the_save_sheet_fills_the_origins_tags_and_save_as_posts_them():
    from test_saved_views import _bench_js

    got = _bench_js(
        """
save.click();
var tags = document.getElementById("sheet-tags");
var before = tags.value;
tags.value = "modes, draft";
document.getElementById("sheet-name").value = "phase";
document.getElementById("sheet-save-as").click();
document.getElementById("sheet-save").click();
return {before: before, posts: __posts.filter(function (p) { return p[0] === "api/saved"; }).map(function (p) { return p[1]; })};
""",
        origin='{id: "p1", name: "TE at 1550", tags: ["modes"]}',
    )
    assert got["before"] == "modes"
    assert got["posts"] == [
        {"action": "save_as", "name": "phase", "tags": ["modes", "draft"]},
        {"action": "save", "tags": ["modes", "draft"]},
    ]


@needs_jsc
def test_the_rename_sheet_posts_the_tags():
    got = _sidebar_js(
        """
renderList();
openSaveSheet({mode: "rename", id: "v2", name: "TE at 1550", tags: ["modes", "C band"]});
var tags = document.getElementById("sheet-tags");
var before = tags.value;
tags.value = "modes";
document.getElementById("sheet-rename").click();
return {before: before, hidden: tags.hidden,
  posts: __posts.filter(function (p) { return p[0] === "api/saved"; }).map(function (p) { return p[1]; })};
"""
    )
    assert got["before"] == "modes, C band" and got["hidden"] is False
    assert got["posts"] == [{"action": "rename", "id": "v2", "name": "TE at 1550", "tags": ["modes"]}]


@needs_jsc
def test_the_session_page_draws_the_holds_row():
    got = run_app(
        """
showSession({emode: {version: "1.0.9"}, holds: "2 profiles, an EME device (solved)", client: {version: "1.2"},
  session_ended: false});
var rows = document.getElementById("session-table").children.map(function (tr) {
  return [__text(tr.children[0]), __text(tr.children[1])];
});
return rows;
"""
    )
    assert ["Holds", "2 profiles, an EME device (solved)"] in got
    assert got.index(["Holds", "2 profiles, an EME device (solved)"]) == 1
