"""The x-y plot's Profile menu names EME, and choosing it never posts a position for another axis (#179).

Reported 2026-11-15: on a mode plot made by ``plot()`` before the device
existed, choosing EME in the z box failed with

    ArgumentError: the argument: (slice_position) to function: (plot) had error:
    "slice_position 0 um is outside the y range of this plot, 0.0125 to 1.9875 um"

The tab posted ``slice_position=0`` for EME. The plot's arguments named no plane,
so EMode drew its default z-x plane and read the 0 as a y position. EMode now
sends no z box on the x-y plane and lists EME in the Profile menu, and the
tab posts EME as ``profile='EME'`` on the x-y plane, with no position.
"""

from __future__ import annotations

import http.client
import json
from typing import Any, Dict, List, Tuple

from emode.types import ArgumentError
from emode.ui.figure import PlotOptions
from emode.ui.server import UIServer
from emode.ui.views import ViewRegistry

from js_harness import needs_jsc, run_app
from plot_fixtures import xy_view, zx_view

#: A profile's x-y plot in a session that has EME sections: EME first, then the profiles.
PROFILE_PLOT = """
var box = document.createElement("div");
var changes = [];
change = function (c) { changes.push(c); };
buildPlotControls(box, {menu: {profiles: ["EME", "FDM", "single", "pair"], modes: ["0, TE"],
  selected: {profile: "single", mode: "0, TE"}}, controls: {}});
var profile = box.querySelectorAll("select").filter(function (s) {
  return s.parentNode && __text(s.parentNode).indexOf("Profile") === 0; })[0];
"""

#: The device's x-y plot: Profile on EME, and the z of the cut as a plain position.
DEVICE_XY = """
var box = document.createElement("div");
var changes = [];
change = function (c) { changes.push(c); };
buildPlotControls(box, {menu: {profiles: ["EME", "FDM", "single"], planes: ["x-y", "z-x", "z-y"],
  position: {label: "z:", min: 0, max: 30, unit: "μm", value: "7.5 μm", options: []},
  selected: {profile: "EME", plane: "x-y"}}, controls: {}});
var range = box.querySelector("input[type=range]");
var labels = box.querySelectorAll("label").map(function (l) { return __text(l); });
"""


def _the_y_range_rule(arguments: Dict[str, Any]) -> None:
    """EM_plot's reading of `slice_position`: on a z-x plane (its default) it is a y in um."""
    plane = arguments.get("plane") or "z-x"
    position = arguments.get("slice_position")
    if plane != "x-y" and isinstance(position, (int, float)) and not 0.0125 <= position <= 1.9875:
        raise ArgumentError(
            f"slice_position {position:g} um is outside the y range of this plot, 0.0125 to 1.9875 um",
            "plot",
            "slice_position",
        )


@needs_jsc
def test_the_profile_menu_lists_eme_and_the_profiles():
    got = run_app(
        PROFILE_PLOT + "return {options: profile.children.map(function (o) { return o.value; }), shown: profile.value};"
    )
    assert got["options"] == ["EME", "FDM", "single", "pair"]
    assert got["shown"] == "single"


@needs_jsc
def test_choosing_eme_posts_the_device_on_the_x_y_plane_and_no_position():
    got = run_app(
        PROFILE_PLOT + 'profile.value = "EME"; profile.dispatch("change", {target: profile}); return changes;'
    )
    assert got == [{"profile": "EME", "plane": "x-y", "slice_position": None}]


@needs_jsc
def test_choosing_a_profile_clears_the_position():
    got = run_app(
        PROFILE_PLOT + 'profile.value = "pair"; profile.dispatch("change", {target: profile}); return changes;'
    )
    assert got == [{"profile": "pair", "slice_position": None}]


@needs_jsc
def test_the_device_x_y_plot_moves_its_cut_with_z_position_only():
    got = run_app(
        DEVICE_XY
        + 'range.value = "29"; range.dispatch("change", {target: range}); return {changes: changes, labels: labels};'
    )
    assert got["changes"] == [{"z_position": 29, "slice_position": None}]
    assert not any(label.startswith("Cut") for label in got["labels"]), (
        "the x-y plot is the cut: its z is not labeled as a cut of it"
    )


@needs_jsc
def test_a_z_panels_cut_still_posts_z_position_alone():
    got = run_app(
        """
var box = document.createElement("div");
var changes = [];
change = function (c) { changes.push(c); };
buildPlotControls(box, {menu: {position: {label: "z:", min: 0, max: 30, unit: "μm", value: "7.5 μm", options: []},
  selected: {plane: "z-x"}}, controls: {}});
var range = box.querySelector("input[type=range]");
range.value = "29"; range.dispatch("change", {target: range});
return changes;
"""
    )
    assert got == [{"z_position": 29}]


# ------------------------------------------------------------------ the exact error, end to end


def _server() -> Tuple[UIServer, List[Dict[str, Any]]]:
    registry = ViewRegistry()
    # the mode plot ``plot()`` made before the device existed: no plane among its arguments
    registry.add(xy_view(), {}, PlotOptions(), (1200, 750))
    fetched: List[Dict[str, Any]] = []

    def fetch(view: Any, arguments: Dict[str, Any], pixels: Tuple[int, int]) -> Any:
        fetched.append(dict(arguments))
        _the_y_range_rule(arguments)
        return xy_view() if arguments.get("plane") == "x-y" else zx_view()

    return UIServer(registry, fetch).start(), fetched


def _post(server: UIServer, controls: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    body = json.dumps({"controls": controls, "pixels": [1170, 720]}).encode()
    headers = {"Host": f"127.0.0.1:{server.port}", "X-EMode-Token": server.token, "Content-Type": "application/json"}
    conn.request("POST", "/api/views/p1/data", body=body, headers=headers)
    response = conn.getresponse()
    data = json.loads(response.read() or b"{}")
    conn.close()
    return response.status, data


def test_the_stand_in_reads_the_old_post_as_emode_did():
    server, _ = _server()
    try:
        _, reply = _post(server, {"slice_position": 0, "z_position": 0, "profile": None})
    finally:
        server.stop()
    assert "outside the y range of this plot" in json.dumps(reply)


@needs_jsc
def test_choosing_eme_on_a_mode_plot_draws_the_device_x_y_plane():
    (post,) = run_app(
        PROFILE_PLOT + 'profile.value = "EME"; profile.dispatch("change", {target: profile}); return changes;'
    )
    server, fetched = _server()
    try:
        status, reply = _post(server, post)
    finally:
        server.stop()
    assert status == 200 and "error" not in reply, reply
    assert fetched[-1]["plane"] == "x-y" and fetched[-1]["profile"] == "EME"
    assert fetched[-1].get("slice_position") is None


def test_no_profile_choice_sends_a_number_for_slice_position():
    # the static form of the rule, for the platforms without osascript
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "emode" / "ui" / "static" / "app.js").read_text()
    start = js.index('select("Profile"')
    handler = js[start : js.index("box.append(", start)]
    assert "slice_position: 0" not in handler and "slice_position: null" in handler
