"""The Slice menu stays on a profile's x-y plot, switches to the device's planes, and never offers the schematic.

Eric, 2026-11-15 (emode#1728): with Profile or FDM chosen on the x-y plane the
Slice dropdown disappeared. EMode now sends the Slice menu on every field plot,
with z-x and z-y when the session has an EME device. Choosing z-x or z-y from a
profile's x-y plot posts the plane with no profile, so EMode draws the device,
and every plane change clears the position, which means a z on the x-y plane
and a transverse position on a z panel. The schematic is the View menu's, so
the Slice menu drops it even from an EMode that still sends it.
"""

from __future__ import annotations

from pathlib import Path

from js_harness import needs_jsc, run_app


def _plot(menu: str) -> str:
    return (
        """
var box = document.createElement("div");
var changes = [];
change = function (c) { changes.push(c); };
buildPlotControls(box, {menu: """
        + menu
        + """, controls: {}});
var slice = box.querySelectorAll("select").filter(function (s) {
  return s.parentNode && __text(s.parentNode).indexOf("Slice") === 0; })[0];
function pick(v) { slice.value = v; slice.dispatch("change", {target: slice}); return changes; }
function options() { return slice ? slice.children.map(function (o) { return o.value; }) : null; }
"""
    )


#: A profile's x-y plot in a session with an EME device.
PROFILE = _plot(
    '{profiles: ["EME", "FDM", "single"], planes: ["x-y", "z-x", "z-y"], modes: ["0, TE"],'
    ' selected: {profile: "single", plane: "x-y", mode: "0, TE"}}'
)
#: The FDM profile's x-y plot with no EME: the x-y plane alone.
FDM_ONLY = _plot('{profiles: ["FDM"], planes: ["x-y"], selected: {profile: "FDM", plane: "x-y"}}')
#: A z-x panel of the device, from an EMode that still lists the schematic among the planes.
DEVICE_ZX = _plot(
    '{planes: ["x-y", "z-x", "z-y", "schematic"], views: ["layout", "simulation", "schematic"],'
    ' slider: {label: "y:", min: 0, max: 2, unit: "μm", value: "1", options: []},'
    ' selected: {plane: "z-x", view: "layout"}}'
)


@needs_jsc
def test_a_profiles_x_y_plot_shows_the_slice_menu_with_the_devices_planes():
    got = run_app(PROFILE + "return {options: options(), shown: slice.value};")
    assert got == {"options": ["x-y", "z-x", "z-y"], "shown": "x-y"}


@needs_jsc
def test_the_fdm_profile_without_eme_still_shows_its_x_y_slice():
    got = run_app(FDM_ONLY + "return options();")
    assert got == ["x-y"]


@needs_jsc
def test_a_z_plane_from_a_profiles_plot_draws_the_device():
    got = run_app(PROFILE + 'return pick("z-x");')
    assert got == [{"plane": "z-x", "profile": None, "slice_position": None}]


@needs_jsc
def test_x_y_from_a_z_panel_clears_the_transverse_position():
    got = run_app(DEVICE_ZX + 'return pick("x-y");')
    assert got == [{"plane": "x-y", "slice_position": None}]


@needs_jsc
def test_the_slice_menu_never_offers_the_schematic():
    got = run_app(
        DEVICE_ZX
        + """
var view = box.querySelectorAll("select").filter(function (s) {
  return s.parentNode && __text(s.parentNode).indexOf("View") === 0; })[0];
return {slice: options(), view: view.children.map(function (o) { return o.value; })};
"""
    )
    assert got["slice"] == ["x-y", "z-x", "z-y"]
    assert "schematic" in got["view"]


def test_the_slice_menu_filters_the_schematic_in_the_source():
    # the static form, for the platforms without osascript
    js = (Path(__file__).resolve().parents[1] / "emode" / "ui" / "static" / "app.js").read_text()
    start = js.index('select("Slice"')
    line = js[js.rindex("\n", 0, start) : js.index("\n", start)]
    assert "schematic" in line
