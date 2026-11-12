"""The GDS layer view: its wire type, its figure, both renderers, the layer panel and its route.

``plot(view='gds')`` asks EMode for a ``GDSView``: one ``GDSLayer`` per GDS
(layer, datatype) with its polygons in nm, placed on the chip as the device
places them, and the simulation's shapes for the overlay. The client draws one
trace per layer (polygons separated by gaps) colored from the line palette,
and the tab's layer panel changes each layer's visibility, color, opacity and
hatch in the browser, kept for the session so a saved file draws the same.
"""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from emode import colormaps, compat
from emode.emodeclient import obj_hook
from emode.gds import GDSLayer, GDSView, read_lyp
from emode.types import ArgumentError, reconstruct, serialize, wire_dump
from emode.ui.figure import PlotOptions, compose
from emode.ui.gds import HATCHES, GDSPanel, layer_rows
from emode.ui.gds import extra as extra_of
from emode.ui.renderers.plotly import layer_style, to_plotly
from emode.ui.views import ViewRegistry, kind_of, title_of

from gds_fixtures import LYP, gds_view
from plot_standin import VERSION_1_0_9, session

mpl = pytest.importorskip("matplotlib")
from emode.ui.renderers import mpl as mpl_renderer  # noqa: E402


def _round_trip(value):
    payload = json.dumps(serialize(wire_dump(value), ndarray=True))
    return reconstruct(json.loads(payload, object_hook=obj_hook))


def _decode(values):
    if isinstance(values, dict):
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=np.dtype(values["dtype"]))
    return np.array([np.nan if v is None else v for v in values], float)


def _layer_traces(figure):
    return [t for t in figure["data"] if str(t.get("meta", "")).startswith("layer:")]


# ------------------------------------------------------------------ the wire


def test_the_view_round_trips_the_wire_with_its_layers_and_polygons():
    view = gds_view()
    back = _round_trip(view)
    assert isinstance(back, GDSView)
    assert [layer.key for layer in back.layers] == ["1/0", "2/0", "2/5"]
    assert [layer.polygon_count for layer in back.layers] == [2, 1, 3]
    assert back.layers[0].name == "core"
    np.testing.assert_allclose(back.layers[0].polygons()[1], view.layers[0].polygons()[1])
    assert back.shapes[0].name == "core"
    assert back.pixel_nm == pytest.approx(7.5)


def test_the_gds_types_are_versioned():
    for name in ("GDSLayer", "GDSView"):
        assert compat.TYPE_SINCE[name] == "1.0.9"


def test_a_layer_refuses_offsets_that_do_not_cover_its_vertices():
    with pytest.raises(ValueError):
        GDSLayer(layer=1, datatype=0, xy=np.zeros((4, 2)), offsets=[0, 3])


# ------------------------------------------------------------------ the figure


def test_one_layer_spec_per_layer_in_the_line_palettes_colors():
    spec = compose(GDSPanel.from_view(gds_view()))
    layers = [s for s in spec.layers if s.key != "shapes"]
    assert [s.key for s in layers] == ["1/0", "2/0", "2/5"]
    palette = colormaps.line_palette()
    assert [s.color for s in layers] == list(palette[:3])
    assert [len(s.polygons) for s in layers] == [2, 1, 3]
    # in μm, as the layout view shows the chip
    assert layers[0].polygons[0][:, 0].max() == pytest.approx(4.0)
    assert spec.axes.equal_aspect
    assert spec.axes.xlabel == "z (μm)"


def test_the_shapes_overlay_is_off_by_default_and_on_when_asked():
    off = compose(GDSPanel.from_view(gds_view()))
    shapes = [s for s in off.layers if s.key == "shapes"]
    assert len(shapes) == 1 and not shapes[0].visible and not shapes[0].filled
    on = compose(GDSPanel.from_view(gds_view()).with_changes({"shapes": True}))
    assert [s for s in on.layers if s.key == "shapes"][0].visible


# ------------------------------------------------------------------ the orientation


def _orientations(spec):
    """(Plotly's y range runs low to high, matplotlib's y axis is inverted, the shapes' collection's axes)."""
    layout = to_plotly(spec)["layout"]
    fig = mpl_renderer.to_figure(spec)
    ax = fig.axes[0]
    shapes = [c for c in ax.collections if c.get_gid() == "layer:shapes"]
    return layout["yaxis"]["range"][0] < layout["yaxis"]["range"][1], ax.yaxis_inverted(), shapes[0].axes is ax


def test_the_default_orientation_is_the_gds_files_with_y_up_as_klayout_draws_it():
    panel = GDSPanel.from_view(gds_view())
    assert panel.orientation == "gds"
    spec = compose(panel)
    assert not spec.axes.invert_y
    # the vertices are the file's (z is the file's x, x its y), drawn y up in both renderers
    marker = [s for s in spec.layers if s.key == "2/5"][0].polygons[0]
    assert marker[:, 1].min() == pytest.approx(1.5)
    assert _orientations(spec) == (True, False, True)


def test_the_toggle_gives_the_layout_views_axes_with_the_shapes_overlay_following():
    panel = GDSPanel.from_view(gds_view()).with_changes({"orientation": "layout", "shapes": True})
    spec = compose(panel)
    assert spec.axes.invert_y
    shapes = [s for s in spec.layers if s.key == "shapes"][0]
    assert shapes.visible
    # the overlay is drawn on the same axes as the layers, so it turns with them
    assert _orientations(spec) == (False, True, True)
    back = panel.with_changes({"orientation": "gds"})
    assert _orientations(compose(back)) == (True, False, True)


def test_the_orientation_is_part_of_the_panels_state_and_checked():
    panel = GDSPanel.from_view(gds_view()).with_changes({"orientation": "layout"})
    assert panel.state()["orientation"] == "layout"
    assert extra_of(panel)["orientation"] == "layout"
    with pytest.raises(ArgumentError) as err:
        panel.with_changes({"orientation": "sideways"})
    assert err.value.argument == "orientation"


# ------------------------------------------------------------------ the layer panel's state


def test_a_layers_style_changes_and_the_rest_stay():
    panel = GDSPanel.from_view(gds_view())
    changed = panel.with_changes(
        {"layers": {"2/0": {"visible": False, "color": "#ff0000", "opacity": 0.2, "hatch": "x"}}}
    )
    style = changed.style("2/0")
    assert (style.visible, style.color, style.opacity, style.hatch) == (False, "#ff0000", 0.2, "x")
    assert changed.style("1/0") == panel.style("1/0")
    assert panel.style("2/0").visible  # the panel itself is not changed


def test_solo_shows_one_layer_and_show_all_shows_every_one():
    panel = GDSPanel.from_view(gds_view()).with_changes({"solo": "2/5"})
    assert [panel.style(k).visible for k in ("1/0", "2/0", "2/5")] == [False, False, True]
    back = panel.with_changes({"show_all": True})
    assert all(back.style(k).visible for k in ("1/0", "2/0", "2/5"))


@pytest.mark.parametrize(
    "change, parameter",
    [
        ({"layers": {"9/9": {"visible": False}}}, "layers"),
        ({"layers": {"1/0": {"hatch": "*"}}}, "hatch"),
        ({"layers": {"1/0": {"opacity": 1.5}}}, "opacity"),
        ({"layers": {"1/0": {"color": "red; x"}}}, "color"),
        ({"solo": "7/0"}, "solo"),
    ],
)
def test_a_bad_layer_change_is_refused_naming_it(change, parameter):
    with pytest.raises(ArgumentError) as err:
        GDSPanel.from_view(gds_view()).with_changes(change)
    assert err.value.argument == parameter


def test_the_rows_of_the_layer_panel_name_each_layer_and_its_style():
    rows = layer_rows(GDSPanel.from_view(gds_view()).with_changes({"layers": {"2/5": {"hatch": "/"}}}))
    assert [r["key"] for r in rows] == ["1/0", "2/0", "2/5"]
    assert rows[0]["label"] == "core (1/0)" and rows[1]["label"] == "2/0"
    assert rows[2]["polygons"] == 3 and rows[2]["hatch"] == "/"
    assert set(HATCHES) == {"", "/", "\\", "x", "-", "|", "+", "."}


def test_a_klayout_layer_properties_file_sets_names_colors_hatches_and_visibility(tmp_path):
    path = tmp_path / "pdk.lyp"
    path.write_text(LYP)
    styles = read_lyp(path)
    assert set(styles) == {"1/0", "2/0", "2/5"}  # the wildcard entry names no layer
    assert styles["1/0"] == {"name": "WG", "color": "#ff8000", "hatch": "/", "visible": True, "opacity": 0.5}
    assert styles["2/0"]["visible"] is False and styles["2/0"]["opacity"] == 0.0
    assert styles["2/5"]["hatch"] == "x" and styles["2/5"]["name"] == "MARK"
    panel = GDSPanel.from_view(gds_view()).with_changes({"layer_properties": str(path)})
    assert panel.style("1/0").color == "#ff8000"
    assert layer_rows(panel)[0]["label"] == "WG (1/0)"
    # the text of a file the tab read works the same as its path
    assert GDSPanel.from_view(gds_view()).with_changes({"layer_properties": LYP}).styles == panel.styles


def test_a_file_that_is_not_layer_properties_is_refused():
    with pytest.raises(ArgumentError):
        read_lyp("<html></html>")


# ------------------------------------------------------------------ the two renderers


def test_plotly_draws_one_trace_per_layer_with_its_polygons_separated_by_gaps():
    panel = GDSPanel.from_view(gds_view()).with_changes(
        {"layers": {"2/0": {"visible": False, "hatch": "\\", "opacity": 0.25}}}
    )
    figure = to_plotly(compose(panel))
    traces = _layer_traces(figure)
    assert [t["meta"] for t in traces] == ["layer:1/0", "layer:2/0", "layer:2/5", "layer:shapes"]
    for trace, count in zip(traces[:3], (2, 1, 3)):
        x = _decode(trace["x"])
        assert trace["fill"] == "toself"
        assert int(np.isnan(x).sum()) == count - 1  # a gap between polygons, none after the last
    slab = traces[1]
    assert slab["visible"] is False
    assert slab["fillpattern"]["shape"] == "\\"
    assert slab["fillcolor"].endswith(",0.25)")
    assert traces[3]["fill"] == "none" and traces[3]["visible"] is False


def test_the_tabs_restyle_of_a_layer_is_its_trace_style():
    spec = compose(GDSPanel.from_view(gds_view()).with_changes({"layers": {"1/0": {"hatch": "+"}}}))
    trace = _layer_traces(to_plotly(spec))[0]
    style = layer_style(spec.layers[0])
    for key, value in style.items():
        assert trace[key] == value


def test_matplotlib_draws_one_collection_per_layer_with_the_same_styles():
    from matplotlib.colors import to_rgba

    panel = GDSPanel.from_view(gds_view()).with_changes(
        {"layers": {"2/0": {"visible": False}, "2/5": {"hatch": "x", "opacity": 0.8}}, "shapes": True}
    )
    spec = compose(panel)
    fig = mpl_renderer.to_figure(spec)
    collections = {c.get_gid(): c for c in fig.axes[0].collections if str(c.get_gid()).startswith("layer:")}
    traces = {t["meta"]: t for t in _layer_traces(to_plotly(spec))}
    assert set(collections) == set(traces)
    for layer in spec.layers:
        c, t = collections[f"layer:{layer.key}"], traces[f"layer:{layer.key}"]
        assert c.get_visible() == t["visible"] == layer.visible
        assert len(c.get_paths()) == len(layer.polygons)
        assert (c.get_hatch() or "") == t.get("fillpattern", {}).get("shape", "")
        if layer.filled:
            assert tuple(c.get_facecolor()[0]) == pytest.approx(to_rgba(layer.color, layer.opacity))
    assert mpl_renderer.render(spec, "png", dpi=50)[:4] == b"\x89PNG"


# ------------------------------------------------------------------ the session's view


def test_a_gds_view_is_its_own_kind_in_the_plots_list():
    panel = GDSPanel.from_view(gds_view())
    assert kind_of(panel) == "gds"
    view = ViewRegistry().add(panel, {"view": "gds"}, PlotOptions(), (1200, 750))
    assert view.id == "g1"
    assert title_of(panel, {}) == "GDS layers"


def _gds_session(views=("plot", "smatrix", "material", "session", "gds")):
    version = {**VERSION_1_0_9, "views": list(views)}
    return session(version=version, plot=lambda request: gds_view())


def test_plot_view_gds_keeps_the_layer_options_in_the_client(monkeypatch):
    monkeypatch.setenv("EMODE_NONINTERACTIVE", "1")
    em, client = _gds_session()
    handle = em.plot(view="gds", layers={"2/0": {"visible": False}}, shapes=True)
    (call,) = client.calls("EM_plot")
    assert call["view"] == "gds"
    assert not {"layers", "shapes", "layer_properties"} & set(call)
    assert isinstance(handle.view, GDSPanel)
    assert not handle.view.style("2/0").visible and handle.view.shapes


def test_an_emode_without_the_gds_view_is_refused_before_the_call(monkeypatch):
    from emode.types import EModeError

    monkeypatch.setenv("EMODE_NONINTERACTIVE", "1")
    em, client = _gds_session(views=("plot", "smatrix", "material", "session"))
    with pytest.raises(EModeError, match="1.0.9"):
        em.plot(view="gds")
    assert not client.calls("EM_plot")


def test_a_headless_save_draws_the_layers(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    em, _ = _gds_session()
    image = em.plot(view="gds", layers={"1/0": {"hatch": "/"}}, file_name="layers", file_type="svg")
    assert image.file.endswith("layers.svg")
    assert "<svg" in (tmp_path / "EM_figures" / "layers.svg").read_text()
