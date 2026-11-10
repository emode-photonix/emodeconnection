"""Every plot of the client draws with EMode's registered colormaps and line colors.

The panels of ``plot()``, ``plot_S_matrix()``, ``material_explorer()`` and
``FarField.show()`` through ``compose()`` and both renderers (Plotly for the
tab, matplotlib for a static image), the schematic, and the matplotlib plots of
``SMatrix.plot`` and ``FarField.plot``. A raster's colormap is a registered
name (``emode-sequential``, ``emode-diverging``, ``emode-cyclic``), never a
matplotlib default, unless ``colormap=`` names another.
"""
from __future__ import annotations

import pytest

from emode import colormaps
from emode.ui.figure import FarFieldPanel, PlotOptions, compose
from emode.ui.renderers.plotly import to_plotly

from plot_fixtures import (
    farfield,
    index_view,
    layout_view,
    material_view,
    schematic,
    smatrix,
    smatrix_view,
    xy_view,
    zx_view,
)

mpl = pytest.importorskip("matplotlib")
from emode.ui.renderers import mpl as mpl_renderer  # noqa: E402

#: Each view and the colormap of the family its raster is drawn with.
VIEWS = {
    "field, signed x-y": (xy_view, "emode-diverging"),
    "field, z-x": (zx_view, "emode-sequential"),
    "layout": (layout_view, "emode-sequential"),
    "S-matrix abs^2": (lambda: smatrix_view("abs^2"), "emode-sequential"),
    "S-matrix angle": (lambda: smatrix_view("angle"), "emode-cyclic"),
    "S-matrix real": (lambda: smatrix_view("real"), "emode-sequential"),
    "far field": (lambda: FarFieldPanel(farfield()), "emode-sequential"),
    "far field screen": (lambda: FarFieldPanel(farfield(screen=True), "screen"), "emode-sequential"),
}


def _mpl_colormap_names(fig) -> set:
    names = set()
    for ax in fig.axes:
        for artist in list(ax.images) + list(ax.collections):
            cmap = getattr(artist, "cmap", None)
            if cmap is not None and artist.get_array() is not None:
                names.add(cmap.name)
    return names


@pytest.mark.parametrize("label", VIEWS)
def test_compose_names_a_registered_colormap(label):
    make, name = VIEWS[label]
    spec = compose(make())
    assert spec.rasters, label
    assert {r.colormap for r in spec.rasters} == {name}
    assert spec.colorbar.colormap == name


@pytest.mark.parametrize("label", VIEWS)
def test_matplotlib_draws_with_the_registered_colormap(label):
    make, name = VIEWS[label]
    fig = mpl_renderer.to_figure(compose(make()))
    assert _mpl_colormap_names(fig) == {name}
    assert mpl.colormaps[name] is not None


@pytest.mark.parametrize("label", VIEWS)
def test_plotly_draws_with_the_same_stops(label):
    make, name = VIEWS[label]
    figure = to_plotly(compose(make()))
    heatmaps = [t for t in figure["data"] if t["type"] == "heatmap"]
    assert heatmaps
    for trace in heatmaps:
        assert trace["colorscale"] == colormaps.plotly_colorscale(name)


@pytest.mark.parametrize("label", VIEWS)
def test_colormap_replaces_the_family_in_both_renderers(label):
    make, _ = VIEWS[label]
    spec = compose(make(), PlotOptions(colormap="viridis"))
    assert {r.colormap for r in spec.rasters} == {"viridis"}
    assert _mpl_colormap_names(mpl_renderer.to_figure(spec)) == {"viridis"}
    heatmap = next(t for t in to_plotly(spec)["data"] if t["type"] == "heatmap")
    assert heatmap["colorscale"][0][1] == mpl.colors.to_hex(mpl.colormaps["viridis"](0.0))


def test_colormap_can_name_another_map_of_the_family():
    spec = compose(xy_view(), PlotOptions(colormap="emode-cyclic"))
    assert spec.rasters[0].colormap == "emode-cyclic"
    heatmap = next(t for t in to_plotly(spec)["data"] if t["type"] == "heatmap")
    assert heatmap["colorscale"] == colormaps.plotly_colorscale("cyclic")


def test_the_index_panel_fills_from_the_sequential_map():
    spec = compose(index_view())
    assert spec.colorbar.colormap == "emode-sequential"
    assert spec.fills[0].color == colormaps.color_at("sequential", 0.0)


def test_the_schematic_is_drawn_in_the_line_palette():
    roles = colormaps.schematic_colors()
    assert {roles["block_edge"], roles["launch"], roles["monitor"]} <= set(colormaps.line_palette())
    spec = compose(schematic())
    blocks = [f for f in spec.fills if f.kind == "block"]
    assert {b.edgecolor for b in blocks} == {roles["block_edge"]}
    assert {b.color for b in blocks} == {roles["block_face"]}
    (launch,) = [line for line in spec.lines if line.kind == "launch"]
    assert launch.color == roles["launch"]
    assert {t.color for t in spec.texts if t.kind == "monitor"} == {roles["monitor"]}
    assert {line.color for line in spec.lines if line.kind == "connection"} == {roles["connection"]}


def test_the_material_explorer_draws_its_curves_in_the_first_line_color():
    spec = compose(material_view(requested=(1000.0, 1700.0)))
    assert spec.lines
    assert {line.color for line in spec.lines} == {colormaps.line_palette()[0]}


def test_overlay_colors_come_from_the_module():
    spec = compose(xy_view())
    colors = colormaps.overlay_colors()
    for line in spec.lines:
        assert line.color == colors[line.kind]


# ------------------------------------------------------------------ the matplotlib plots of the wire types


@pytest.mark.parametrize(("kind", "name"), [("abs**2", "emode-sequential"), ("phase", "emode-cyclic")])
def test_smatrix_plot_uses_the_family(kind, name):
    fig = smatrix().plot(kind)
    assert _mpl_colormap_names(fig) == {name}
    mpl.pyplot.close(fig)


def test_smatrix_plot_takes_a_colormap_by_name():
    fig = smatrix().plot("phase", colormap="twilight")
    assert _mpl_colormap_names(fig) == {"twilight"}
    mpl.pyplot.close(fig)


def test_farfield_plot_uses_the_family_and_the_line_palette():
    fig = farfield(screen=True).plot()
    assert _mpl_colormap_names(fig) == {"emode-sequential"}
    cuts = fig.axes[1].lines
    palette = colormaps.line_palette()
    assert [mpl.colors.to_hex(line.get_color()) for line in cuts] == [c.lower() for c in palette[: len(cuts)]]
    mpl.pyplot.close(fig)


def test_farfield_plot_takes_a_colormap_by_name():
    fig = farfield().plot(colormap="magma")
    assert _mpl_colormap_names(fig) == {"magma"}
    mpl.pyplot.close(fig)


def test_an_unknown_colormap_on_a_wire_types_plot_is_refused():
    from emode.types import ArgumentError

    with pytest.raises(ArgumentError, match="emode-sequential"):
        farfield().plot(colormap="nope")
    with pytest.raises(ArgumentError, match="emode-cyclic"):
        smatrix().plot(colormap="nope")
