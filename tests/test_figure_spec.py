"""``compose()``: EMode's plot data and the figure options become one ``FigureSpec``.

Every drawing decision is made in ``compose``, once, so the browser tab and a
saved image agree. These hold it to the fixture's data (rasters, overlays,
labels, units, ranges, orientation, aspect) and to each figure option.
"""
from __future__ import annotations

import numpy as np
import pytest

from emode import colormaps
from emode.types import ArgumentError
from emode.ui.figure import DEFAULT_SIZE_MM, PlotOptions, compose, split_arguments

from plot_fixtures import index_view, layout_view, schematic, xy_view, zx_view


def test_the_raster_is_the_views_values_on_its_centers():
    view = xy_view()
    spec = compose(view)
    (r,) = spec.rasters
    np.testing.assert_array_equal(r.z, view.rasters[0].z)
    np.testing.assert_array_equal(r.x, view.rasters[0].x)
    assert (r.vmin, r.vmax) == (-1.0, 1.0)
    assert r.colormap == "emode-diverging"
    assert r.interpolation == "bilinear"


def test_an_xy_panel_has_contours_and_a_z_panel_does_not_by_default():
    assert compose(xy_view()).rasters[0].contour_levels == tuple(float(v) for v in xy_view().rasters[0].contour_levels)
    assert all(r.contour_levels == () for r in compose(zx_view()).rasters)
    assert compose(xy_view(), PlotOptions(contours=False)).rasters[0].contour_levels == ()


def test_the_default_labels_and_units_come_from_the_view():
    spec = compose(zx_view())
    assert spec.axes.xlabel == "z (μm)"
    assert spec.axes.ylabel == "x (nm)"
    # z in um, x in nm, from coordinates in nm
    assert spec.axes.xrange == (0.0, 3.0)
    assert spec.axes.yrange == (-1200.0, 1200.0)
    np.testing.assert_allclose(spec.rasters[1].x, np.linspace(1, 3, 9))
    np.testing.assert_allclose(spec.rasters[1].y, np.linspace(-1200, 1200, 6))


def test_a_z_x_panel_has_x_increasing_downward():
    assert compose(zx_view()).axes.invert_y
    assert not compose(xy_view()).axes.invert_y


def test_the_layout_view_keeps_an_equal_aspect_and_its_mask():
    spec = compose(layout_view())
    assert spec.axes.equal_aspect
    assert spec.axes.xlabel == "z (μm)" and spec.axes.ylabel == "x (μm)"
    np.testing.assert_array_equal(spec.rasters[0].mask, layout_view().rasters[0].mask)


def test_units_scales_both_axes_and_their_labels_agree():
    spec = compose(xy_view(), PlotOptions(units="um"))
    assert spec.axes.xlabel == "x (μm)" and spec.axes.ylabel == "y (μm)"
    assert spec.axes.xrange == (-1.0, 1.0)
    np.testing.assert_allclose(spec.rasters[0].x, xy_view().rasters[0].x / 1000)
    np.testing.assert_allclose(spec.lines[0].x, np.array([-250, 250, 250, -250, -250]) / 1000)


def test_the_users_labels_size_font_and_dpi_override_the_defaults():
    options = PlotOptions(
        title="mode 0", xlabel="width", ylabel="height", font_size=11, figure_size=(120, 60),
        colorbar_label="field", dpi=200,
    )
    spec = compose(xy_view(), options)
    assert (spec.title, spec.axes.xlabel, spec.axes.ylabel) == ("mode 0", "width", "height")
    assert (spec.width_mm, spec.height_mm, spec.font_size, spec.dpi) == (120, 60, 11, 200)
    assert spec.colorbar.label == "field"
    assert all(t.size == 11 for t in spec.texts)


def test_the_default_size_is_the_windows():
    spec = compose(xy_view())
    assert (spec.width_mm, spec.height_mm) == pytest.approx(DEFAULT_SIZE_MM)
    assert spec.dpi == 600


def test_the_outline_toggle_removes_the_lines():
    assert compose(zx_view()).lines
    assert not compose(zx_view(), PlotOptions(index_outline=False)).lines


def test_the_text_toggle_removes_the_annotation_and_keeps_the_port_labels():
    """The window drew port and facet labels with their lines, under the outline toggle."""
    kinds = [t.kind for t in compose(zx_view(), PlotOptions(text_display=False)).texts]
    assert "title" not in kinds
    assert kinds == ["facet", "port"]


def test_ports_facets_and_their_labels_follow_the_outline_toggle():
    spec = compose(zx_view())
    kinds = [line.kind for line in spec.lines]
    assert kinds == ["facet", "port", "outline", "outline"]
    labels = {t.text: t.kind for t in spec.texts}
    assert labels["left"] == "port" and labels["facet"] == "facet"


def test_the_mesh_and_window_show_only_when_asked():
    assert not compose(xy_view()).segments
    assert len(compose(xy_view(), PlotOptions(mesh_display=True)).segments) == 1
    assert not [f for f in compose(xy_view()).fills if f.kind in ("window", "pml")]
    shown = compose(xy_view(), PlotOptions(window_display=True)).fills
    assert [f.kind for f in shown] == ["window", "pml"]
    assert shown[1].hatch == "xx"


def test_the_index_panel_fills_with_the_family_and_ticks_its_indices():
    spec = compose(index_view())
    assert [f.color for f in spec.fills] == [colormaps.color_at("sequential", 0.0), colormaps.color_at("sequential", 1.0)]
    assert spec.colorbar.ticks == (0.0, 1.0)
    assert spec.colorbar.tick_labels == ("1.44", "3.48")
    assert [e.label for e in spec.legend] == ["core"]


def test_a_legend_of_unfilled_shapes_is_not_drawn():
    """The field panels' shape names are the window's measuring legend, never shown."""
    assert compose(xy_view()).legend == ()


def test_a_named_colormap_replaces_the_kind():
    spec = compose(xy_view(), PlotOptions(colormap="viridis"))
    assert spec.rasters[0].colormap == "viridis"


def test_the_schematic_is_blocks_stubs_connections_and_a_launch_marker():
    spec = compose(schematic())
    blocks = [f for f in spec.fills if f.kind == "block"]
    assert len(blocks) == 2
    kinds = [line.kind for line in spec.lines]
    assert kinds.count("stub") == 4 and kinds.count("connection") == 1 and kinds.count("launch") == 1
    texts = {t.text: t for t in spec.texts}
    assert texts["a.left"].weight == "bold"
    assert "monitor m" in texts
    assert not spec.axes.visible and spec.axes.equal_aspect
    assert spec.axes.xrange == (-4.5, 10.9)


def test_the_arguments_split_into_data_and_figure():
    data, figure = split_arguments({"plane": "z-x", "component": "Ex", "contours": True, "title": "t"})
    assert data == {"plane": "z-x", "component": "Ex"}
    assert figure == {"contours": True, "title": "t"}


def test_a_bad_unit_is_refused():
    with pytest.raises(ArgumentError):
        PlotOptions.from_arguments({"units": "mm"})


def test_the_window_display_shows_the_whole_grid_when_the_view_gives_its_limits():
    view = xy_view()
    view = view.model_copy(
        update={"axes": view.axes.model_copy(update={"xlim_window": (-1100.0, 1100.0), "ylim_window": (-600.0, 800.0)})}
    )
    assert compose(view).axes.xrange == (-1000.0, 1000.0)
    spec = compose(view, PlotOptions(window_display=True))
    assert (spec.axes.xrange, spec.axes.yrange) == ((-1100.0, 1100.0), (-600.0, 800.0))
