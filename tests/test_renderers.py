"""The two renderers of one ``FigureSpec``: Plotly JSON for the tab, matplotlib for an image.

They read the same spec and decide nothing, so the browser tab and a saved
file agree. Each is checked against the spec, and the two side by side.
"""
from __future__ import annotations

import base64
import builtins
import sys

import numpy as np
import pytest

from emode.types import EModeError
from emode.ui.figure import PlotOptions, compose
from emode.ui.renderers.plotly import to_plotly

from plot_fixtures import index_view, layout_view, schematic, xy_view, zx_view

mpl = pytest.importorskip("matplotlib")
from emode.ui.renderers import mpl as mpl_renderer  # noqa: E402


def _decode(typed: dict) -> np.ndarray:
    array = np.frombuffer(base64.b64decode(typed["bdata"]), dtype=np.dtype(typed["dtype"]))
    if "shape" in typed:
        array = array.reshape([int(n) for n in typed["shape"].split(",")])
    return array


def _traces(figure: dict, kind: str) -> list:
    return [t for t in figure["data"] if t["type"] == kind]


# ------------------------------------------------------------------ Plotly


def test_a_raster_is_a_heatmap_with_its_values_as_typed_arrays():
    spec = compose(xy_view())
    (heatmap,) = _traces(to_plotly(spec), "heatmap")
    np.testing.assert_array_equal(_decode(heatmap["z"]), spec.rasters[0].z)
    np.testing.assert_array_equal(_decode(heatmap["x"]), spec.rasters[0].x)
    assert (heatmap["zmin"], heatmap["zmax"]) == (-1.0, 1.0)
    assert heatmap["zsmooth"] == "best"


def test_the_colorscale_is_the_rasters_kind():
    from emode.colormaps import plotly_colorscale

    (heatmap,) = _traces(to_plotly(compose(xy_view())), "heatmap")
    assert heatmap["colorscale"] == plotly_colorscale("diverging")
    first = _traces(to_plotly(compose(zx_view())), "heatmap")[0]
    assert first["colorscale"] == plotly_colorscale("sequential")


def test_contours_are_drawn_when_on_and_not_when_off():
    levels = xy_view().rasters[0].contour_levels
    on = _traces(to_plotly(compose(xy_view())), "contour")
    assert [t["contours"]["start"] for t in on] == pytest.approx(list(levels))
    assert not _traces(to_plotly(compose(xy_view(), PlotOptions(contours=False))), "contour")


def test_one_scatter_per_overlay_styled_by_its_kind():
    figure = to_plotly(compose(zx_view()))
    lines = [t for t in _traces(figure, "scatter") if t.get("name") in ("outline", "port", "facet")]
    assert [t["name"] for t in lines] == ["facet", "port", "outline", "outline"]
    port = next(t for t in lines if t["name"] == "port")
    assert port["line"]["color"] == "#00ff00"
    assert port["line"]["width"] == pytest.approx(2.0)


def test_the_axes_titles_reversed_y_and_fixed_aspect():
    layout = to_plotly(compose(zx_view()))["layout"]
    assert layout["xaxis"]["title"]["text"] == "z (μm)"
    assert layout["yaxis"]["range"] == [1200.0, -1200.0]
    assert "scaleanchor" not in layout["yaxis"]
    layout = to_plotly(compose(layout_view()))["layout"]
    assert layout["yaxis"]["scaleanchor"] == "x"


def test_legend_entries_and_the_index_colorbar():
    figure = to_plotly(compose(index_view()))
    legend = [t for t in figure["data"] if t.get("showlegend")]
    assert [t["name"] for t in legend] == ["core"]
    bar = next(t for t in figure["data"] if t.get("marker", {}).get("showscale"))
    assert bar["marker"]["colorbar"]["ticktext"] == ["1.44", "3.48"]


def test_a_masked_cell_is_empty():
    (heatmap,) = _traces(to_plotly(compose(layout_view())), "heatmap")
    z = _decode(heatmap["z"])
    assert np.isnan(z[:3]).all() and not np.isnan(z[3:]).any()


def test_the_arrows_are_annotations_naming_their_section():
    layout = to_plotly(compose(zx_view()))["layout"]
    (arrow,) = [a for a in layout["annotations"] if a["name"] == "arrow"]
    assert arrow["hovertext"] == "go to s2"


# ------------------------------------------------------------------ matplotlib


def test_the_matplotlib_figure_has_the_specs_size_dpi_and_labels():
    spec = compose(xy_view(), PlotOptions(figure_size=(127.0, 50.8), dpi=200))
    fig = mpl_renderer.to_figure(spec)
    assert tuple(fig.get_size_inches()) == pytest.approx((5.0, 2.0))
    assert fig.dpi == 200
    ax = fig.axes[0]
    assert ax.get_xlabel() == "x (nm)" and ax.get_ylabel() == "y (nm)"


def test_the_matplotlib_image_holds_the_raster():
    spec = compose(xy_view())
    ax = mpl_renderer.to_figure(spec).axes[0]
    np.testing.assert_array_equal(np.asarray(ax.images[0].get_array()), spec.rasters[0].z)


def _pixel(fig, ax, x: float, y: float) -> np.ndarray:
    """The rendered RGB at data point (x, y)."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    px, py = ax.transData.transform((x, y))
    return rgba[rgba.shape[0] - 1 - int(round(py)), int(round(px)), :3].astype(int)


def test_each_matplotlib_image_stays_inside_its_own_extent():
    """A z panel is one image per slice. An image painted past its cells covered
    the panel with its edge values, so the last slice drawn showed everywhere and
    a field growing along z rendered flat."""
    spec = compose(zx_view())
    fig = mpl_renderer.to_figure(spec)
    ax = fig.axes[0]
    first, second = spec.rasters
    norm = ax.images[0].norm
    cmap = ax.images[0].get_cmap()

    def color(value: float) -> np.ndarray:
        return np.round(np.asarray(cmap(norm(value)))[:3] * 255).astype(int)

    # the middle of the first slice is the first slice's value, not the last's
    at_first = _pixel(fig, ax, float(np.mean(first.x)), 0.0)
    assert np.abs(at_first - color(0.5)).max() <= 2
    # the second slice is still drawn where it is
    at_second = _pixel(fig, ax, float(np.mean(second.x)), 0.0)
    assert np.abs(at_second - color(0.25)).max() <= 2
    # past the first slice's cells across x (it spans +-1067 nm, the panel +-1200)
    # is the axes background, not the slice's edge row stretched outward
    outside = _pixel(fig, ax, float(np.mean(first.x)), 1150.0)
    assert np.abs(outside - color(0.5)).max() > 10
    assert np.abs(outside - color(0.25)).max() > 10


def test_one_matplotlib_artist_per_overlay():
    spec = compose(zx_view())
    ax = mpl_renderer.to_figure(spec).axes[0]
    assert [line.get_gid() for line in ax.lines] == [line.kind for line in spec.lines]
    assert len(ax.images) == 2


# ------------------------------------------------------------------ both


@pytest.mark.parametrize("make", [xy_view, zx_view, layout_view, index_view, schematic])
def test_the_two_renderers_agree_on_one_spec(make):
    spec = compose(make())
    figure = to_plotly(spec)
    ax = mpl_renderer.to_figure(spec).axes[0]
    heatmaps = _traces(figure, "heatmap")
    assert len(heatmaps) == len(ax.images) == len(spec.rasters)
    for heatmap, image in zip(heatmaps, ax.images):
        np.testing.assert_array_equal(
            np.ma.filled(np.asarray(image.get_array(), float), np.nan), _decode(heatmap["z"]).astype(float)
        )
    plotted = [t for t in _traces(figure, "scatter") if t.get("name") in {line.kind for line in spec.lines}]
    assert len(plotted) == len(ax.lines) == len(spec.lines)
    assert ax.get_xlim() == pytest.approx(figure["layout"]["xaxis"]["range"])
    y = figure["layout"]["yaxis"]["range"]
    assert ax.get_ylim() == pytest.approx(y)
    if spec.axes.visible:
        assert ax.get_xlabel() == figure["layout"]["xaxis"]["title"]["text"]
        assert ax.get_ylabel() == figure["layout"]["yaxis"]["title"]["text"]


def test_a_png_render_has_the_requested_resolution():
    spec = compose(xy_view())
    png = mpl_renderer.render(spec, "png", dpi=150)
    assert png.startswith(b"\x89PNG")
    from matplotlib.image import imread
    import io

    height, width = imread(io.BytesIO(png)).shape[:2]
    # cropped to the drawing, so no larger than the figure at that dpi
    assert width <= spec.width_mm / 25.4 * 150 * 1.5 and height > 50


def test_without_matplotlib_a_save_names_the_install_command_and_plotly_still_works(monkeypatch):
    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    for name in [m for m in sys.modules if m == "matplotlib" or m.startswith("matplotlib.")]:
        monkeypatch.delitem(sys.modules, name)
    spec = compose(zx_view())
    assert to_plotly(spec)["data"]
    with pytest.raises(EModeError, match="pip install matplotlib"):
        mpl_renderer.render(spec)
