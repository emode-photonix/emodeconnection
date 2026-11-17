"""The figure options of ``plot()``, ``plot_S_matrix()`` and ``material_explorer()``.

``title``, ``xlabel``, ``ylabel``, ``units``, ``font_size``, ``figure_size`` (mm),
``colorbar_label`` and ``dpi`` are applied by ``compose()`` over the view's own
labels and units. Both renderers read them from the spec, so a spec with each
option renders the same in the browser tab and in a saved image. The views'
data is never changed. These check the matplotlib objects and the Plotly
figure dictionary, not pixels.
"""
from __future__ import annotations

import numpy as np
import pytest

from emode.types import ArgumentError
from emode.ui.figure import PlotOptions, compose
from emode.ui.renderers.plotly import to_plotly

from plot_fixtures import material_view, smatrix_view, xy_view, zx_view

pytest.importorskip("matplotlib")
from emode.ui.renderers import mpl as mpl_renderer  # noqa: E402

ALL = dict(
    title="T", xlabel="XL", ylabel="YL", units="um", font_size=11.0, figure_size=(120.0, 60.0),
    colorbar_label="CB", dpi=200,
)


def _both(view, options):
    spec = compose(view, options)
    return spec, to_plotly(spec)["layout"], mpl_renderer.to_figure(spec)


@pytest.mark.parametrize("make", [xy_view, zx_view])
def test_every_option_reaches_both_renderers_of_a_field_view(make):
    spec, layout, fig = _both(make(), PlotOptions(**ALL))
    ax = fig.axes[0]
    assert ax.get_title() == "T" and layout["title"]["text"] == "T"
    assert ax.get_xlabel() == "XL" == layout["xaxis"]["title"]["text"]
    assert ax.get_ylabel() == "YL" == layout["yaxis"]["title"]["text"]
    assert ax.xaxis.label.get_fontsize() == 11.0
    assert layout["font"]["size"] == pytest.approx(11.0 + 3)
    assert tuple(fig.get_size_inches()) == pytest.approx((120 / 25.4, 60 / 25.4))
    assert fig.dpi == 200
    assert spec.colorbar.label == "CB"
    assert fig.axes[-1].get_ylabel() == "CB"
    cb = to_plotly(spec)["data"][0]["colorbar"]
    assert cb["title"]["text"] == "CB"


def test_units_scale_the_coordinates_and_the_ranges():
    view = xy_view()
    nm = compose(view, PlotOptions(units="nm"))
    um = compose(view, PlotOptions(units="um"))
    np.testing.assert_allclose(nm.rasters[0].x, um.rasters[0].x * 1000, rtol=1e-5)
    assert um.axes.xrange == pytest.approx(tuple(v / 1000 for v in nm.axes.xrange))
    assert um.axes.xlabel.endswith("(μm)") and nm.axes.xlabel.endswith("(nm)")


def test_units_can_differ_per_axis():
    spec = compose(zx_view(), PlotOptions(units=("um", "nm")))
    default = compose(zx_view(), PlotOptions(units=("nm", "nm")))
    assert spec.axes.xlabel.endswith("(μm)") and spec.axes.ylabel.endswith("(nm)")
    assert spec.axes.xrange == pytest.approx(tuple(v / 1000 for v in default.axes.xrange))
    assert spec.axes.yrange == pytest.approx(default.axes.yrange)


@pytest.mark.parametrize("units", ["mm", ("nm", "mm"), ("nm",), 3])
def test_an_invalid_unit_raises_naming_the_option(units):
    with pytest.raises(ArgumentError) as info:
        PlotOptions.from_arguments({"units": units})
    assert "units" in str(info.value)


def test_the_views_data_is_unchanged_by_the_options():
    view, reference = xy_view(), xy_view()
    compose(view, PlotOptions(**ALL))
    for got, want in zip(view.rasters, reference.rasters):
        np.testing.assert_array_equal(got.x, want.x)
        np.testing.assert_array_equal(got.y, want.y)
        np.testing.assert_array_equal(got.z, want.z)
    assert view.axes == reference.axes


@pytest.mark.parametrize("make", [smatrix_view, material_view])
def test_every_option_reaches_both_renderers_of_the_other_panels(make):
    options = PlotOptions(**{k: v for k, v in ALL.items() if k != "units"})
    spec, layout, fig = _both(make(), options)
    ax = fig.axes[0]
    assert ax.get_title() == "T" and layout["title"]["text"] == "T"
    assert ax.get_xlabel() == "XL" == layout["xaxis"]["title"]["text"]
    assert ax.get_ylabel() == "YL" == layout["yaxis"]["title"]["text"]
    assert ax.xaxis.label.get_fontsize() == 11.0
    assert tuple(fig.get_size_inches()) == pytest.approx((120 / 25.4, 60 / 25.4))
    assert fig.dpi == 200
