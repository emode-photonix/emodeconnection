"""A z panel is drawn as one image, so no seams show between its slices.

Eric's phone showed the EME taper's z-x panel with hairline gaps at every slice
boundary. EMode sends one raster per slice (here two 32-column straight sections
and 50 taper slices of 2 or 3 columns, all on one 200-row grid), and each was
drawn as its own Plotly heatmap, antialiased on its own. ``compose()`` now joins
neighboring slices on the same transverse grid into one raster whose cell edges
are the slices' own (``RasterSpec.x_edges``), and both renderers draw that one.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from emode.plotting import Raster
from emode.ui.figure import compose
from emode.ui.renderers import mpl
from emode.ui.renderers.plotly import to_plotly

from plot_fixtures import zx_view

LENGTH = 30000.0  # nm


def _slice(z0: float, z1: float, columns: int, y: np.ndarray, value: float) -> Raster:
    edges = np.linspace(z0, z1, columns + 1)
    centers = (edges[1:] + edges[:-1]) / 2
    z = np.full((y.size, columns), value) + np.linspace(0, 0.01, columns)
    return Raster(x=centers, y=y, z=z)


def taper_view():
    """Two 32-column straights around 50 taper slices of 2 or 3 columns, on one 200-row grid."""
    y = np.linspace(-1500, 1500, 200)
    rasters = [_slice(0.0, 5000.0, 32, y, 0.2)]
    bounds = np.linspace(5000.0, 25000.0, 51)
    for i in range(50):
        rasters.append(_slice(bounds[i], bounds[i + 1], 2 + i % 2, y, 0.3 + 0.01 * i))
    rasters.append(_slice(25000.0, LENGTH, 32, y, 0.9))
    view = zx_view()
    axes = view.axes.model_copy(
        update={"xlim": (0.0, LENGTH), "ylim": (-1500.0, 1500.0)}
    )
    return view.model_copy(update={"rasters": rasters, "axes": axes}), rasters


def test_the_taper_is_one_raster_over_the_whole_device():
    view, slices = taper_view()
    spec = compose(view)
    (raster,) = spec.rasters
    columns = sum(r.x.size for r in slices)
    assert np.asarray(raster.z).shape == (200, columns)
    assert np.asarray(raster.x).size == columns
    edges = np.asarray(raster.x_edges)
    assert edges.size == columns + 1
    assert edges[0] == pytest.approx(0.0, abs=1e-3)
    assert edges[-1] == pytest.approx(LENGTH / 1000, rel=1e-6)  # z in um
    assert np.all(np.diff(edges) > 0)
    # each slice keeps its own boundaries: the straight's last edge is 5 um
    assert np.any(np.isclose(edges, 5.0, atol=1e-4))


def test_the_values_stay_in_their_columns():
    view, slices = taper_view()
    (raster,) = compose(view).rasters
    expected = np.concatenate([np.asarray(r.z) for r in slices], axis=1)
    assert np.allclose(np.asarray(raster.z), expected)


def test_plotly_draws_one_heatmap_on_the_edges():
    view, slices = taper_view()
    heatmaps = [t for t in to_plotly(compose(view))["data"] if t["type"] == "heatmap"]
    assert len(heatmaps) == 1
    columns = sum(r.x.size for r in slices)
    x = np.frombuffer(base64.b64decode(heatmaps[0]["x"]["bdata"]), np.float32)
    assert x.size == columns + 1


def test_matplotlib_draws_one_mesh():
    view, _ = taper_view()
    fig = mpl.to_figure(compose(view))
    ax = fig.axes[0]
    from matplotlib.collections import QuadMesh

    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    assert len(meshes) == 1
    assert len(ax.images) == 0


def test_slices_on_different_transverse_grids_stay_apart():
    spec = compose(zx_view())  # two slices of 4 and 6 rows
    assert len(spec.rasters) == 2


def test_slices_with_a_gap_between_them_stay_apart():
    y = np.linspace(-100, 100, 5)
    view = zx_view().model_copy(
        update={"rasters": [_slice(0, 1000, 4, y, 0.1), _slice(1500, 2000, 4, y, 0.2)]}
    )
    assert len(compose(view).rasters) == 2
