"""The S-matrix, material and far-field views: their wire types, figures and both renderers.

From EMode 1.0.9 ``plot_S_matrix()`` and ``material_explorer()`` return the data
of their views (``SMatrixView``, ``MaterialView``) and ``FarField.show()`` and
``SMatrix.show()`` show data the client already holds. ``compose()`` turns each
into a ``FigureSpec``, which the browser's Plotly renderer and the static
matplotlib renderer both draw.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from emode import compat
from emode.emodeclient import obj_hook
from emode.plotting import MaterialView, SessionInfo, SMatrixView
from emode.types import ArgumentError, reconstruct, wire_dump
from emode.ui.figure import FarFieldPanel, PlotOptions, compose
from emode.ui.renderers import mpl
from emode.ui.renderers.plotly import to_plotly
from emode.ui.views import _select, kind_of, menu_of, title_of

from plot_fixtures import farfield, material_view, session_info, smatrix_view


def _round_trip(value):
    """`value` sent as an EMode reply is (arrays as ``__ndarray__``) and read back as the client reads one."""
    from emode.types import serialize

    payload = json.dumps(serialize(wire_dump(value), ndarray=True))
    return reconstruct(json.loads(payload, object_hook=obj_hook))


# ------------------------------------------------------------------ the wire


@pytest.mark.parametrize("make", [smatrix_view, material_view, session_info], ids=["smatrix", "material", "session"])
def test_each_view_round_trips_the_wire(make):
    value = make()
    back = _round_trip(value)
    assert type(back) is type(value)
    if isinstance(value, SMatrixView):
        assert np.array_equal(back.smatrix.data, value.smatrix.data)
        assert back.menu == value.menu
    elif isinstance(value, MaterialView):
        assert back.n.dtype == np.float64 and np.array_equal(back.n, value.n)
        assert back.requested == value.requested
    else:
        assert back == value


def test_the_new_types_are_versioned():
    for name in ("SMatrixView", "MaterialView", "SessionInfo"):
        assert compat.TYPE_SINCE[name] == "1.0.9"


# ------------------------------------------------------------------ the S-matrix


def test_the_s_matrix_figure_is_the_selected_block():
    view = smatrix_view("abs^2")
    spec = compose(view)
    (raster,) = spec.rasters
    expected = np.abs(view.smatrix.block(to_ports=1, from_ports=0)) ** 2
    assert np.array_equal(raster.z, expected)
    assert raster.z.shape == (1, 2)  # one output mode, two input modes
    assert spec.axes.xlabel == "Input mode" and spec.axes.ylabel == "Output mode"
    assert spec.axes.xrange == (-0.5, 1.5) and spec.axes.yrange == (-0.5, 0.5)
    assert spec.axes.integer_ticks


def test_the_automatic_scale_is_the_windows():
    """The block's extremes, ``abs^2`` snapped to 0 to 1 over a range of a half, the tick format by size."""
    from emode.smatrix import SMatrix

    view = smatrix_view("abs^2")
    raster = compose(view).rasters[0]
    assert (raster.vmin, raster.vmax) == pytest.approx((0.01, 0.49))
    reflection = _select(view, {"output": "in"})  # 0.0025 and zeros
    spec = compose(reflection)
    assert (spec.rasters[0].vmin, spec.rasters[0].vmax) == pytest.approx((0.0, 0.0025))
    assert spec.colorbar.tick_format == "%.1e"
    identity = view.model_copy(update={"smatrix": SMatrix(
        port_sizes=(2, 1, 1), data=np.eye(4, dtype=complex), port_names=("in", "out_a", "out_b"),
    )})
    snapped = compose(_select(identity, {"output": "in"})).rasters[0]
    assert (snapped.vmin, snapped.vmax) == (0.0, 1.0)


def test_phase_is_cyclic_and_the_typed_scale_wins():
    spec = compose(smatrix_view("angle"), PlotOptions(vmin=-3.0, vmax=3.0))
    assert spec.rasters[0].colormap == "cyclic"
    assert (spec.rasters[0].vmin, spec.rasters[0].vmax) == (-3.0, 3.0)
    assert spec.colorbar.tick_format == "%3.2f"


def test_ports_select_by_index_label_or_name():
    view = smatrix_view()
    assert _select(view, {"input": 0, "output": 2}).menu.selected["output"] == "out_b"
    assert _select(view, {"output": "out_b"}).menu.selected["output"] == "out_b"
    assert _select(view, {"plot_function": "real"}).menu.selected["component"] == "real"
    with pytest.raises(ArgumentError):
        _select(view, {"input": "nowhere"})


def test_an_smatrix_the_user_holds_becomes_a_view():
    from emode.ui.session import smatrix_view as from_smatrix

    view = from_smatrix(smatrix_view().smatrix)
    assert view.port_labels == ["in", "out_a", "out_b"]
    assert view.menu.selected == {"input": "in", "output": "out_b", "component": "abs^2"}
    assert [len(m) for m in view.mode_labels] == [2, 1, 1]


# ------------------------------------------------------------------ the material explorer


def test_the_material_figure_draws_the_model_and_its_citation():
    view = material_view(requested=(1200.0, 1700.0))
    spec = compose(view)
    (line,) = spec.lines  # inside the model's range: no extrapolation
    assert line.dash is None and np.array_equal(line.x, view.wavelengths)
    assert np.array_equal(line.y, view.n[0])
    assert spec.axes.xrange == (1200.0, 1700.0)
    lo, hi = spec.axes.yrange
    inside = view.n[0][(view.wavelengths >= 1200) & (view.wavelengths <= 1700)]
    span = inside.max() - inside.min()
    assert lo == pytest.approx(inside.min() - 0.05 * span) and hi == pytest.approx(inside.max() + 0.05 * span)
    assert spec.axes.grid
    (footer,) = [t for t in spec.texts if t.kind == "footer"]
    assert "made-up model" in footer.text
    assert spec.axes.xlabel == "Wavelength (nm)" and spec.axes.ylabel == "Refractive index (-)"


def test_a_range_past_the_model_is_drawn_dashed_first():
    spec = compose(material_view(requested=(800.0, 2500.0)))
    dashed, solid = spec.lines
    assert dashed.dash == "dash" and solid.dash is None
    assert dashed.x[0] == 800.0


def test_the_component_is_switched_in_the_client():
    view = _select(material_view(), {"component": "n_z"})
    assert np.array_equal(compose(view).lines[-1].y, view.n[2])


# ------------------------------------------------------------------ the far field


def test_the_far_field_figure_is_its_intensity_in_direction_cosines():
    ff = farfield()
    spec = compose(ff)
    (raster,) = spec.rasters
    assert np.array_equal(raster.z, np.asarray(ff.intensity).T)
    assert np.array_equal(raster.x, ff.ux) and np.array_equal(raster.y, ff.uy)
    assert spec.axes.equal_aspect
    assert any("divergence 12.50°" in t.text and "M² 1.010" in t.text for t in spec.texts)
    assert "1550 nm" in spec.title


def test_the_screen_is_the_far_fields_second_panel():
    panel = FarFieldPanel(farfield(screen=True))
    assert panel.panels == ["intensity", "screen"]
    screen = _select(panel, {"data": "screen"})
    spec = compose(screen)
    assert spec.rasters[0].z.shape == (3, 5)
    assert spec.axes.xlabel == "u (μm)"
    assert menu_of(screen).selected == {"data": "screen"}
    assert FarFieldPanel(farfield()).panels == ["intensity"]


# ------------------------------------------------------------------ kinds and both renderers


@pytest.mark.parametrize(
    "make, kind, prefix",
    [
        (smatrix_view, "smatrix", "S-matrix in to out_a abs^2"),
        (material_view, "material", "material Made n_x"),
        (lambda: FarFieldPanel(farfield()), "farfield", "far field 1550 nm intensity"),
    ],
)
def test_each_kind_has_a_title(make, kind, prefix):
    data = make()
    assert kind_of(data) == kind
    assert title_of(data, {}) == prefix


@pytest.mark.parametrize(
    "make", [smatrix_view, material_view, farfield, lambda: FarFieldPanel(farfield(screen=True), "screen")]
)
def test_both_renderers_draw_each_view(make):
    spec = compose(make())
    figure = to_plotly(spec)
    kinds = [t["type"] for t in figure["data"]]
    assert ("heatmap" in kinds) == bool(spec.rasters)
    assert kinds.count("scatter") >= len(spec.lines)
    fig = mpl.to_figure(spec)
    ax = fig.axes[0]
    assert len(ax.images) == len(spec.rasters)
    assert len(ax.lines) == len(spec.lines)
    assert ax.get_xlabel() == spec.axes.xlabel
    assert mpl.render(spec, "png", dpi=72).startswith(b"\x89PNG")


def test_the_footer_gets_room_in_the_tab():
    figure = to_plotly(compose(material_view()))
    assert figure["layout"]["margin"]["b"] > 50
    assert figure["layout"]["xaxis"]["showgrid"] is True
