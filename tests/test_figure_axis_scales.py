"""A log axis scale on a panel and its twin, and a secondary x axis (#138).

The Dispersion example draws its wavelength axis on a log scale with the matching
frequency in THz on a top axis. Both renderers draw the same ticks.
"""
from __future__ import annotations

import numpy as np
import pytest

import emode
from emode.types import ArgumentError
from emode.ui.figure import compose
from emode.ui.renderers.plotly import to_plotly

C_THZ_UM = 299.792458  # c in um THz


def dispersion() -> "emode.Figure":
    fig = emode.Figure()
    p = fig.panel(0)
    wl = np.linspace(1.0, 2.0, 21)
    p.line(wl, 1e3 / wl**2, label="GVD")
    p.set(xlabel="Wavelength (um)", ylabel="GVD", xscale="log", yscale="log")
    p.secondary_x(lambda w: C_THZ_UM / w, label="Frequency (THz)")
    p.twin().line(wl, 2.0 - 0.1 * wl).set(ylabel="n", yscale="log")
    return fig


def test_the_scales_reach_the_spec():
    (spec,) = compose(dispersion()).panels
    a = spec.axes
    assert (a.xscale, a.yscale, a.y2scale) == ("log", "log", "log")
    assert a.xrange == (1.0, 2.0)
    # the automatic y padding keeps a log range positive
    assert 0.0 < a.yrange[0] < 250.0 < 1000.0 < a.yrange[1]


def test_a_panel_without_a_scale_is_linear():
    (spec,) = compose(emode.Figure()).panels
    assert (spec.axes.xscale, spec.axes.yscale, spec.axes.y2scale) == ("linear",) * 3
    assert spec.axes.xsecondary is None


def test_a_scale_must_be_linear_or_log():
    with pytest.raises(ArgumentError):
        emode.Figure().panel(0).set(xscale="loga")
    with pytest.raises(ArgumentError):
        emode.Figure().panel(0).twin().set(yscale="sqrt")


def test_a_log_axis_refuses_a_range_that_is_not_positive():
    fig = emode.Figure()
    fig.panel(0).line([0.0, 1.0], [1.0, 2.0]).set(xscale="log")
    with pytest.raises(ArgumentError):
        compose(fig)


def test_the_secondary_axis_ticks_are_frequencies_placed_at_their_wavelengths():
    (spec,) = compose(dispersion()).panels
    s = spec.axes.xsecondary
    assert s.label == "Frequency (THz)"
    freqs = np.array([float(t) for t in s.ticklabels])
    pos = np.asarray(s.tickvals)
    assert len(pos) >= 2
    # every tick sits inside the x range, at the wavelength of its frequency
    assert ((pos >= 1.0) & (pos <= 2.0)).all()
    np.testing.assert_allclose(C_THZ_UM / pos, freqs, rtol=2e-3)
    assert freqs.min() >= C_THZ_UM / 2.0 - 1e-6 and freqs.max() <= C_THZ_UM / 1.0 + 1e-6


def test_the_secondary_axis_refuses_a_function_that_is_not_monotonic():
    fig = emode.Figure()
    fig.panel(0).line([-1.0, 1.0], [0.0, 1.0]).secondary_x(lambda x: x * x)
    with pytest.raises(ArgumentError):
        compose(fig)


def test_plotly_draws_log_axes_and_a_top_axis_with_the_same_ticks():
    spec = compose(dispersion())
    layout = to_plotly(spec)["layout"]
    assert layout["xaxis"]["type"] == "log" and layout["yaxis"]["type"] == "log"
    assert layout["xaxis"]["range"] == pytest.approx([0.0, np.log10(2.0)])
    assert layout["yaxis2"]["type"] == "log"
    top = layout["xaxis2"]
    s = spec.panels[0].axes.xsecondary if spec.panels else spec.axes.xsecondary
    assert top["side"] == "top" and top["overlaying"] == "x"
    assert top["type"] == "log" and top["range"] == layout["xaxis"]["range"]
    assert top["tickvals"] == list(s.tickvals) and top["ticktext"] == list(s.ticklabels)
    assert top["title"]["text"] == "Frequency (THz)"


def test_plotly_puts_each_panels_top_axis_past_every_panels_own():
    fig = emode.Figure(cols=2)
    for i in (0, 1):
        fig.panel(i).line([1.0, 2.0], [1.0, 2.0]).set(xscale="log").secondary_x(lambda w: 300.0 / w)
    layout = to_plotly(compose(fig))["layout"]
    assert layout["xaxis3"]["overlaying"] == "x" and layout["xaxis4"]["overlaying"] == "x2"
    assert layout["xaxis3"]["side"] == "top"


def test_matplotlib_draws_log_axes_and_a_top_axis_with_the_same_ticks():
    pytest.importorskip("matplotlib")
    from emode.ui.renderers import mpl

    spec = compose(dispersion())
    fig = mpl.to_figure(spec)
    ax, twin, top = fig.axes[0], fig.axes[1], fig.axes[2]
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log" and twin.get_yscale() == "log"
    s = spec.panels[0].axes.xsecondary if spec.panels else spec.axes.xsecondary
    assert top.get_xlabel() == "Frequency (THz)"
    np.testing.assert_allclose(top.get_xticks(), s.tickvals)
    assert [t.get_text() for t in top.get_xticklabels()] == list(s.ticklabels)
    assert mpl.render(spec, "png")[:4] == b"\x89PNG"
