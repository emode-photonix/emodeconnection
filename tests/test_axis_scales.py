"""``Panel.set(xscale=, yscale=)`` and ``Panel.secondary_x()``: a log axis and a second x axis.

The Dispersion plot draws wavelength on a log scale with the matching
frequency on a top axis. Both renderers read the same spec: matplotlib scales
and a twin x axes, Plotly a ``type='log'`` axis (whose range is in decades) and
an overlaying top axis whose ticks sit where the secondary values map back to x.
"""
from __future__ import annotations

import numpy as np
import pytest

import emode
from emode.types import ArgumentError
from emode.ui.figure import compose
from emode.ui.renderers.plotly import to_plotly

pytest.importorskip("matplotlib")
from emode.ui.renderers import mpl  # noqa: E402

C = 299792.458  # nm THz


def _fig(**kw):
    fig = emode.Figure()
    p = fig.panel(0)
    p.line([1000.0, 1500.0, 2000.0], [1.0, 10.0, 100.0], label="n")
    p.set(xlabel="Wavelength (nm)", ylabel="GVD", **kw)
    return fig, p


def test_scales_default_to_linear_and_are_checked():
    fig, p = _fig()
    assert compose(fig).panels[0].axes.xscale == "linear"
    for bad in ("ln", "logit"):
        with pytest.raises(ArgumentError) as info:
            p.set(xscale=bad)
        assert "xscale" in str(info.value)
    with pytest.raises(ArgumentError) as info:
        p.twin().set(yscale="sqrt")
    assert "yscale" in str(info.value)


def test_a_log_scale_reaches_the_spec_and_keeps_the_default_range_positive():
    fig, _ = _fig(xscale="log", yscale="log")
    a = compose(fig).panels[0].axes
    assert (a.xscale, a.yscale) == ("log", "log")
    assert 0 < a.xrange[0] <= 1000.0 and a.xrange[1] >= 2000.0
    assert 0 < a.yrange[0] <= 1.0 and a.yrange[1] >= 100.0


def test_matplotlib_and_plotly_both_draw_the_log_axes():
    fig, _ = _fig(xscale="log", yscale="log")
    spec = compose(fig)
    ax = mpl.to_figure(spec).axes[0]
    assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log")
    assert ax.get_xlim() == pytest.approx(spec.panels[0].axes.xrange)
    layout = to_plotly(spec)["layout"]
    assert layout["xaxis"]["type"] == "log" and layout["yaxis"]["type"] == "log"
    # Plotly gives a log axis's range in decades
    assert layout["xaxis"]["range"] == pytest.approx(list(np.log10(spec.panels[0].axes.xrange)))
    assert layout["yaxis"]["range"] == pytest.approx(list(np.log10(spec.panels[0].axes.yrange)))


def test_a_linear_axis_has_no_plotly_type_change():
    fig, _ = _fig()
    assert to_plotly(compose(fig))["layout"]["xaxis"].get("type", "linear") == "linear"


def test_the_twin_takes_its_own_scale():
    fig, p = _fig()
    right = p.twin()
    right.line([1000.0, 2000.0], [1.0, 1000.0], style="--", label="m")
    right.set(ylabel="m", yscale="log", ylim=(1.0, 1000.0))
    spec = compose(fig)
    assert spec.panels[0].axes.y2scale == "log" and spec.panels[0].axes.yscale == "linear"
    ax, twin = mpl.to_figure(spec).axes[:2]
    assert twin.get_yscale() == "log" and ax.get_yscale() == "linear"
    layout = to_plotly(spec)["layout"]
    assert layout["yaxis2"]["type"] == "log"
    assert layout["yaxis2"]["range"] == pytest.approx([0.0, 3.0])


def _freq(p, **kw):
    p.secondary_x(lambda x: C / np.asarray(x, float), label="Frequency (THz)", **kw)


def test_a_secondary_x_axis_puts_its_ticks_where_the_function_maps_back():
    fig, p = _fig(xscale="log")
    _freq(p)
    a = compose(fig).panels[0].axes
    assert a.x2label == "Frequency (THz)"
    assert len(a.x2ticks) >= 2
    for position, text in a.x2ticks:
        assert a.xrange[0] <= position <= a.xrange[1]
        # the label is the function's value at that x, to the digits shown
        assert float(text) == pytest.approx(C / position, rel=0.02)


def test_explicit_secondary_ticks_are_used_with_a_given_inverse():
    fig, p = _fig()
    _freq(p, inverse=lambda f: C / np.asarray(f, float), ticks=[150, 200])
    a = compose(fig).panels[0].axes
    assert [t for _, t in a.x2ticks] == ["150", "200"]
    assert [x for x, _ in a.x2ticks] == pytest.approx([C / 150, C / 200])


def test_without_an_inverse_it_is_found_from_the_function_when_monotonic():
    fig, p = _fig()
    _freq(p, ticks=[150, 200])
    assert [x for x, _ in compose(fig).panels[0].axes.x2ticks] == pytest.approx([C / 150, C / 200], rel=1e-3)


def test_matplotlib_draws_the_secondary_x_axis_on_top():
    fig, p = _fig(xscale="log")
    _freq(p)
    spec = compose(fig)
    axes = mpl.to_figure(spec).axes
    top = [a for a in axes if a.get_xlabel() == "Frequency (THz)"]
    assert len(top) == 1
    (top,) = top
    a = spec.panels[0].axes
    assert list(top.get_xticks()) == pytest.approx([x for x, _ in a.x2ticks])
    assert [t.get_text() for t in top.get_xticklabels()] == [t for _, t in a.x2ticks]
    assert top.get_xscale() == "log" and top.get_xlim() == pytest.approx(a.xrange)
    assert top.xaxis.get_ticks_position() in ("top", "default")
    assert mpl.render(spec, "png")[:4] == b"\x89PNG"


def test_plotly_draws_the_secondary_x_axis_overlaying_on_top():
    fig, p = _fig(xscale="log")
    _freq(p)
    spec = compose(fig)
    layout = to_plotly(spec)["layout"]
    a = spec.panels[0].axes
    top = layout["xaxis2"]
    assert top["overlaying"] == "x" and top["side"] == "top"
    assert top["title"]["text"] == "Frequency (THz)"
    assert top["tickvals"] == pytest.approx([x for x, _ in a.x2ticks])
    assert top["ticktext"] == [t for _, t in a.x2ticks]
    assert top["type"] == "log" and top["range"] == layout["xaxis"]["range"]
    # an axis nothing is drawn against is not drawn, so a hidden trace sits on it
    assert [t for t in to_plotly(spec)["data"] if t.get("xaxis") == "x2"]


def test_a_secondary_x_axis_of_a_second_panel_gets_its_own_axis():
    fig = emode.Figure(cols=2)
    for k in (0, 1):
        q = fig.panel(k)
        q.line([1000.0, 2000.0], [0.0, 1.0])
        q.secondary_x(lambda x: C / np.asarray(x, float), label="THz")
    layout = to_plotly(compose(fig))["layout"]
    tops = [v for k, v in layout.items() if k.startswith("xaxis") and v.get("side") == "top"]
    assert len(tops) == 2
    assert {v["overlaying"] for v in tops} == {"x", "x2"}


def test_a_panel_without_a_secondary_x_axis_is_unchanged():
    fig, _ = _fig()
    a = compose(fig).panels[0].axes
    assert a.x2label is None and a.x2ticks == ()
    assert "xaxis2" not in to_plotly(compose(fig))["layout"]
