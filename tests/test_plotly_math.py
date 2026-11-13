"""Math in labels drawn by the browser tab (emodeconnection#145).

The tab's bundled plotly has no MathJax, so a label such as ``$n_{eff}$``
showed its dollar signs. The plotly renderer turns the simple math EMode and
the examples write (subscripts, superscripts, Greek letters, ``\\mathrm``)
into plotly's HTML subset. The matplotlib renderer keeps its mathtext.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from emode.figures import Figure
from emode.ui.figure import compose
from emode.ui.renderers.plotly import math_html, to_plotly

from plot_fixtures import index_view, layout_view, schematic, xy_view, zx_view


@pytest.mark.parametrize(
    ("label", "html"),
    [
        # every math label EMode sends today
        ("$n_\\mathrm{eff} = $2.400", "n<sub>eff</sub> = 2.400"),
        ("|Ex|$^2$", "|Ex|<sup>2</sup>"),
        ("10log$_{10}$(|Ex|)", "10log<sub>10</sub>(|Ex|)"),
        # what the examples and users write
        ("$n_{eff}$", "n<sub>eff</sub>"),
        ("$\\lambda$ (nm)", "λ (nm)"),
        ("$\\Delta n$", "Δn"),
        ("GVD (ps/nm/km)", "GVD (ps/nm/km)"),
        ("$\\chi^{(2)}$ (pm/V)", "χ<sup>(2)</sup> (pm/V)"),
        ("$\\beta_2$ (ps$^2$/km)", "β<sub>2</sub> (ps<sup>2</sup>/km)"),
        ("$|E_x|^2$", "|E<sub>x</sub>|<sup>2</sup>"),
        ("$\\mu$m", "μm"),
        ("$\\epsilon_r$", "ε<sub>r</sub>"),
        ("$\\text{T}_{\\mathrm{TE}}$", "T<sub>TE</sub>"),
        ("$10^{-3}$", "10<sup>-3</sup>"),
        ("cost \\$5", "cost $5"),
        ("a < b", "a &lt; b"),
    ],
)
def test_simple_math_becomes_plotly_html(label: str, html: str) -> None:
    assert math_html(label) == html


def test_text_without_math_is_unchanged() -> None:
    assert math_html("Re(Ex), 0") == "Re(Ex), 0"
    assert math_html("line one\nline two") == "line one\nline two"


def _strings(value) -> list:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() if k not in ("bdata",) for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


@pytest.mark.parametrize("make", [xy_view, zx_view, index_view, layout_view, schematic])
def test_no_label_of_a_frozen_view_shows_a_dollar_sign(make) -> None:
    figure = to_plotly(compose(make()))
    assert not [s for s in _strings(figure) if "$" in s]


def test_the_frozen_n_eff_label_is_drawn_as_html() -> None:
    figure = to_plotly(compose(xy_view()))
    texts = [a["text"] for a in figure["layout"]["annotations"]]
    assert "n<sub>eff</sub> = 2.400" in texts


def test_axis_colorbar_and_trace_labels_of_a_figure_are_translated() -> None:
    fig = Figure(title="$\\Delta n$")
    panel = fig.panel()
    panel.line(np.array([1.0, 2.0]), np.array([1.0, 2.0]), label="$n_{eff}$")
    panel.set(xlabel="$\\lambda$ (nm)", ylabel="$\\beta_2$", title="$\\chi^{(2)}$")
    text = json.dumps(to_plotly(compose(fig)), ensure_ascii=False)
    assert "$" not in text
    assert "λ (nm)" in text and "β<sub>2</sub>" in text and "n<sub>eff</sub>" in text


def test_the_matplotlib_renderer_keeps_mathtext() -> None:
    spec = compose(xy_view())
    assert any(t.text == "$n_\\mathrm{eff} = $2.400" for t in spec.texts)
