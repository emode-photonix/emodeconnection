"""Series colors are assigned in order, never cycle, and fold into one "Other" in Mist after 8.

The brand rules for a curve plot (``colormaps.series_color``): slot 1, 2, 3 and so on,
at most 8, every extra series drawn in Mist with a single legend entry, and only the
first 3 colors for marker-only series (scatter plots), where any two colors can touch.
"""
from __future__ import annotations

import numpy as np
import pytest

import emode
from emode import colormaps
from emode.ui.figure import compose
from emode.ui.gds import GDSPanel
from emode.ui.renderers.plotly import to_plotly

from gds_fixtures import gds_view, layer, _rect

mpl = pytest.importorskip("matplotlib")
from emode.ui.renderers import mpl as mpl_renderer  # noqa: E402


def ten_curves(style: str = "-") -> "emode.Figure":
    fig = emode.Figure()
    panel = fig.panel(0)
    x = np.linspace(0.0, 1.0, 5)
    for i in range(10):
        panel.line(x, x + i, style=style, label=f"curve {i}")
    return fig


def test_ten_curves_take_the_8_colors_in_order_then_mist():
    (spec,) = compose(ten_curves()).panels
    colors = [line.color for line in spec.lines]
    assert colors == list(colormaps.SERIES) + [colormaps.BRAND["Mist"]] * 2


def test_the_folded_curves_share_one_other_legend_entry():
    (spec,) = compose(ten_curves()).panels
    assert [line.label for line in spec.lines] == [f"curve {i}" for i in range(8)] + ["Other", None]


def test_matplotlib_and_plotly_show_one_other_entry():
    spec = compose(ten_curves())
    (panel,) = spec.panels
    fig = mpl_renderer.to_figure(panel)
    texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert texts == [f"curve {i}" for i in range(8)] + ["Other"]
    named = [t["name"] for t in to_plotly(panel)["data"] if t.get("showlegend") is not False and t.get("name")]
    assert named.count("Other") == 1


def test_marker_only_series_use_the_first_3_colors_then_mist():
    fig = emode.Figure()
    panel = fig.panel(0)
    for i in range(5):
        panel.line([0.0, 1.0], [0.0, float(i)], style="o", label=f"s{i}")
    (spec,) = compose(fig).panels
    lines = colormaps.SERIES[:3]
    assert [m.color for m in spec.lines] == list(lines) + [colormaps.OTHER] * 2


def test_a_series_keeps_its_color_whatever_the_others():
    fig = emode.Figure()
    panel = fig.panel(0)
    panel.line([0, 1], [0, 1], color=3, label="pinned")
    (spec,) = compose(fig).panels
    assert spec.lines[0].color == colormaps.SERIES[3]


def test_gds_layers_past_8_are_mist_and_never_cycle():
    view = gds_view()
    view.layers[:] = [layer(n, 0, [_rect(0, 0, 10, 10)]) for n in range(1, 11)]
    spec = compose(GDSPanel.from_view(view))
    colors = [s.color for s in spec.layers if s.key != "shapes"]
    assert colors == list(colormaps.SERIES) + [colormaps.OTHER] * 2


# ------------------------------------------------------------------ text is never in a series or status color

TEXT_COLORS = {
    colormaps.GUI["text"].lower(),
    colormaps.GUI["text_secondary"].lower(),
    colormaps.PLOT["dark"]["text"].lower(),  # Haze, for labels drawn on a field map
}


def _text_colors(spec) -> set:
    return {t.color.lower() for t in spec.texts}


def test_schematic_labels_are_text_colors():
    from plot_fixtures import schematic

    spec = compose(schematic())
    assert spec.texts and _text_colors(spec) <= TEXT_COLORS
    # the monitor mark keeps its series color
    assert colormaps.schematic_colors()["monitor"] == colormaps.SERIES[2]


def test_overlay_labels_are_haze_while_their_lines_keep_their_slots():
    from plot_fixtures import zx_view

    spec = compose(zx_view())
    labels = [t for t in spec.texts if t.kind in ("port", "facet", "monitor")]
    assert labels, "the fixture has no overlay label"
    assert {t.color.lower() for t in labels} == {colormaps.PLOT["dark"]["text"].lower()}
    assert {line.color for line in spec.lines if line.kind == "facet"} == {colormaps.overlay_colors()["facet"]}


def test_every_text_of_a_composed_figure_and_its_plotly_form_is_a_text_color():
    from plot_fixtures import schematic, xy_view, zx_view

    for make in (xy_view, zx_view, schematic):
        spec = compose(make())
        assert _text_colors(spec) <= TEXT_COLORS
        figure = to_plotly(spec)
        for a in figure["layout"]["annotations"]:
            color = a.get("font", {}).get("color")
            assert color is None or color.lower() in TEXT_COLORS | {colormaps.GUI["primary"].lower()}
        assert figure["layout"]["font"]["color"].lower() in TEXT_COLORS


def test_a_twin_axis_label_is_not_drawn_in_a_series_color():
    fig = emode.Figure()
    p = fig.panel(0)
    p.line([0, 1], [0, 1])
    p.twin().line([0, 1], [1, 0])
    (spec,) = compose(fig).panels
    assert spec.axes.ylabel_color is None and spec.axes.y2label_color is None
