"""A trend is drawn sqrt(2) wider than tall, legend or not (#188).

Eric, 2026-11-15: the demo's transmission against MMI length (an ``emode.Figure``
with a legend) showed at about 1:1. Measured in headless Chrome at 390 CSS
pixels: the plot box was 294 by 238 px (1.24:1), at 1280 px 956 by 706
(1.35:1). The tab sized the figure as the box's height plus the margins plus 30
px for the legend, but the legend sits inside the top margin, so Plotly gave
those 30 px to the box. Now the 30 px go into the top margin it sends Plotly.
"""

from __future__ import annotations

import math

import pytest

from emode.figures import Figure
from emode.ui.figure import compose, figure_aspect

from js_harness import needs_jsc, run_app


def test_a_one_panel_figure_asks_for_the_sqrt2_box():
    fig = Figure()
    p = fig.panel(0)
    p.line([1.0, 2.0], [0.1, 0.9], style="o-", label="both outputs")
    p.vline(1.5, style="--", label="L")
    assert figure_aspect(compose(fig)) == pytest.approx(1 / math.sqrt(2))


DRAW = """
var calls = [];
Plotly.react = function (el, data, layout) { calls.push(layout); };
function reply(legend) {
  return {id: "c1", kind: "figure", aspect: Math.SQRT1_2, menu: {}, controls: {},
    figure: {data: [], layout: {showlegend: legend, margin: {l: 60, r: 20, t: 60, b: 50}}}};
}
"""


@needs_jsc
@pytest.mark.parametrize("legend", [True, False])
def test_the_plot_box_is_sqrt2_wider_than_tall(legend):
    got = run_app(DRAW + f"draw(reply({'true' if legend else 'false'})); return calls[0];")
    m = got["margin"]
    box_w = got["width"] - m["l"] - m["r"]
    box_h = got["height"] - m["t"] - m["b"]
    assert box_w / box_h == pytest.approx(math.sqrt(2), rel=0.01)
    assert m["t"] == (90 if legend else 60)
