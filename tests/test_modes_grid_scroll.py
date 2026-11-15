"""An All modes grid taller than the screen can be scrolled to its last panel (#185).

Eric, 2026-11-15, on his phone: with 20 modes the grid showed the first 10 and
the rest could not be reached. Measured in headless Chrome at 390 by 844 CSS
pixels: the tab set the figure 3400 px tall, but under 800 px the stage is a
flex column, so ``#figure``'s ``flex: 1 1 auto`` shrank it back to the stage's
736 px and its ``overflow: hidden`` cut the rest off. The stage then had
nothing to scroll. Now the figure keeps its height there.

A plot taller than the stage is also drawn with Plotly's drag turned off:
with a drag mode, Plotly takes every touch on the plot for a zoom box
(``preventDefault`` on ``touchstart``), so a grid filling the screen could
not be scrolled by touch at all. A click on a panel still opens its mode.
"""

from __future__ import annotations

import re
from pathlib import Path

from js_harness import needs_jsc, run_app

CSS = Path(__file__).resolve().parents[1] / "emode" / "ui" / "static" / "emode.css"


def _rule(css: str, selector: str) -> str:
    found = re.search(r"(?:^|\n)\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert found, selector
    return found.group(1)


def test_the_figure_keeps_its_height_in_the_phones_column_stage():
    css = CSS.read_text()
    phone = css[css.index("@media (max-width: 800px)") :]
    rule = _rule(phone, "#figure")
    assert re.search(r"flex:\s*0 0 auto", rule) or re.search(r"flex-shrink:\s*0", rule), rule
    # the stage scrolls what it cannot show
    assert "overflow-y: auto" in _rule(css, "#stage")


#: A reply drawn on the stand-in page, whose stage is 400 by 600 CSS pixels.
DRAW = """
var calls = [];
Plotly.react = function (el, data, layout, config) { calls.push({layout: layout, config: config}); };
document.getElementById("stage").clientHeight = 600;
function reply(aspect) {
  return {id: "p1", kind: "plot", aspect: aspect, menu: {}, controls: {},
    figure: {data: [], layout: {dragmode: "zoom", margin: {l: 60, r: 20, t: 60, b: 50}}}};
}
"""


@needs_jsc
def test_a_plot_taller_than_the_stage_is_drawn_without_a_drag_mode():
    got = run_app(DRAW + "draw(reply(5)); return calls[0].layout;")
    assert got["height"] > 600
    assert got["dragmode"] is False


@needs_jsc
def test_a_plot_that_fits_keeps_its_zoom():
    got = run_app(DRAW + "draw(reply(0.7)); return calls[0].layout;")
    assert got["height"] <= 600
    assert got["dragmode"] == "zoom"
