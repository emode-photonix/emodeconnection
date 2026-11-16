"""Home returns to the range of the figure on screen, not of the first one drawn (#201).

Eric, 2026-11-15, on his phone: with the window display on, Home and then Autoscale
showed the hatch on the wrong half. Plotly's Reset axes goes back to the range the plot
was first drawn with, and ``Plotly.react`` keeps that first range when a later figure
has another (the window display widens the range, a mode or plane switch changes it).
So the tab drew a figure with the window display on, and Home went back to the range of
the figure before it. The tab now draws a figure whose ranges differ from the one on
screen with ``Plotly.newPlot``, which takes the new ranges as the ones Home returns to.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

import pytest

from chrome_cdp import Page
from test_colorbar_render import CHROME
from test_phone_controls import _cut_server


def _reply(yrange: list) -> Dict[str, Any]:
    return {
        "id": "p1",
        "kind": "plot",
        "title": "x-y Ex",
        "aspect": 0.7,
        "menu": {"data": ["Ex"], "selected": {"data": "Ex"}},
        "controls": {},
        "extra": {"menus": []},
        "figure": {
            "data": [{"type": "scatter", "x": [0, 1], "y": [0, 1]}],
            "layout": {"xaxis": {"range": [-10, 10]}, "yaxis": {"range": yrange}},
        },
    }


@pytest.mark.skipif(CHROME is None, reason="needs a headless Chrome")
@pytest.mark.parametrize("size", [(390, 844), (1280, 900)], ids=["phone", "desktop"])
def test_home_returns_to_the_range_of_the_figure_on_screen(size):
    server, _ = _cut_server()
    page = Page(
        CHROME, f"http://127.0.0.1:{server.port}/?token={server.token}#/plots/p1", *size
    )
    try:
        page.wait_for("document.querySelector('#figure .modebar-btn') != null")
        for yrange in ([0, 2000], [-500, 700]):
            reply = json.dumps(_reply(yrange))
            page.eval(f"(function(){{var r={reply}; state.reply=r; draw(r); return 1;}})()")
            time.sleep(0.6)
        page.eval(
            "document.querySelector('.modebar-btn[data-title=\"Reset axes\"]').click()"
        )
        time.sleep(0.6)
        got = page.eval("JSON.stringify(document.getElementById('figure').layout.yaxis.range)")
        assert json.loads(got) == [-500, 700]
    finally:
        page.close()
        server.stop()
