"""The far field panel's annotation stays on the image (#209).

The annotation (divergence and M², or the screen's distance and power) is white
text in the panel's top left corner. It was placed at a fraction of the axes,
but the equal aspect ratio widens one axis range past the image whenever the
plot box's shape differs from the data's, so the text fell on the white
background beside the image and read as cut off ("ergence 12.50°"). It is now
placed from the image's own extent, inside the axis ranges.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from emode.ui.figure import FarFieldPanel, PlotOptions, compose
from emode.ui.renderers.plotly import to_plotly
from emode.ui.server import _json_default

from plot_fixtures import farfield
from test_colorbar_render import CHROME, PLOTLY, _run

PANELS = {
    "intensity": lambda: FarFieldPanel(farfield()),
    "screen": lambda: FarFieldPanel(farfield(screen=True), "screen"),
}


def _image_box(raster):
    """The image's extent in data units: half a cell past the first and last samples."""
    hx = (raster.x[1] - raster.x[0]) / 2
    hy = (raster.y[1] - raster.y[0]) / 2
    return raster.x[0] - hx, raster.x[-1] + hx, raster.y[0] - hy, raster.y[-1] + hy


@pytest.mark.parametrize("panel", PANELS)
def test_the_annotation_is_placed_on_the_image_inside_the_axis_ranges(panel):
    spec = compose(PANELS[panel](), PlotOptions())
    (text,) = [t for t in spec.texts if t.kind == "annotation"]
    assert text.frame == "data"
    x0, x1, y0, y1 = _image_box(spec.rasters[0])
    assert max(x0, spec.axes.xrange[0]) < text.x < min(x1, spec.axes.xrange[1])
    assert max(y0, spec.axes.yrange[0]) < text.y < min(y1, spec.axes.yrange[1])
    # in the top left corner
    assert text.x < (x0 + x1) / 2 and text.y > (y0 + y1) / 2
    assert (text.ha, text.va) == ("left", "top")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><script src="{plotly}"></script></head>
<body style="margin:0;background:#fff"><div id="p"></div><pre id="out"></pre><script>
var fig = {figure}, gd = document.getElementById("p");
Plotly.newPlot(gd, fig.data, Object.assign({{}}, fig.layout, {{width: {w}, height: {h}}}));
var box = function (e) {{ var b = e.getBoundingClientRect(); return [b.left, b.top, b.right, b.bottom]; }};
var size = gd._fullLayout._size;
document.getElementById("out").textContent = "OUT" + JSON.stringify({{
  text: box(gd.querySelector(".annotation-text-g text")),
  image: box(gd.querySelector(".hm image")),
  plot: [size.l, size.t, size.l + size.w, size.t + size.h]
}}) + "OUT";
</script></body></html>"""


@pytest.mark.skipif(CHROME is None, reason="needs a headless Chrome")
@pytest.mark.parametrize("panel", PANELS)
@pytest.mark.parametrize("size", [(960, 610), (500, 400)])
def test_the_annotation_is_drawn_on_the_image_in_a_browser(tmp_path: Path, panel, size):
    """Wider than the data (the x range widens past the image) and narrower (the image
    overhangs the x range): the drawn text lies on the drawn image, inside the plot box."""
    w, h = size
    figure = to_plotly(compose(PANELS[panel](), PlotOptions()))
    page = tmp_path / "page.html"
    page.write_text(
        PAGE.format(
            plotly=PLOTLY.as_uri(),
            figure=json.dumps(figure, default=_json_default),
            w=w,
            h=h,
        ),
        encoding="utf-8",
    )
    dom = _run(f"--window-size={w + 40},{h + 200}", "--dump-dom", page.as_uri())
    found = re.search(r"OUT(.*)OUT", dom)
    assert found, "the page did not draw: " + dom[-400:]
    boxes = json.loads(found.group(1).replace("&quot;", '"'))
    text, image, plot = boxes["text"], boxes["image"], boxes["plot"]
    left, top = max(image[0], plot[0]), max(image[1], plot[1])
    right, bottom = min(image[2], plot[2]), min(image[3], plot[3])
    assert left <= text[0] and text[2] <= right, (text, image, plot)
    assert top <= text[1] and text[3] <= bottom, (text, image, plot)
