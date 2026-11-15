"""The contour marks show on the colorbar in a real browser (#186).

#178 placed the marks right and they still never showed: Plotly drew them under
the bar. So this test draws the tab's figures with the tab's own Plotly
(``emode/ui/static/plotly-cartesian.min.js``) in headless Chrome, takes a
screenshot, and looks at the pixels. Each scenario is drawn twice, once as the
tab receives it and once with the marks' clear colorbar taken out of the last
figure. The two images differ only across the bar, and at every level.

The scenarios follow what the tab does: a first plot, a mode switch and a
profile change (``Plotly.react`` with a new figure), linear to log and back, the
All modes grid (no colorbar) and back, and a z panel with the x-y field below it.

Chrome is found on PATH, at ``EMODE_TEST_CHROME``, as Google Chrome on macOS, or
in Playwright's cache. Without one the test skips.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from emode.ui.figure import FigureSpec, compose
from emode.ui.renderers.plotly import to_plotly
from emode.ui.server import _json_default

from test_colorbar_marks import _log_view, _view

PLOTLY = (
    Path(__file__).resolve().parents[1]
    / "emode"
    / "ui"
    / "static"
    / "plotly-cartesian.min.js"
)

#: Each plot's size in CSS pixels, and the page's.
PW, PH = 600, 420
GAP = 20


def _chrome() -> Optional[str]:
    candidates = [os.environ.get("EMODE_TEST_CHROME") or ""]
    candidates += [
        shutil.which(n) or ""
        for n in (
            "chrome-headless-shell",
            "chromium",
            "chromium-browser",
            "google-chrome",
            "chrome",
        )
    ]
    candidates += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    home = Path.home()
    for cache in (
        home / "Library" / "Caches" / "ms-playwright",
        home / ".cache" / "ms-playwright",
    ):
        candidates += sorted(
            glob.glob(
                str(cache / "chromium_headless_shell-*" / "*" / "chrome-headless-shell")
            ),
            reverse=True,
        )
        candidates += sorted(
            glob.glob(
                str(
                    cache
                    / "chromium-*"
                    / "*"
                    / "Chromium.app"
                    / "Contents"
                    / "MacOS"
                    / "Chromium"
                )
            ),
            reverse=True,
        )
        candidates += sorted(
            glob.glob(str(cache / "chromium-*" / "chrome-linux" / "chrome")),
            reverse=True,
        )
    return next((c for c in candidates if c and os.access(c, os.X_OK)), None)


CHROME = _chrome()
pytestmark = pytest.mark.skipif(CHROME is None, reason="needs a headless Chrome")


def _grid() -> Dict[str, Any]:
    """An All modes grid: small panels, no colorbar."""
    one = replace(compose(_view()), colorbar=None)
    return to_plotly(FigureSpec(axes=one.axes, panels=(one,) * 4, panel_grid=(2, 2)))


def _scenarios() -> Dict[str, List[Dict[str, Any]]]:
    signed = to_plotly(compose(_view()))
    positive = to_plotly(compose(_view(peak=0.6, signed=False)))
    log = to_plotly(compose(_log_view()))
    one = compose(_view())
    cut = to_plotly(
        FigureSpec(
            axes=one.axes, panels=(one, compose(_view(signed=False))), panel_grid=(2, 1)
        )
    )
    return {
        "first plot": [signed],
        "mode switch": [signed, positive],
        "linear to log": [positive, log],
        "log to linear": [log, signed],
        "grid and back": [signed, _grid(), positive],
        "cut panels": [signed, cut],
    }


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<script src="{plotly}"></script></head>
<body style="margin:0;background:#fff">{divs}<pre id="out"></pre>
<script>
var SCENARIOS = {scenarios};
var BARE = {bare};
var out = [];
SCENARIOS.forEach(function (figs, k) {{
  var gd = document.getElementById("p" + k);
  figs.forEach(function (fig, i) {{
    var data = fig.data;
    if (BARE && i === figs.length - 1) data = data.filter(function (t) {{ return t.meta !== "colorbar_levels"; }});
    var layout = Object.assign({{}}, fig.layout, {{width: {pw}, height: {ph}}});
    if (i === 0) Plotly.newPlot(gd, data, layout); else Plotly.react(gd, data, layout);
  }});
  // where each clear bar's ticks belong, in page pixels, from Plotly's own plot area
  var size = gd._fullLayout._size, top = gd.offsetTop, left = gd.offsetLeft, bars = [];
  gd.data.forEach(function (t) {{
    if (t.meta !== "colorbar_levels") return;
    var cb = t.marker.colorbar, lo = t.marker.cmin, hi = t.marker.cmax;
    var cy = cb.y == null ? 0.5 : cb.y, clen = cb.len == null ? 1 : cb.len;
    var bottom = cy - clen / 2;
    var x0 = left + size.l + cb.x * size.w;
    var row = function (v) {{ return top + size.t + (1 - (bottom + (v - lo) / (hi - lo) * clen)) * size.h; }};
    bars.push({{x0: x0, x1: x0 + cb.thickness, ends: [row(hi), row(lo)], rows: cb.tickvals.map(row)}});
  }});
  out.push(bars);
}});
document.getElementById("out").textContent = "MARKS" + JSON.stringify(out) + "MARKS";
</script></body></html>"""


def _page(path: Path, scenarios: List[List[Dict[str, Any]]], bare: bool) -> None:
    divs = "".join(
        f'<div id="p{k}" style="position:absolute;left:0;top:{k * (PH + GAP)}px;width:{PW}px;height:{PH}px"></div>'
        for k in range(len(scenarios))
    )
    text = PAGE.format(
        plotly=PLOTLY.as_uri(),
        divs=divs,
        scenarios=json.dumps(scenarios, default=_json_default),
        bare="true" if bare else "false",
        pw=PW,
        ph=PH,
    )
    path.write_text(text, encoding="utf-8")


def _run(*args: str) -> str:
    assert CHROME is not None
    done = subprocess.run(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            "--allow-file-access-from-files",
            "--virtual-time-budget=5000",
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return done.stdout


def _shot(page: Path, png: Path, height: int) -> np.ndarray:
    import matplotlib.image

    _run(f"--window-size={PW + 40},{height}", f"--screenshot={png}", page.as_uri())
    assert png.exists(), "Chrome made no screenshot"
    return np.asarray(matplotlib.image.imread(str(png)))[..., :3] * 255.0


def test_every_level_shows_across_the_bar_and_nothing_else_changes(
    tmp_path: Path,
) -> None:
    named = _scenarios()
    scenarios = list(named.values())
    height = len(scenarios) * (PH + GAP) + 40
    _page(tmp_path / "marked.html", scenarios, bare=False)
    _page(tmp_path / "bare.html", scenarios, bare=True)
    dom = _run(
        f"--window-size={PW + 40},{height}",
        "--dump-dom",
        (tmp_path / "marked.html").as_uri(),
    )
    found = re.search(r"MARKS(.*)MARKS", dom)
    assert found, "the page did not draw: " + dom[-400:]
    marks = json.loads(found.group(1).replace("&quot;", '"'))
    marked = _shot(tmp_path / "marked.html", tmp_path / "marked.png", height)
    bare = _shot(tmp_path / "bare.html", tmp_path / "bare.png", height)
    diff = np.abs(marked - bare).max(axis=2)
    expected = np.zeros(diff.shape, bool)
    for name, bars in zip(named, marks):
        assert bars, f"{name}: no clear bar of marks"
        for bar in bars:
            cols = slice(int(np.ceil(bar["x0"])) + 2, int(np.floor(bar["x1"])) - 2)
            assert cols.stop > cols.start
            # a level within 2 px of an end lies on the bar's own outline, which looks the same
            for y in [
                r for r in bar["rows"] if min(abs(r - e) for e in bar["ends"]) > 2
            ]:
                rows = slice(int(np.floor(y)) - 1, int(np.ceil(y)) + 2)
                seen = diff[rows, cols].max(axis=0)
                assert (seen > 40).mean() > 0.6, (
                    f"{name}: the mark at y = {y:.1f} px is not seen across the bar"
                )
                # a black mark: darker than the bar under it
                assert (
                    marked[rows, cols].sum(axis=2).min()
                    < bare[rows, cols].sum(axis=2).min() + 1
                )
                expected[
                    rows.start - 1 : rows.stop + 1,
                    int(bar["x0"]) - 2 : int(bar["x1"]) + 3,
                ] = True
    assert not (diff[~expected] > 40).any(), (
        "the clear bar changed pixels off its marks"
    )
