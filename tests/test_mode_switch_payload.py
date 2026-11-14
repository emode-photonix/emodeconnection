"""A mode switch sends only what the tab draws, and the All modes view shows every mode at once.

Measured on the EME taper example (a 200 x 222 x-y grid, 24 contour levels):
the reply to one mode switch was 6.0 MB, 96 % of it the field sent again in
each of 24 Plotly contour traces, which the browser then contoured 24 times.
The contours now go as their lines, computed here from the full-resolution
field, the field goes once, rounded to 4 significant digits, and a raster
finer than the panel's pixels is decimated to them.
"""

from __future__ import annotations

import base64
import http.client
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from emode.plotting import Label, Raster
from emode.ui.figure import PlotOptions, compose
from emode.ui.modes import ALL_MODES, columns_for
from emode.ui.renderers.plotly import to_plotly
from emode.ui.server import UIServer, _json_default
from emode.ui.views import ViewRegistry, menu_of

from plot_fixtures import xy_view, zx_view

#: The levels EMode contours a signed field at (``plots/utils.py`` lpos and lneg).
LEVELS = np.array(
    sorted(
        [
            0.001,
            0.01,
            *np.linspace(0.1, 0.9, 9),
            0.99,
            -0.99,
            *np.linspace(-0.9, -0.1, 9),
            -0.01,
            -0.001,
        ]
    )
)

MODES = [f"{k}, TE" for k in range(10)]


def mode_view(label: str = "0, TE", nx: int = 200, ny: int = 222) -> Any:
    """An x-y panel of one mode on the taper example's grid, with EMode's 24 contour levels."""
    k = int(label.split(",")[0])
    x = np.linspace(-2000.0, 2000.0, nx)
    y = np.linspace(-1110.0, 1110.0, ny)
    X, Y = np.meshgrid(x, y)
    z = np.cos((k + 1) * X / 400) * np.exp(-((X / 900) ** 2) - (Y / 300) ** 2)
    z = (z / np.abs(z).max()).astype(np.float32)
    base = xy_view()
    menu = base.menu.model_copy(
        update={"modes": list(MODES), "selected": {**base.menu.selected, "mode": label}}
    )
    return base.model_copy(
        update={
            "rasters": [
                Raster(
                    x=x,
                    y=y,
                    z=z,
                    kind="diverging",
                    contour_levels=LEVELS,
                    interpolation="bilinear",
                )
            ],
            "texts": [
                Label(text=f"Re(Ex), {k}", x=0.04, y=1.02, kind="title"),
                Label(
                    text=f"$n_\\mathrm{{eff}} = $2.{400 - k:03d}",
                    x=0.96,
                    y=1.02,
                    ha="right",
                    kind="neff",
                ),
            ],
            "menu": menu,
        }
    )


def _body(figure: Dict[str, Any]) -> bytes:
    return json.dumps({"figure": figure}, default=_json_default).encode()


def _decode(arr: Dict[str, Any]) -> np.ndarray:
    values = np.frombuffer(base64.b64decode(arr["bdata"]), dtype=np.dtype(arr["dtype"]))
    if "shape" in arr:
        values = values.reshape([int(n) for n in arr["shape"].split(",")])
    return values


def _heatmaps(figure: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [t for t in figure["data"] if t["type"] == "heatmap"]


def _contours(figure: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        t
        for t in figure["data"]
        if t.get("meta") == "contour" or t["type"] == "contour"
    ]


# ------------------------------------------------------------------ one panel


def test_the_contours_do_not_send_the_field_again():
    figure = to_plotly(compose(mode_view()))
    (heatmap,) = _heatmaps(figure)
    assert all("z" not in t for t in figure["data"] if t is not heatmap)
    assert len(_body(figure)) < 1.6 * len(json.dumps(heatmap))


def test_the_contours_are_lines_at_the_levels_on_each_side_of_zero():
    view = mode_view()
    lines = _contours(to_plotly(compose(view)))
    assert lines and all(t["type"] == "scatter" for t in lines)
    assert {t["line"]["color"] for t in lines} == {"#000000", "#ffffff"}
    r = view.rasters[0]
    for t in lines:
        x, y = _decode(t["x"]), _decode(t["y"])
        ok = np.isfinite(x)
        assert ok.sum() > 100
        # every point lies on the field's level set: interpolate the field there
        ix = np.interp(x[ok], r.x, np.arange(len(r.x)))
        iy = np.interp(y[ok], r.y, np.arange(len(r.y)))
        z = np.asarray(r.z)[np.round(iy).astype(int), np.round(ix).astype(int)]
        positive = t["line"]["color"] == "#000000"
        assert (z > -0.05).all() if positive else (z < 0.05).all()


def test_the_contours_are_off_when_turned_off():
    assert not _contours(to_plotly(compose(mode_view(), PlotOptions(contours=False))))


def test_the_field_is_rounded_to_four_significant_digits():
    view = mode_view()
    (heatmap,) = _heatmaps(to_plotly(compose(view)))
    sent = _decode(heatmap["z"])
    raw = np.asarray(view.rasters[0].z)
    assert np.allclose(sent, raw, rtol=2e-4, atol=0)
    # the bits past the fourth digit are gone, so the bytes compress
    bits = sent.view(np.uint32) & np.uint32(0x1FF)
    assert not bits.any()


def test_a_raster_finer_than_the_panel_is_decimated_to_its_pixels():
    view = mode_view(nx=1600, ny=1200)
    (heatmap,) = _heatmaps(to_plotly(compose(view), pixels=(800, 500)))
    ny, nx = _decode(heatmap["z"]).shape
    assert nx <= 800 and ny <= 500
    assert nx >= 400 and ny >= 250
    assert _decode(heatmap["x"]).size == nx and _decode(heatmap["y"]).size == ny


def test_a_raster_coarser_than_the_panel_is_sent_whole():
    view = mode_view()
    (heatmap,) = _heatmaps(to_plotly(compose(view), pixels=(1200, 750)))
    assert _decode(heatmap["z"]).shape == (222, 200)


# ------------------------------------------------------------------ the server


def _server(views: Optional[Dict[str, Any]] = None) -> UIServer:
    registry = ViewRegistry()
    registry.add(mode_view("0, TE"), {"plane": "x-y"}, PlotOptions(), (1200, 750))
    fetched: List[Tuple[Dict[str, Any], Tuple[int, int]]] = []

    def fetch(view: Any, arguments: Dict[str, Any], pixels: Tuple[int, int]) -> Any:
        fetched.append((dict(arguments), tuple(pixels)))
        return mode_view(str(arguments.get("mode", "0, TE")))

    server = UIServer(registry, fetch).start()
    server.fetched = fetched  # type: ignore[attr-defined]
    return server


@pytest.fixture
def server():
    s = _server()
    yield s
    s.stop()


def post(
    server: UIServer, controls: Dict[str, Any], **extra: Any
) -> Tuple[Dict[str, Any], int]:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    body = json.dumps({"controls": controls, "pixels": [1170, 720], **extra}).encode()
    headers = {
        "Host": f"127.0.0.1:{server.port}",
        "X-EMode-Token": server.token,
        "Content-Type": "application/json",
    }
    conn.request("POST", "/api/views/p1/data", body=body, headers=headers)
    payload = conn.getresponse().read()
    conn.close()
    return json.loads(payload), len(payload)


def test_a_mode_switch_reply_stays_small(server):
    reply, size = post(server, {"mode": "3, TE"})
    assert server.fetched[-1][0]["mode"] == "3, TE"  # type: ignore[attr-defined]
    assert "error" not in reply
    assert size < 600_000


def test_the_mode_menu_offers_all_modes_on_an_x_y_panel_of_several_modes():
    assert menu_of(mode_view()).modes[-1] == ALL_MODES
    assert ALL_MODES not in (menu_of(zx_view()).modes or [])
    one = mode_view().model_copy(
        update={"menu": mode_view().menu.model_copy(update={"modes": ["0, TE"]})}
    )
    assert ALL_MODES not in menu_of(one).modes


@pytest.mark.parametrize("width, columns", [(390, 2), (760, 3), (1400, 4)])
def test_the_grid_has_two_columns_on_a_phone_and_more_on_a_desktop(width, columns):
    assert columns_for(width, 10) == columns
    assert columns_for(width, 1) == 1


def _panel_titles(figure: Dict[str, Any]) -> List[str]:
    return [
        a["text"]
        for a in figure["layout"]["annotations"]
        if a.get("name") == "panel_title"
    ]


def test_all_modes_shows_every_mode_in_a_grid_of_titled_clickable_panels(server):
    reply, size = post(server, {"mode": ALL_MODES}, width=390)
    assert "error" not in reply, reply.get("error")
    asked = [a.get("mode") for a, _ in server.fetched]  # type: ignore[attr-defined]
    assert sorted(asked) == sorted(MODES[1:]), (
        "each mode is fetched once, the shown one is kept"
    )
    figure = reply["figure"]
    heatmaps = _heatmaps(figure)
    assert [t["meta"] for t in heatmaps] == [f"mode:{m}" for m in MODES]
    titles = _panel_titles(figure)
    assert len(titles) == 10
    assert all("n<sub>eff</sub>" in t for t in titles), titles
    assert titles[3].startswith("3")
    # two columns on a phone: the second panel's axes sit beside the first
    layout = figure["layout"]
    assert layout["xaxis2"]["domain"][0] > layout["xaxis"]["domain"][1]
    assert layout["yaxis3"]["domain"][1] < layout["yaxis"]["domain"][0]
    assert reply["menu"]["selected"]["mode"] == ALL_MODES
    assert reply["controls"]["all_modes"] is True
    # decimated to the small panels: ten modes in well under the size of ten single replies
    assert size < 1_500_000
    for t in heatmaps:
        ny, nx = _decode(t["z"]).shape
        assert nx <= 200 and ny <= 222


def test_a_click_on_a_panel_opens_its_mode_without_asking_emode_again(server):
    post(server, {"mode": ALL_MODES}, width=1400)
    count = len(server.fetched)  # type: ignore[attr-defined]
    reply, _ = post(server, {"mode": "4, TE"}, width=1400)
    assert len(server.fetched) == count  # type: ignore[attr-defined]
    assert reply["controls"]["all_modes"] is False
    assert len(_heatmaps(reply["figure"])) == 1
    assert reply["menu"]["selected"]["mode"] == "4, TE"
    assert "xaxis2" not in reply["figure"]["layout"]


def test_all_modes_follows_a_component_change(server):
    post(server, {"mode": ALL_MODES}, width=1400)
    server.fetched.clear()  # type: ignore[attr-defined]
    reply, _ = post(server, {"component": "Ey"}, width=1400)
    asked = server.fetched  # type: ignore[attr-defined]
    assert len(asked) == 10 and all(a["component"] == "Ey" for a, _ in asked)
    assert len(_heatmaps(reply["figure"])) == 10


def test_a_figure_control_on_all_modes_redraws_without_asking_emode(server):
    post(server, {"mode": ALL_MODES}, width=1400)
    count = len(server.fetched)  # type: ignore[attr-defined]
    reply, _ = post(server, {"colormap": "sequential"}, width=1400)
    assert len(server.fetched) == count  # type: ignore[attr-defined]
    assert len(_heatmaps(reply["figure"])) == 10


def test_the_tab_sends_its_width_and_opens_a_clicked_panels_mode():
    from emode.ui.server import STATIC

    app = (STATIC / "app.js").read_text()
    assert "width: width" in app
    assert 'meta.startsWith("mode:")' in app and "change({ mode: meta.slice(" in app
