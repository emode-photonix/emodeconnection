"""Figures of several panels, and ``emode.Figure`` that builds one from results and curves.

A ``FigureSpec`` with ``panels`` is a grid of panels, each a ``FigureSpec`` of
its own. The matplotlib renderer draws one axes per panel, the Plotly renderer
one pair of axes per panel on one page, so the tab and a saved file agree.
``emode.Figure`` puts sweeps, an optimizer's history, a pulse and the script's
own curves (a fit, a theory, a computed quantity) on its panels, which is
what the website examples drew with their own matplotlib.
"""
from __future__ import annotations

import base64

import numpy as np
import pytest

import emode
from emode.ui.figure import FigureSpec, PlotOptions, compose
from emode.ui.renderers.plotly import to_plotly

from test_sweep_plots import beam_sweep, length_sweep, pulse


def _decode(typed) -> np.ndarray:
    if isinstance(typed, list):
        return np.asarray(typed, float)
    return np.frombuffer(base64.b64decode(typed["bdata"]), dtype=np.dtype(typed["dtype"]))


def two_panels() -> "emode.Figure":
    fig = emode.Figure(rows=2, title="two")
    top = fig.panel(0)
    top.sweep(length_sweep(), output_mode=0)
    top.line([10000.0, 30000.0], [0.8, 0.8], style="--", label="theory")
    top.set(ylabel="Transmission", ylim=(0.0, 1.0), title="top")
    bottom = fig.panel(1)
    bottom.line([1.0, 2.0, 3.0], [3.0, 1.0, 2.0], style="o-", label="data", color=2)
    bottom.vline(2.0, label="center")
    bottom.hline(1.5)
    bottom.vspan(1.2, 1.4, label="band")
    bottom.set(xlabel="x (a.u.)", ylabel="y", xlim=(0.5, 3.5))
    return fig


# ------------------------------------------------------------------ the spec


def test_a_figure_of_two_panels_is_a_spec_with_two_panels():
    spec = compose(two_panels())
    assert isinstance(spec, FigureSpec)
    assert len(spec.panels) == 2 and spec.panel_grid == (2, 1)
    assert spec.title == "two"
    top, bottom = spec.panels
    assert top.title == "top" and top.axes.ylabel == "Transmission" and top.axes.yrange == (0.0, 1.0)
    assert top.axes.xlabel == "length (nm)"  # the sweep's own label where none is set
    assert [line.label for line in top.lines] == ["left 0 to right 0", "theory"]
    theory = top.lines[1]
    assert theory.dash == "dash"
    assert bottom.axes.xrange == (0.5, 3.5) and bottom.axes.xlabel == "x (a.u.)"


def test_a_style_string_draws_a_line_and_its_markers():
    bottom = compose(two_panels()).panels[1]
    data = [line for line in bottom.lines if line.kind == "user"]
    assert [(l.dash, l.marker, l.label) for l in data] == [(None, None, "data"), (None, "circle", None)]
    assert data[0].color == emode.colormaps.line_palette()[2] == data[1].color


def test_reference_lines_and_bands_span_the_panel():
    bottom = compose(two_panels()).panels[1]
    (vline,) = [l for l in bottom.lines if l.kind == "vline"]
    assert list(vline.x) == [2.0, 2.0] and tuple(vline.y) == bottom.axes.yrange
    (hline,) = [l for l in bottom.lines if l.kind == "hline"]
    assert tuple(hline.x) == bottom.axes.xrange and list(hline.y) == [1.5, 1.5]
    (band,) = bottom.fills
    assert sorted(set(np.asarray(band.x).tolist())) == [1.2, 1.4]
    assert sorted(set(np.asarray(band.y).tolist())) == list(bottom.axes.yrange)


def test_a_panel_takes_a_pulse_map_with_its_colorbar():
    fig = emode.Figure(cols=2)
    fig.panel(0).sweep(pulse(), panel="spectrum against z")
    fig.panel(1).sweep(pulse(), panel="energy")
    left, right = compose(fig).panels
    assert len(left.rasters) == 1 and left.colorbar is not None
    assert not right.rasters and right.axes.ylabel == "Energy (pJ)"


def test_a_panel_out_of_the_grid_is_refused():
    from emode.types import ArgumentError

    with pytest.raises(ArgumentError, match="panel"):
        emode.Figure(rows=1, cols=2).panel(2)
    fig = emode.Figure(rows=2, cols=2)
    assert fig.panel((1, 0)) is fig.panel(2)  # by (row, column) or row-major index


def test_the_figure_size_grows_with_the_grid():
    spec = compose(two_panels())
    single = compose(emode.Figure())
    assert spec.height_mm == pytest.approx(2 * single.height_mm)
    sized = compose(two_panels(), PlotOptions(figure_size=(100.0, 120.0)))
    assert (sized.width_mm, sized.height_mm) == (100.0, 120.0)


# ------------------------------------------------------------------ both renderers


def test_matplotlib_draws_one_axes_per_panel():
    pytest.importorskip("matplotlib")
    from emode.ui.renderers import mpl

    spec = compose(two_panels())
    fig = mpl.to_figure(spec)
    axes = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    assert len(axes) == 2
    for ax, panel in zip(axes, spec.panels):
        drawn = [l for l in ax.lines if l.get_gid() in {line.kind for line in panel.lines}]
        assert len(drawn) == len(panel.lines)
        for line, artist in zip(panel.lines, drawn):
            np.testing.assert_allclose(np.asarray(artist.get_ydata(), float), line.y)
        assert ax.get_title() == (panel.title or "")
        assert ax.get_xlim() == pytest.approx(panel.axes.xrange)
    assert fig._suptitle.get_text() == "two"
    assert mpl.render(spec, "png")[:4] == b"\x89PNG"


def test_plotly_draws_each_panel_on_its_own_axes():
    spec = compose(two_panels())
    figure = to_plotly(spec)
    layout = figure["layout"]
    assert {"xaxis", "yaxis", "xaxis2", "yaxis2"} <= set(layout)
    top, bottom = layout["yaxis"]["domain"], layout["yaxis2"]["domain"]
    assert top[0] > bottom[1]  # the first panel on top
    assert layout["xaxis2"]["anchor"] == "y2" and layout["yaxis2"]["anchor"] == "x2"
    assert layout["yaxis"]["range"] == [0.0, 1.0]
    for k, panel in enumerate(spec.panels):
        axis = "x" if k == 0 else f"x{k + 1}"
        traces = [t for t in figure["data"] if t.get("xaxis", "x") == axis and t.get("meta") in {l.kind for l in panel.lines}]
        assert len(traces) == len(panel.lines)
        for line, trace in zip(panel.lines, traces):
            np.testing.assert_allclose(_decode(trace["y"]), line.y)
    titles = [a["text"] for a in layout["annotations"] if a.get("name") == "panel_title"]
    assert titles == ["top"]
    assert layout["title"]["text"] == "two"


def test_a_panel_heatmaps_colorbar_sits_beside_its_panel():
    fig = emode.Figure(cols=2)
    fig.panel(0).sweep(pulse(), panel="spectrum against z")
    fig.panel(1).sweep(beam_sweep())
    figure = to_plotly(compose(fig))
    (heatmap,) = [t for t in figure["data"] if t["type"] == "heatmap"]
    assert heatmap["xaxis"] == "x" and heatmap["colorbar"]["x"] <= figure["layout"]["xaxis2"]["domain"][0]


# ------------------------------------------------------------------ showing it


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.delenv("EMODE_HEADLESS", raising=False)
    monkeypatch.delenv("EMODE_PLOT_INLINE", raising=False)
    monkeypatch.setenv("EMODE_NONINTERACTIVE", "1")


def test_figure_plot_with_a_file_name_saves_it(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.chdir(tmp_path)
    image = two_panels().plot(file_name="two", dpi=100)
    assert image.file.endswith("two.png")
    assert (tmp_path / "EM_figures" / "two.png").read_bytes()[:4] == b"\x89PNG"


def test_em_plot_of_a_figure_registers_a_figure_view(quiet):
    from plot_standin import session

    em, client = session()
    handle = em.plot(two_panels())
    assert handle.id == "c1"
    assert len(handle.figure.panels) == 2
    assert client.calls("EM_plot") == []


def test_the_tab_draws_a_figure_of_panels():
    import http.client
    import json

    from emode.ui.server import UIServer
    from emode.ui.views import ViewRegistry, kind_of, title_of

    fig = two_panels()
    assert kind_of(fig) == "figure" and title_of(fig, {}) == "two"
    registry = ViewRegistry()
    view = registry.add(fig, {}, PlotOptions(), (1200, 750))
    server = UIServer(registry, lambda *a: None).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
        headers = {"Host": f"127.0.0.1:{server.port}", "X-EMode-Token": server.token, "Content-Type": "application/json"}
        conn.request("POST", f"/api/views/{view.id}/data", body=json.dumps({"controls": {}}), headers=headers)
        reply = json.loads(conn.getresponse().read())
        conn.close()
    finally:
        server.stop()
    assert reply["kind"] == "figure"
    assert "xaxis2" in reply["figure"]["layout"]
