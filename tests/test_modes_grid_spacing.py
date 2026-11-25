"""The All modes grid keeps less whitespace between its panels (Eric, 2026-11-24).

"Reduce the whitespace between modes when plotting all modes. Not completely, but
not as much whitespace as right now." Today's gaps (fractions of the grid) were
sized for tick labels on every panel: 0.05 between columns and 0.1 between rows
at 4 columns and 3 rows. The target is about half, never zero, and the grid is
as tall as its panels' data make it, so a wide, short cross section has no empty
box around it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple

import pytest

from emode.ui.figure import PlotOptions, figure_aspect
from emode.ui.modes import compose_modes
from emode.ui.renderers.plotly import to_plotly

from test_modes_grid_axes import _axes, _grid


def _domains(figure: Dict[str, Any]) -> List[Tuple[List[float], List[float]]]:
    return [(x["domain"], y["domain"]) for x, y in zip(_axes(figure, "x"), _axes(figure, "y"))]


def _gaps(figure: Dict[str, Any], columns: int) -> Tuple[float, float]:
    d = _domains(figure)
    gap_x = d[1][0][0] - d[0][0][1] if len(d) > 1 else 0.0
    rows = -(-len(d) // columns)
    gap_y = d[0][1][0] - d[columns][1][1] if rows > 1 else 0.0
    return gap_x, gap_y


def _figure(count: int, columns: int, pixels=None, **options) -> Dict[str, Any]:
    return to_plotly(compose_modes(_grid(count, columns), PlotOptions(**options)), pixels)


def test_a_ten_mode_grid_has_half_the_gaps_it_had():
    # before: 0.05 between columns, 0.1 between rows
    gap_x, gap_y = _gaps(_figure(10, 4), 4)
    assert gap_x == pytest.approx(0.025)
    assert gap_y == pytest.approx(0.05)


def test_a_four_mode_grid_has_half_the_gaps_it_had():
    # before: 4 columns 0.05 between columns, 2 columns 0.1 between columns and 0.14 between rows
    assert _gaps(_figure(4, 4), 4)[0] == pytest.approx(0.025)
    gap_x, gap_y = _gaps(_figure(4, 2), 2)
    assert gap_x == pytest.approx(0.05)
    assert gap_y == pytest.approx(0.07)


@pytest.mark.parametrize("count, columns", [(10, 4), (4, 4), (4, 2), (20, 4), (6, 3)])
def test_the_gaps_stay_open_and_the_panels_do_not_overlap(count, columns):
    domains = _domains(_figure(count, columns))
    gap_x, gap_y = _gaps(_figure(count, columns), columns)
    assert gap_x > 0
    for k, (x, y) in enumerate(domains):
        assert x[0] < x[1] and y[0] < y[1]
        if k % columns:
            assert domains[k - 1][0][1] < x[0]
        if k >= columns:
            assert domains[k - columns][1][0] > y[1]
    if len(domains) > columns:
        assert gap_y > 0


@pytest.mark.parametrize("size", [640, 960, 1280])
@pytest.mark.parametrize("count, columns", [(10, 4), (20, 4), (6, 3)])
def test_a_row_gap_holds_the_panel_title(count, columns, size):
    spec = compose_modes(_grid(count, columns), PlotOptions())
    width = size - 80
    height = figure_aspect(spec) * width
    figure = to_plotly(spec, (width, int(height)))
    _, gap_y = _gaps(figure, columns)
    title = (spec.font_size + 4) * 1.3
    assert gap_y * height >= title


def test_the_grid_height_follows_the_panels_data_aspect():
    # the cross section is 4000 nm wide and 2220 nm tall, in 4 columns
    grid = _grid(10, 4)
    wide = tuple(
        v.model_copy(update={"axes": v.axes.model_copy(update={"xlim": (-2000.0, 2000.0), "ylim": (-1110.0, 1110.0)})})
        for v in grid.views
    )
    spec = compose_modes(replace(grid, views=wide), PlotOptions())
    assert spec.panels[0].axes.xrange == (-2000.0, 2000.0)
    width = 880.0
    height = figure_aspect(spec) * width
    x, y = _domains(to_plotly(spec))[0]
    box = ((y[1] - y[0]) * height) / ((x[1] - x[0]) * width)
    data = 2220 / 4000
    assert box == pytest.approx(data, rel=0.02)
    # before: the boxes were 0.707 tall for 0.555 of data, 27% slack
    assert box / data < 1.05


def test_a_chosen_aspect_ratio_still_wins():
    spec = compose_modes(_grid(10, 4), PlotOptions(aspect_ratio=1.0))
    assert all(p.axes.box_aspect == 1.0 for p in spec.panels)
    assert figure_aspect(spec) > 3 * 0.9 / 4


def test_a_cut_and_a_figure_of_panels_keep_their_gaps():
    from test_figure_panels import two_panels
    from emode.ui.figure import compose
    from emode.ui.cut import compose_cut
    from emode.ui.cut import CutPanel
    from test_cut_panel import cut_view
    from plot_fixtures import xy_view

    cut = compose_cut(CutPanel(zpanel=cut_view(), xy=xy_view(), arguments=()), PlotOptions())
    figure = to_plotly(cut)
    ys = [a["domain"] for a in _axes(figure, "y")]
    # 2 rows: 0.14 between rows
    assert ys[0][0] - ys[1][1] == pytest.approx(0.14)
    assert cut.panel_gap is None
    spec = compose(two_panels())
    ys = [a["domain"] for a in _axes(to_plotly(spec), "y")]
    assert ys[0][0] - ys[1][1] == pytest.approx(0.14)
    assert spec.panel_gap is None


def _px_gap(count: int, columns: int, css: int) -> Tuple[float, float]:
    """The row gap and the least it may be, in CSS px, for a tab `css` px wide."""
    from emode.ui.modes import grid_pixels

    spec = compose_modes(_grid(count, columns), PlotOptions())
    pixels = grid_pixels(css, figure_aspect(spec))
    _, gap_y = _gaps(to_plotly(spec, pixels), columns)
    least = (spec.font_size + 4) * 1.3 + 4 + 4
    return gap_y * pixels[1], least


@pytest.mark.parametrize("css, columns", [(375, 2), (390, 2), (768, 3)])
def test_a_phone_row_gap_holds_the_title_the_ticks_and_air(css, columns):
    # the panel above ends in outside ticks (4 px), then a few px of air, then this title
    gap, least = _px_gap(10, columns, css)
    assert gap >= least - 0.5


def test_a_desktop_plot_keeps_the_gaps_it_had():
    spec = compose_modes(_grid(4, 2), PlotOptions())
    gap_x, gap_y = _gaps(to_plotly(spec, (1280, 1280)), 2)
    assert (gap_x, gap_y) == (pytest.approx(0.05), pytest.approx(0.07))


def test_the_grid_is_sized_by_its_plot_area_in_css_pixels():
    from emode.ui.modes import grid_pixels

    # the layout margins (left 60, right 20) are not plot area
    assert grid_pixels(375, 0.5) == (375, int((375 - 80) * 0.5))
    assert grid_pixels(40, 0.5)[1] >= 1
