"""The All modes grid labels its axes on the outer panels only (#184).

Eric, 2026-11-15: show the y tick labels and the y axis label only on the
leftmost column, and the x tick labels and the x axis label only on the bottom
row. Every panel keeps its ticks, so each still reads against the same scale.
A column whose last row is empty (5 modes on 2 columns) labels its x axis on
its lowest panel, the one with nothing under it.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from emode.ui.figure import PlotOptions
from emode.ui.modes import ModesPanel, compose_modes, frozen_arguments
from emode.ui.renderers.plotly import to_plotly

from test_mode_switch_payload import mode_view


def _grid(count: int, columns: int) -> ModesPanel:
    labels = tuple(f"{k}, TE" for k in range(count))
    return ModesPanel(
        views=tuple(mode_view(label, 40, 44) for label in labels),
        labels=labels,
        columns=columns,
        arguments=frozen_arguments({"plane": "x-y"}),
        shown=labels[0],
    )


def _axes(figure: Dict[str, Any], letter: str) -> List[Dict[str, Any]]:
    keys = sorted(
        (k for k in figure["layout"] if k.startswith(f"{letter}axis") and figure["layout"][k].get("overlaying") is None),
        key=lambda k: int(k[5:] or 1),
    )
    return [figure["layout"][k] for k in keys]


def _bottom(k: int, count: int, columns: int) -> bool:
    return k + columns >= count


@pytest.mark.parametrize("count, columns", [(20, 2), (20, 4), (6, 3), (5, 2), (4, 4)])
def test_tick_labels_and_titles_only_on_the_left_column_and_the_bottom_row(count, columns):
    spec = compose_modes(_grid(count, columns), PlotOptions())
    for k, panel in enumerate(spec.panels):
        left, bottom = k % columns == 0, _bottom(k, count, columns)
        assert panel.axes.show_yticklabels is left, k
        assert bool(panel.axes.ylabel) is left, k
        assert panel.axes.show_xticklabels is bottom, k
        assert bool(panel.axes.xlabel) is bottom, k


@pytest.mark.parametrize("count, columns", [(20, 2), (5, 2)])
def test_plotly_hides_the_inner_tick_labels_and_keeps_the_ticks(count, columns):
    figure = to_plotly(compose_modes(_grid(count, columns), PlotOptions()))
    xs, ys = _axes(figure, "x"), _axes(figure, "y")
    assert len(xs) == len(ys) == count
    for k in range(count):
        assert xs[k]["showticklabels"] is _bottom(k, count, columns), k
        assert ys[k]["showticklabels"] is (k % columns == 0), k
        assert xs[k]["ticks"] == ys[k]["ticks"] == "inside"
        assert bool(xs[k]["title"]["text"]) is _bottom(k, count, columns)
        assert bool(ys[k]["title"]["text"]) is (k % columns == 0)


def test_a_single_plot_keeps_its_tick_labels():
    from emode.ui.figure import compose

    figure = to_plotly(compose(mode_view()))
    assert figure["layout"]["xaxis"]["showticklabels"] is True
    assert figure["layout"]["yaxis"]["showticklabels"] is True
