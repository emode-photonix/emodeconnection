"""The All modes view: every mode of an x-y plot at once, as a grid of small panels.

The Mode menu of an x-y plot with more than one mode offers ``All modes``
after its modes. Choosing it asks EMode for each mode's panel with the plot's
other data arguments, and the tab draws them on a grid whose columns follow
the tab's width (two on a phone, three or four on a desktop). Each panel is
titled with its mode and its effective index, shares the plot's component,
normalization and colormap, and opens that mode in the single view when
clicked, from the panels already fetched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

from ..plotting import MenuState, PlotView
from .figure import FigureSpec, PlotOptions, compose

#: The Mode menu's entry for the grid of every mode.
ALL_MODES = "All modes"

#: The tab's widths (CSS pixels) below which the grid has two, then three columns.
PHONE_WIDTH = 700
TABLET_WIDTH = 1100


def mode_labels(view: Any) -> Tuple[str, ...]:
    """The modes an All modes grid of `view` shows: its Mode menu's, without the free-space fields."""
    if not isinstance(view, PlotView) or view.plane != "x-y" or not view.rasters:
        return ()
    modes = tuple(
        m
        for m in (view.menu.modes or ())
        if m != ALL_MODES and not m.endswith("(free space)")
    )
    return modes if len(modes) > 1 else ()


def offers_all_modes(view: Any) -> bool:
    """Whether `view` is an x-y field panel of more than one mode."""
    return bool(mode_labels(view))


def columns_for(width: Optional[float], count: int) -> int:
    """The grid's columns for a tab `width` CSS pixels wide: 2 on a phone, 3 on a tablet, 4 wider."""
    if width is None or width <= 0:
        width = TABLET_WIDTH
    columns = 2 if width < PHONE_WIDTH else 3 if width < TABLET_WIDTH else 4
    return max(1, min(columns, count))


def with_all_modes(menu: MenuState, selected: bool = False) -> MenuState:
    """`menu` with ``All modes`` after its modes, and chosen when `selected`."""
    modes = [m for m in (menu.modes or []) if m != ALL_MODES] + [ALL_MODES]
    update: Dict[str, Any] = {"modes": modes}
    if selected:
        update["selected"] = {**menu.selected, "mode": ALL_MODES}
    return menu.model_copy(update=update)


@dataclass(frozen=True)
class ModesPanel:
    """Every mode's panel of an x-y plot, in menu order, and the grid's columns.

    `arguments` are the data arguments the panels were fetched with (the mode
    aside) and `pixels` the panel size asked for, so the grid is fetched again
    only when they change. `shown` is the mode the single view showed.
    """

    views: Tuple[PlotView, ...]
    labels: Tuple[str, ...]
    columns: int
    arguments: Tuple[Tuple[str, Any], ...]
    shown: str

    @property
    def menu(self) -> MenuState:
        return with_all_modes(self.views[0].menu, selected=True)

    @property
    def plane(self) -> str:
        return "x-y"

    def view_of(self, label: str) -> Optional[PlotView]:
        """The panel of mode `label`, None when the grid does not have it."""
        return self.views[self.labels.index(label)] if label in self.labels else None


def frozen_arguments(arguments: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """`arguments` without the mode, as a comparable key."""
    return tuple(sorted((k, repr(v)) for k, v in arguments.items() if k != "mode"))


def panel_title(view: PlotView, label: str) -> str:
    """A panel's title: its mode's label and the effective index the single view shows, as math."""
    neff = next((t.text for t in view.texts if "eff" in t.text), None)
    return f"{label}: {neff}" if neff else label


def compose_modes(panel: ModesPanel, options: PlotOptions) -> FigureSpec:
    """The grid of every mode, each panel drawn as the single view draws it, without its texts.

    The panels share the plot's options (component, colormap, contours and
    outlines). Their own texts, legend and colorbar are left out, which a
    small panel has no room for: the title carries the mode and its index.
    """
    each = replace(options, text_display=False, legend=False, title=None)
    columns = max(1, min(panel.columns, len(panel.views)))
    rows = math.ceil(len(panel.views) / columns)
    panels = []
    for k, (view, label) in enumerate(zip(panel.views, panel.labels)):
        spec = compose(view, each)
        rasters = tuple(replace(r, target=f"mode:{label}") for r in spec.rasters)
        # the axis titles on the outer panels only: the bottom of each column, the left row
        bottom = k + columns >= len(panel.views)
        axes = replace(
            spec.axes,
            xlabel=spec.axes.xlabel if bottom else "",
            ylabel=spec.axes.ylabel if k % columns == 0 else "",
        )
        panels.append(
            replace(
                spec,
                axes=axes,
                rasters=rasters,
                colorbar=None,
                legend=(),
                title=panel_title(view, label),
            )
        )
    first = panels[0]
    return FigureSpec(
        axes=first.axes,
        panels=tuple(panels),
        panel_grid=(rows, columns),
        title=options.title,
        width_mm=first.width_mm * columns / 2,
        height_mm=first.height_mm * rows / 2,
        font_size=options.font_size,
        dpi=options.dpi,
    )
