"""A z panel of an EME device with the x-y field at its cut drawn below it (#180).

An EME device's z-x or z-y plot offers the z of its x-y cut (``MenuState.position``)
and marks it with a line. The tab also asks EMode for the device's x-y field at
that z, with the plot's component, normalization, mode, port and excitation,
and draws it as a second panel under the z panel, on the plot's colormap. The
x-y panel is fetched again only when its arguments change (the cut moved, or a
data control changed), and a figure option redraws both from what is held.

Only the tab does this. A script's ``plot()`` handle, its PNG and the headless
path draw the z panel alone, as the call asked.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

from ..plotting import MenuState, PlotView
from .figure import FigureSpec, PlotOptions, compose

#: A z panel's arguments that mean nothing on the x-y plane: its transverse position
#: or reduction, its section zoom and its z sampling.
Z_ONLY = ("slice_position", "section", "resolution")


def offers_cut(view: Any) -> bool:
    """Whether `view` is a z panel that offers the z of an x-y cut."""
    return isinstance(view, PlotView) and view.plane in ("z-x", "z-y") and view.menu.position is not None


def cut_z(view: PlotView, arguments: Dict[str, Any]) -> float:
    """The z of `view`'s cut in um: the one asked for, else the one EMode marked."""
    asked = arguments.get("z_position")
    if asked is not None:
        try:
            return float(asked)
        except (TypeError, ValueError):
            pass
    assert view.menu.position is not None
    try:
        return float(str(view.menu.position.value).split()[0])
    except (ValueError, IndexError):
        return 0.0


def cut_arguments(view: PlotView, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """The data arguments of the device's x-y field at `view`'s cut."""
    out = {k: v for k, v in arguments.items() if k not in Z_ONLY}
    out.update(plane="x-y", z_position=cut_z(view, arguments))
    return out


@dataclass(frozen=True)
class CutPanel:
    """A z panel and the x-y field at its cut, with the arguments the x-y field was fetched with."""

    zpanel: PlotView
    xy: PlotView
    arguments: Tuple[Tuple[str, Any], ...]

    @property
    def menu(self) -> MenuState:
        return self.zpanel.menu

    @property
    def plane(self) -> str:
        return self.zpanel.plane


def frozen(arguments: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """`arguments` as a comparable key."""
    return tuple(sorted((k, repr(v)) for k, v in arguments.items()))


def held_xy(previous: Optional[CutPanel], arguments: Dict[str, Any]) -> Optional[PlotView]:
    """The x-y field `previous` holds, when it was fetched with `arguments`."""
    if previous is not None and previous.arguments == frozen(arguments):
        return previous.xy
    return None


def compose_cut(panel: CutPanel, options: PlotOptions) -> FigureSpec:
    """The z panel on top and the x-y field at its cut below it, each drawn as alone.

    Both take the plot's options (component, colormap, contours as each plane
    defaults them, outlines), and each keeps its own colorbar, texts and axes.
    """
    each = replace(options, title=None)
    top = compose(panel.zpanel, each)
    bottom = compose(panel.xy, each)
    return FigureSpec(
        axes=top.axes,
        panels=(top, bottom),
        panel_grid=(2, 1),
        title=options.title,
        width_mm=top.width_mm,
        height_mm=top.height_mm + bottom.height_mm,
        font_size=options.font_size,
        dpi=options.dpi,
    )
