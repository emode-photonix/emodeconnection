"""The GDS layer view: the layer panel's state and the figure drawn from it.

``plot(view='gds')`` fetches a ``GDSView`` (``emode/gds.py``) and keeps it here
as a ``GDSPanel``: the view and each layer's style (visibility, color, opacity,
hatch) and whether the simulation's shapes are drawn over the layers. The tab's
layer panel restyles the figure in the browser and posts each change to the
view's ``layers`` route, which applies it here with ``with_changes``, so the
panel's state lasts for the session and a saved file draws the same layers.

The colors default to the line palette (``emode.colormaps``), in layer order.
A KLayout layer-properties file (``layer_properties``) sets a PDK's names,
colors and hatches instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from .. import colormaps
from ..gds import GDSView, read_lyp
from ..plotting import MenuState
from ..types import ArgumentError
from .figure import (
    DEFAULT_SIZE_MM,
    LW,
    UNIT_SCALE,
    AxesSpec,
    FigureSpec,
    LayerSpec,
    PlotOptions,
    _axis_label,
)

#: The hatches a layer takes: Plotly's fill patterns, which matplotlib draws alike. ``''`` is none.
HATCHES = ("", "/", "\\", "x", "-", "|", "+", ".")

#: A layer's fill opacity by default.
DEFAULT_OPACITY = 0.5

#: The key of the simulation shapes' overlay among the figure's layers.
SHAPES = "shapes"

#: The arguments of ``plot(view='gds')`` this client applies, never sent to EMode.
LOCAL_ARGUMENTS = ("layers", "layer_properties", "shapes", "solo", "show_all")

_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


@dataclass(frozen=True)
class LayerStyle:
    """How one layer is drawn. ``name`` replaces the layer's own name when set (from a ``.lyp`` file)."""

    color: str
    opacity: float = DEFAULT_OPACITY
    hatch: str = ""
    visible: bool = True
    name: str = ""


@dataclass(frozen=True)
class GDSPanel:
    """A ``GDSView`` and its layer panel's state: each layer's style, and the shapes overlay."""

    view: GDSView
    styles: Tuple[Tuple[str, LayerStyle], ...]
    shapes: bool = False

    @classmethod
    def from_view(cls, view: GDSView) -> "GDSPanel":
        """`view` with every layer visible, colored from the line palette in layer order."""
        palette = colormaps.line_palette()
        styles = tuple(
            (layer.key, LayerStyle(color=palette[i % len(palette)])) for i, layer in enumerate(view.layers)
        )
        return cls(view=view, styles=styles)

    @property
    def menu(self) -> MenuState:
        return self.view.menu

    @property
    def keys(self) -> List[str]:
        return [key for key, _ in self.styles]

    def style(self, key: str) -> LayerStyle:
        for k, style in self.styles:
            if k == key:
                return style
        raise ArgumentError(f"no layer {key!r} in this view, which has {self.keys}", "plot", "layers")

    def with_view(self, view: GDSView) -> "GDSPanel":
        """`view` (fetched again) with this panel's styles kept for the layers it still has."""
        fresh = GDSPanel.from_view(view)
        kept = dict(self.styles)
        styles = tuple((key, kept.get(key, style)) for key, style in fresh.styles)
        return replace(fresh, styles=styles, shapes=self.shapes)

    def with_changes(self, changes: Mapping[str, Any]) -> "GDSPanel":
        """This panel with the layer panel's `changes` applied, in this order.

        - ``layer_properties``: a KLayout ``.lyp`` file's path or text, whose
          layers set their names, colors, hatches, opacity and visibility.
        - ``layers``: ``{'layer/datatype': {'visible', 'color', 'opacity', 'hatch'}}``,
          any of the four for each layer named.
        - ``solo``: a layer's key, the only one left visible.
        - ``show_all``: True shows every layer.
        - ``shapes``: whether the simulation's shapes are drawn over the layers.
        """
        styles = dict(self.styles)
        if changes.get("layer_properties"):
            for key, entry in read_lyp(changes["layer_properties"]).items():
                if key in styles:
                    color = entry["color"] or styles[key].color
                    styles[key] = LayerStyle(
                        color=color,
                        opacity=entry["opacity"],
                        hatch=entry["hatch"],
                        visible=entry["visible"],
                        name=entry["name"],
                    )
        for key, change in (changes.get("layers") or {}).items():
            if key not in styles:
                raise ArgumentError(f"no layer {key!r} in this view, which has {self.keys}", "plot", "layers")
            styles[key] = _restyled(styles[key], change)
        solo = changes.get("solo")
        if solo:
            if solo not in styles:
                raise ArgumentError(f"no layer {solo!r} in this view, which has {self.keys}", "plot", "solo")
            styles = {k: replace(s, visible=(k == solo)) for k, s in styles.items()}
        if changes.get("show_all"):
            styles = {k: replace(s, visible=True) for k, s in styles.items()}
        shapes = self.shapes if changes.get("shapes") is None else bool(changes["shapes"])
        return replace(self, styles=tuple((k, styles[k]) for k in self.keys), shapes=shapes)

    def state(self) -> Dict[str, Any]:
        """The panel's whole state as ``with_changes`` takes it, which the view keeps for the session."""
        return {
            "layers": {
                key: {"visible": s.visible, "color": s.color, "opacity": s.opacity, "hatch": s.hatch}
                for key, s in self.styles
            },
            "shapes": self.shapes,
        }


def _restyled(style: LayerStyle, change: Mapping[str, Any]) -> LayerStyle:
    """`style` with a layer panel row's `change`, each value checked."""
    if not isinstance(change, Mapping):
        raise ArgumentError("each layer's change is a dict of visible, color, opacity and hatch", "plot", "layers")
    out = style
    if "visible" in change:
        out = replace(out, visible=bool(change["visible"]))
    if "color" in change:
        color = str(change["color"])
        if not _COLOR.match(color):
            raise ArgumentError(f"a layer's color is '#rrggbb', got {color!r}", "plot", "color")
        out = replace(out, color=color.lower())
    if "opacity" in change:
        try:
            opacity = float(change["opacity"])
        except (TypeError, ValueError):
            opacity = -1.0
        if not 0.0 <= opacity <= 1.0:
            raise ArgumentError(f"a layer's opacity is from 0 to 1, got {change['opacity']!r}", "plot", "opacity")
        out = replace(out, opacity=opacity)
    if "hatch" in change:
        hatch = "" if change["hatch"] is None else str(change["hatch"])
        if hatch not in HATCHES:
            raise ArgumentError(f"a layer's hatch is one of {HATCHES}, got {hatch!r}", "plot", "hatch")
        out = replace(out, hatch=hatch)
    return out


def _label(name: str, key: str) -> str:
    return f"{name} ({key})" if name else key


def layer_rows(panel: GDSPanel) -> List[Dict[str, Any]]:
    """The layer panel's rows: each layer's key, label, polygon count and style."""
    rows = []
    for layer in panel.view.layers:
        style = panel.style(layer.key)
        rows.append(
            {
                "key": layer.key,
                "label": _label(style.name or layer.name, layer.key),
                "polygons": layer.polygon_count,
                "visible": style.visible,
                "color": style.color,
                "opacity": style.opacity,
                "hatch": style.hatch,
            }
        )
    return rows


def _shape_rings(panel: GDSPanel) -> List[np.ndarray]:
    rings = []
    for region in panel.view.shapes:
        rings.append(np.column_stack([np.asarray(region.x, float), np.asarray(region.y, float)]))
        rings.extend(np.asarray(h, float) for h in region.holes)
    return rings


def compose_gds(panel: GDSPanel, options: PlotOptions) -> FigureSpec:
    """The figure of the GDS layer view: one layer spec per layer, then the shapes' outlines."""
    axes_info = panel.view.axes
    xu = options.units or axes_info.x_unit
    yu = options.units or axes_info.y_unit
    scale = np.array([UNIT_SCALE[xu], UNIT_SCALE[yu]])
    layers: List[LayerSpec] = []
    for layer in panel.view.layers:
        style = panel.style(layer.key)
        layers.append(
            LayerSpec(
                key=layer.key,
                label=_label(style.name or layer.name, layer.key),
                polygons=tuple(np.asarray(p, float) / scale for p in layer.polygons()),
                color=style.color,
                opacity=style.opacity,
                hatch=style.hatch,
                visible=style.visible,
                width=LW,
                legend=options.legend,
            )
        )
    layers.append(
        LayerSpec(
            key=SHAPES,
            label="simulation shapes",
            polygons=tuple(r / scale for r in _shape_rings(panel)),
            color="#000000",
            opacity=0.0,
            visible=panel.shapes,
            filled=False,
            width=LW * 2,
            dash="dash",
            legend=False,
        )
    )
    x0, x1 = axes_info.xlim
    y0, y1 = axes_info.ylim
    width, height = options.figure_size or DEFAULT_SIZE_MM
    axes = AxesSpec(
        xlabel=options.xlabel or _axis_label(axes_info.xlabel, xu),
        ylabel=options.ylabel or _axis_label(axes_info.ylabel, yu),
        xrange=(x0 / scale[0], x1 / scale[0]),
        yrange=(y0 / scale[1], y1 / scale[1]),
        invert_y=axes_info.invert_y,
        equal_aspect=axes_info.equal_aspect,
    )
    return FigureSpec(
        axes=axes,
        layers=tuple(layers),
        title=options.title,
        width_mm=float(width),
        height_mm=float(height),
        font_size=options.font_size,
        dpi=options.dpi,
    )


def title(panel: GDSPanel) -> str:
    return "GDS layers"


def extra(panel: GDSPanel) -> Dict[str, Any]:
    """What the tab's layer panel needs beyond the figure: its rows, the overlay, the hatches."""
    return {
        "layers": layer_rows(panel),
        "shapes": panel.shapes,
        "has_shapes": bool(panel.view.shapes),
        "hatches": list(HATCHES),
        "decimated": panel.view.decimated,
        "pixel_nm": panel.view.pixel_nm,
    }


def apply(data: Any, controls: Mapping[str, Any]) -> GDSPanel:
    """A fetched ``GDSView`` (or a panel) with the local `controls` applied."""
    panel = GDSPanel.from_view(data) if isinstance(data, GDSView) else data
    changes = {k: controls[k] for k in LOCAL_ARGUMENTS if controls.get(k) is not None}
    return panel.with_changes(changes) if changes else panel


def state_arguments(arguments: Mapping[str, Any], panel: GDSPanel) -> Dict[str, Any]:
    """A view's `arguments` with the panel's resolved state in place of the changes that made it."""
    out = {k: v for k, v in arguments.items() if k not in LOCAL_ARGUMENTS}
    out.update(panel.state())
    return out

