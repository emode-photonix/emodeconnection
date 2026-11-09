"""A plot as a renderer-neutral figure: ``FigureSpec``, and ``compose()`` that builds it.

EMode returns the data of a panel (``PlotView`` or ``Schematic``,
``emode/plotting.py``). Every drawing decision is made here, once, in Python:
which overlay is shown, in which color and line width, with which label, unit
and colormap. Two renderers read the result and decide nothing:
``renderers/plotly.py`` for the browser tab and ``renderers/mpl.py`` for a
static image, so the tab and a saved file agree.

The options of ``plot()`` that change only the figure (``PlotOptions``) stop
here and never reach EMode, so toggling one redraws without a round trip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .. import colormaps
from ..plotting import Arrow, PlotView, Schematic
from ..types import ArgumentError

#: The unit symbols an axis label shows.
UNIT_SYMBOL = {"nm": "nm", "um": "μm"}

#: nm per unit.
UNIT_SCALE = {"nm": 1.0, "um": 1000.0}

#: A plot's size by default: 80 mm wide, 80 / sqrt(2) mm high, as the window was.
DEFAULT_SIZE_MM = (80.0, 80.0 / math.sqrt(2))

#: The window's line width (points), from which every overlay's width is taken.
LW = 0.5

#: How each overlay kind is drawn: (color, width in points, dash, alpha).
LINE_STYLES: Dict[str, Tuple[str, float, Optional[str], float]] = {
    "outline": ("#ffffff", LW * 2 / 5, None, 0.7),
    "port": ("#00ff00", LW * 4, None, 1.0),
    "facet": ("#00bfbf", LW, None, 1.0),
    "monitor": ("#bf00bf", LW, "dash", 1.0),
    "connection": ("#4a4a4a", 1.0, None, 1.0),
    "loop": ("#4a4a4a", 1.0, None, 1.0),
    "stub": ("#4a4a4a", 1.0, None, 1.0),
}

#: Data arguments: the ones ``plot()`` sends to EMode. Everything else stays here.
DATA_ARGUMENTS = (
    "plane",
    "view",
    "component",
    "plot_function",
    "normalization",
    "mode",
    "port",
    "excitation",
    "profile",
    "wavelength",
    "slice_position",
    "forward_only",
    "blend_slices",
    "resolution",
    "section",
)


@dataclass(frozen=True)
class PlotOptions:
    """The figure-only arguments of ``plot()``, with their defaults.

    ``contours`` None is on for the x-y plane and off for the z planes, as the
    window had it. ``units`` shows both axes in one unit (``'nm'`` or ``'um'``)
    instead of each axis's own. ``figure_size`` is ``(width, height)`` in mm.
    ``colormap`` names a registered matplotlib colormap to use instead of the
    family's kind.
    """

    index_outline: bool = True
    contours: Optional[bool] = None
    text_display: bool = True
    mesh_display: bool = False
    window_display: bool = False
    legend: bool = True
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    units: Optional[str] = None
    font_size: float = 7.0
    figure_size: Optional[Tuple[float, float]] = None
    colorbar_label: Optional[str] = None
    dpi: int = 600
    colormap: Optional[str] = None
    aspect_ratio: Union[str, float, None] = None

    @classmethod
    def from_arguments(cls, arguments: Dict[str, Any]) -> "PlotOptions":
        """The options among `arguments`, the rest ignored. Raises on a bad value."""
        known = {f for f in cls.__dataclass_fields__}
        options = cls(**{k: v for k, v in arguments.items() if k in known and v is not None})
        if options.units not in (None, "nm", "um"):
            raise ArgumentError("units must be 'nm' or 'um'", "plot", "units")
        if options.figure_size is not None and len(options.figure_size) != 2:
            raise ArgumentError("figure_size is (width, height) in mm", "plot", "figure_size")
        return options


# ------------------------------------------------------------------ the spec


@dataclass(frozen=True)
class RasterSpec:
    """An image: ``z`` on the centers ``x`` and ``y`` (display units), colored by ``colormap``."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    colormap: str
    vmin: float
    vmax: float
    mask: Optional[np.ndarray] = None
    interpolation: str = "nearest"
    contour_levels: Tuple[float, ...] = ()


@dataclass(frozen=True)
class LineSpec:
    x: np.ndarray
    y: np.ndarray
    kind: str
    color: str
    width: float
    dash: Optional[str] = None
    alpha: float = 1.0
    marker: Optional[str] = None
    marker_size: float = 0.0


@dataclass(frozen=True)
class FillSpec:
    x: np.ndarray
    y: np.ndarray
    kind: str
    color: Optional[str]
    holes: Tuple[np.ndarray, ...] = ()
    edgecolor: Optional[str] = None
    hatch: Optional[str] = None
    alpha: float = 1.0
    width: float = 0.0


@dataclass(frozen=True)
class SegmentsSpec:
    xy: np.ndarray
    kind: str
    color: str
    width: float
    alpha: float


@dataclass(frozen=True)
class TextSpec:
    text: str
    x: float
    y: float
    frame: str
    ha: str
    va: str
    color: str
    size: float
    kind: str = "text"
    weight: str = "normal"


@dataclass(frozen=True)
class AxesSpec:
    xlabel: str
    ylabel: str
    xrange: Tuple[float, float]
    yrange: Tuple[float, float]
    invert_y: bool = False
    equal_aspect: bool = False
    box_aspect: Optional[float] = None
    background: Optional[str] = None
    visible: bool = True


@dataclass(frozen=True)
class ColorbarSpec:
    colormap: str
    vmin: float
    vmax: float
    ticks: Optional[Tuple[float, ...]] = None
    tick_labels: Optional[Tuple[str, ...]] = None
    label: Optional[str] = None
    tick_format: str = "%3.1f"


@dataclass(frozen=True)
class LegendEntrySpec:
    label: str
    color: Optional[str]


@dataclass(frozen=True)
class FigureSpec:
    """A figure in display units, with every color, size and label decided."""

    axes: AxesSpec
    rasters: Tuple[RasterSpec, ...] = ()
    fills: Tuple[FillSpec, ...] = ()
    lines: Tuple[LineSpec, ...] = ()
    segments: Tuple[SegmentsSpec, ...] = ()
    texts: Tuple[TextSpec, ...] = ()
    colorbar: Optional[ColorbarSpec] = None
    legend: Tuple[LegendEntrySpec, ...] = ()
    title: Optional[str] = None
    width_mm: float = DEFAULT_SIZE_MM[0]
    height_mm: float = DEFAULT_SIZE_MM[1]
    font_size: float = 7.0
    dpi: int = 600
    arrows: Tuple[Arrow, ...] = ()


# ------------------------------------------------------------------ compose


def compose(view: Union[PlotView, Schematic], options: Optional[PlotOptions] = None) -> FigureSpec:
    """The figure of `view` with `options` applied."""
    options = options or PlotOptions()
    if isinstance(view, Schematic):
        return _compose_schematic(view, options)
    if not isinstance(view, PlotView):
        raise TypeError(f"compose() takes a PlotView or a Schematic, not {type(view).__name__}")
    return _compose_view(view, options)


def _cmap(kind: str, options: PlotOptions) -> str:
    """The colormap a raster of `kind` is drawn with: the option's name, else the family's kind."""
    return options.colormap or kind


def _color(value: Optional[float], color: Optional[str], options: PlotOptions) -> Optional[str]:
    if color is not None:
        return {"grey": "#808080", "k": "#000000", "w": "#ffffff"}.get(color, color)
    if value is None:
        return None
    if options.colormap:
        from matplotlib.colors import to_hex

        return to_hex(colormaps.mpl_colormap("sequential", options.colormap)(value))
    return colormaps.color_at("sequential", value)


def _units(view: PlotView, options: PlotOptions) -> Tuple[str, str]:
    if options.units is not None:
        return options.units, options.units
    return view.axes.x_unit, view.axes.y_unit


def _axis_label(name: str, unit: str) -> str:
    return f"{name} ({UNIT_SYMBOL[unit]})"


def _compose_view(view: PlotView, options: PlotOptions) -> FigureSpec:
    xu, yu = _units(view, options)
    sx, sy = UNIT_SCALE[xu], UNIT_SCALE[yu]
    z_plane = view.plane in ("z-x", "z-y")
    contours = options.contours if options.contours is not None else not z_plane
    rasters = []
    for r in view.rasters:
        cmap = _cmap(r.kind, options)
        levels: Tuple[float, ...] = ()
        if contours and r.contour_levels is not None:
            levels = tuple(float(v) for v in r.contour_levels)
        rasters.append(
            RasterSpec(
                x=np.asarray(r.x, np.float32) / sx,
                y=np.asarray(r.y, np.float32) / sy,
                z=np.asarray(r.z, np.float32),
                colormap=cmap,
                vmin=view.scale.vmin,
                vmax=view.scale.vmax,
                mask=r.mask,
                interpolation=r.interpolation,
                contour_levels=levels,
            )
        )

    fills = [
        FillSpec(
            x=np.asarray(p.x) / sx,
            y=np.asarray(p.y) / sy,
            kind=p.kind,
            color=_color(p.value, p.color, options),
            holes=tuple(np.asarray(h) / np.array([sx, sy]) for h in p.holes),
        )
        for p in view.fills
    ]
    if options.window_display:
        for p in view.window:
            fills.append(
                FillSpec(
                    x=np.asarray(p.x) / sx,
                    y=np.asarray(p.y) / sy,
                    kind=p.kind,
                    color=None,
                    holes=tuple(np.asarray(h) / np.array([sx, sy]) for h in p.holes),
                    edgecolor="#000000" if p.kind == "window" else "#808080",
                    hatch=p.hatch,
                    width=LW,
                )
            )

    lines: List[LineSpec] = []

    def add(kind: str, x: Any, y: Any) -> None:
        color, width, dash, alpha = LINE_STYLES[kind]
        lines.append(LineSpec(np.asarray(x) / sx, np.asarray(y) / sy, kind, color, width, dash, alpha))

    texts: List[TextSpec] = []
    # facets, monitors and ports follow the index outline toggle, as the window drew them
    if options.index_outline:
        for line in view.monitors:
            add("monitor", line.x, line.y)
            if line.label and line.label_at:
                texts.append(_data_label(line, "monitor", sx, sy, options))
        for line in view.facets:
            add("facet", line.x, line.y)
            if line.label and line.label_at:
                texts.append(_data_label(line, "facet", sx, sy, options))
        for line in view.ports:
            add("port", line.x, line.y)
            if line.label and line.label_at:
                texts.append(_data_label(line, "port", sx, sy, options))
        for line in view.outlines:
            add("outline", line.x, line.y)

    segments = []
    if options.mesh_display:
        for s in view.mesh:
            xy = np.asarray(s.xy, np.float32) / np.array([sx, sy], np.float32)
            segments.append(SegmentsSpec(xy=xy, kind=s.kind, color="#000000", width=LW / 2, alpha=s.alpha))

    if options.text_display:
        for t in view.texts:
            texts.append(
                TextSpec(t.text, t.x, t.y, t.frame, t.ha, t.va, t.color or "#000000", options.font_size, t.kind)
            )

    colorbar = None
    if view.scale.kind != "none":
        cmap = _cmap("sequential" if view.scale.kind in ("index", "positive", "db") else "diverging", options)
        if view.rasters:
            cmap = rasters[0].colormap
        colorbar = ColorbarSpec(
            colormap=cmap,
            vmin=view.scale.vmin,
            vmax=view.scale.vmax,
            ticks=None if view.scale.ticks is None else tuple(view.scale.ticks),
            tick_labels=None if view.scale.tick_labels is None else tuple(view.scale.tick_labels),
            label=options.colorbar_label,
            tick_format=view.scale.tick_format,
        )

    legend: Tuple[LegendEntrySpec, ...] = ()
    if options.legend and any(e.value is not None for e in view.legend):
        legend = tuple(LegendEntrySpec(e.label, _color(e.value, None, options)) for e in view.legend)

    background = _color(view.background, None, options)
    width, height = options.figure_size or DEFAULT_SIZE_MM
    box_aspect = view.axes.box_aspect
    if options.aspect_ratio is not None and not view.axes.equal_aspect:
        box_aspect = _box_aspect(options.aspect_ratio, view)
    x0, x1 = view.axes.xlim
    y0, y1 = view.axes.ylim
    axes = AxesSpec(
        xlabel=options.xlabel or _axis_label(view.axes.xlabel, xu),
        ylabel=options.ylabel or _axis_label(view.axes.ylabel, yu),
        xrange=(x0 / sx, x1 / sx),
        yrange=(y0 / sy, y1 / sy),
        invert_y=view.axes.invert_y,
        equal_aspect=view.axes.equal_aspect,
        box_aspect=box_aspect,
        background=background,
    )
    return FigureSpec(
        axes=axes,
        rasters=tuple(rasters),
        fills=tuple(fills),
        lines=tuple(lines),
        segments=tuple(segments),
        texts=tuple(texts),
        colorbar=colorbar,
        legend=legend,
        title=options.title,
        width_mm=float(width),
        height_mm=float(height),
        font_size=options.font_size,
        dpi=options.dpi,
        arrows=tuple(view.arrows),
    )


def _data_label(line: Any, kind: str, sx: float, sy: float, options: PlotOptions) -> TextSpec:
    color = LINE_STYLES[kind][0]
    x, y = line.label_at
    return TextSpec(
        line.label, x / sx, y / sy, "data", line.align, "bottom", color, options.font_size, kind
    )


def _box_aspect(aspect_ratio: Union[str, float], view: PlotView) -> Optional[float]:
    if isinstance(aspect_ratio, (int, float)):
        return float(aspect_ratio)
    if str(aspect_ratio).lower() == "real":
        (x0, x1), (y0, y1) = view.axes.xlim, view.axes.ylim
        return abs(y1 - y0) / abs(x1 - x0) if x1 != x0 else None
    return view.axes.box_aspect


# ------------------------------------------------------------------ the schematic

BLOCK_FACE = "#e8f0fb"
BLOCK_EDGE = "#2b5c9e"
TEXT = "#1f2933"
LAUNCH = "#d1495b"
MONITOR = "#b5179e"


def _compose_schematic(view: Schematic, options: PlotOptions) -> FigureSpec:
    fills: List[FillSpec] = []
    texts: List[TextSpec] = []
    lines: List[LineSpec] = []
    for b in view.blocks:
        x0, x1 = b.x - b.width / 2, b.x + b.width / 2
        y0, y1 = b.y - b.height / 2, b.y + b.height / 2
        selected = b.name == view.selected_section
        fills.append(
            FillSpec(
                x=np.array([x0, x1, x1, x0, x0]),
                y=np.array([y0, y0, y1, y1, y0]),
                kind="block",
                color=BLOCK_FACE,
                edgecolor=LAUNCH if selected else BLOCK_EDGE,
                width=2.4 if selected else 1.2,
            )
        )
        texts.append(TextSpec(b.label, b.x, b.y, "data", "center", "center", TEXT, 6.5, "label"))
    for s in view.stubs:
        lines.append(LineSpec(np.array([s.x0, s.x1]), np.array([s.y0, s.y1]), "stub", LINE_STYLES["stub"][0], 1.0))
        if s.label:
            out = 1.0 if s.side == "right" else -1.0
            launched = view.launch is not None and view.launch.label == s.label
            if s.external:
                gap = 0.5 if launched else 0.12
                texts.append(
                    TextSpec(
                        s.label, s.x1 + out * gap, s.y1, "data",
                        "left" if s.side == "right" else "right", "center", TEXT, 6.0, "port",
                        "bold" if launched else "normal",
                    )
                )
            else:
                texts.append(
                    TextSpec(
                        s.label, s.x1 - out * 0.08, s.y1 + 0.12, "data",
                        "right" if s.side == "right" else "left", "bottom", TEXT, 5.5, "port",
                    )
                )
    for c in view.connections:
        color, width, dash, alpha = LINE_STYLES[c.kind]
        lines.append(LineSpec(np.asarray(c.x), np.asarray(c.y), c.kind, color, width, dash, alpha))
    if view.launch is not None:
        marker = "triangle-right" if view.launch.side == "left" else "triangle-left"
        lines.append(
            LineSpec(np.array([view.launch.x1]), np.array([view.launch.y1]), "launch", LAUNCH, 0.0, None, 1.0, marker, 7.0)
        )
    for m in view.monitors:
        texts.append(TextSpec(m.text, m.x, m.y, "data", m.ha, m.va, MONITOR, 6.0, "monitor"))
    x0, x1, y0, y1 = view.frame or view.limits
    width, height = options.figure_size or (
        max(4.0, (view.limits[1] - view.limits[0]) * 0.42) * 25.4,
        max(2.5, (view.limits[3] - view.limits[2]) * 0.42) * 25.4,
    )
    return FigureSpec(
        axes=AxesSpec(
            xlabel="", ylabel="", xrange=(x0, x1), yrange=(y0, y1), equal_aspect=True, visible=False
        ),
        fills=tuple(fills),
        lines=tuple(lines),
        texts=tuple(texts),
        title=options.title,
        width_mm=float(width),
        height_mm=float(height),
        font_size=options.font_size,
        dpi=options.dpi,
        arrows=tuple(view.arrows),
    )


def with_options(spec: FigureSpec, **changes: Any) -> FigureSpec:
    """`spec` with some of its top-level fields changed (the size, the dpi, the title)."""
    return replace(spec, **changes)


def split_arguments(arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """`arguments` of ``plot()`` split into those sent to EMode and the figure's."""
    data = {k: v for k, v in arguments.items() if k in DATA_ARGUMENTS}
    figure = {k: v for k, v in arguments.items() if k not in DATA_ARGUMENTS}
    return data, figure


def check_figure_arguments(arguments: Dict[str, Any], known: Sequence[str] = ()) -> None:
    """Refuse an argument ``plot()`` does not take, naming it, before anything is sent."""
    allowed = set(PlotOptions.__dataclass_fields__) | set(DATA_ARGUMENTS) | set(known)
    for name in arguments:
        if name not in allowed:
            raise ArgumentError(
                f"plot() takes no argument {name!r}. It takes {', '.join(sorted(allowed))}.",
                "plot",
                name,
            )
