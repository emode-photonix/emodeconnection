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
import textwrap
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .. import colormaps
from ..plotting import Arrow, MaterialView, PlotView, Schematic, SMatrixView
from ..types import ArgumentError

#: The unit symbols an axis label shows.
UNIT_SYMBOL = {"nm": "nm", "um": "μm"}

#: nm per unit.
UNIT_SCALE = {"nm": 1.0, "um": 1000.0}

#: A plot's size by default: 80 mm wide, 80 / sqrt(2) mm high, as the window was.
DEFAULT_SIZE_MM = (80.0, 80.0 / math.sqrt(2))

#: The window's line width (points), from which every overlay's width is taken.
LW = 0.5

#: How each overlay kind is drawn: (width in points, dash, alpha). Its color is
#: ``colormaps.overlay_colors()``.
LINE_STYLES: Dict[str, Tuple[float, Optional[str], float]] = {
    "outline": (LW * 2 / 5, None, 0.7),
    "port": (LW * 4, None, 1.0),
    "facet": (LW, None, 1.0),
    "monitor": (LW, "dash", 1.0),
    "connection": (1.0, None, 1.0),
    "loop": (1.0, None, 1.0),
    "stub": (1.0, None, 1.0),
    "slice": (LW * 2, "dash", 1.0),
}


def line_style(kind: str) -> Tuple[str, float, Optional[str], float]:
    """An overlay kind's (color, width in points, dash, alpha)."""
    return (colormaps.overlay_colors()[kind], *LINE_STYLES[kind])


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
    "z_position",
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
    ``colormap`` names the colormap every raster is drawn with instead of the
    family's map for its kind: one of ``colormaps.NAMES``, a kind, or any
    colormap matplotlib has registered.
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
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    @classmethod
    def from_arguments(cls, arguments: Dict[str, Any]) -> "PlotOptions":
        """The options among `arguments`, the rest ignored. Raises on a bad value."""
        known = {f for f in cls.__dataclass_fields__}
        options = cls(**{k: v for k, v in arguments.items() if k in known and v is not None})
        if options.units not in (None, "nm", "um"):
            raise ArgumentError("units must be 'nm' or 'um'", "plot", "units")
        if options.figure_size is not None and len(options.figure_size) != 2:
            raise ArgumentError("figure_size is (width, height) in mm", "plot", "figure_size")
        if "colormap" in arguments and arguments["colormap"] is not None:
            colormaps.check_colormap(arguments["colormap"])
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
    #: The line's legend entry, None for a line without one.
    label: Optional[str] = None
    #: The y axis the line is read against: the panel's own ('left') or its twin ('right').
    axis: str = "left"


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
    grid: bool = False
    integer_ticks: bool = False
    #: A second y axis on the right (``Panel.twin()``): its label and range, None without one.
    y2label: Optional[str] = None
    y2range: Optional[Tuple[float, float]] = None
    #: Each y axis's label and tick color, None for the default black.
    ylabel_color: Optional[str] = None
    y2label_color: Optional[str] = None


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
class LayerSpec:
    """A GDS layer (or the simulation's shapes): many polygons drawn as one, with one style.

    ``polygons`` are ``(n, 2)`` rings in display units, each open. ``filled``
    layers are filled with ``color`` at ``opacity`` and hatched with ``hatch``
    (one of ``'/'``, ``'\\'``, ``'x'``, ``'-'``, ``'|'``, ``'+'``, ``'.'``, or ``''``),
    the others only outlined. A layer that is not ``visible`` is kept, hidden, so the
    tab can show it again without asking for the figure.
    """

    key: str
    label: str
    polygons: Tuple[np.ndarray, ...]
    color: str
    opacity: float = 0.5
    hatch: str = ""
    visible: bool = True
    filled: bool = True
    width: float = 0.5
    dash: Optional[str] = None
    legend: bool = True


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
    #: A figure of several panels: each panel's spec, row-major on a (rows, columns) grid.
    #: The figure's own axes and drawables are then unused.
    panels: Tuple["FigureSpec", ...] = ()
    panel_grid: Tuple[int, int] = (1, 1)
    title: Optional[str] = None
    width_mm: float = DEFAULT_SIZE_MM[0]
    height_mm: float = DEFAULT_SIZE_MM[1]
    font_size: float = 7.0
    dpi: int = 600
    arrows: Tuple[Arrow, ...] = ()
    layers: Tuple[LayerSpec, ...] = ()


# ------------------------------------------------------------------ compose


#: Anything ``compose()`` draws.
Drawable = Union[PlotView, Schematic, SMatrixView, MaterialView, Any]


def compose(view: Drawable, options: Optional[PlotOptions] = None) -> FigureSpec:
    """The figure of `view` with `options` applied.

    `view` is a ``PlotView``, a ``Schematic``, an ``SMatrixView``, a
    ``MaterialView``, a ``FarField`` or a sweep's ``SweepPanel``.
    """
    from ..free_space import FarField
    from ..gds import GDSView
    from .gds import GDSPanel, compose_gds

    options = options or PlotOptions()
    if isinstance(view, GDSView):
        view = GDSPanel.from_view(view)
    if isinstance(view, GDSPanel):
        return compose_gds(view, options)
    if isinstance(view, Schematic):
        return _compose_schematic(view, options)
    if isinstance(view, SMatrixView):
        return _compose_smatrix(view, options)
    if isinstance(view, MaterialView):
        return _compose_material(view, options)
    if isinstance(view, FarField):
        return _compose_farfield(view, options)
    if isinstance(view, FarFieldPanel):
        return _compose_farfield(view.farfield, options, view.panel)
    from .sweep_figure import SweepPanel, compose_sweep

    if isinstance(view, SweepPanel):
        return compose_sweep(view, options)
    from ..figures import Figure
    from .panels import compose_figure

    if isinstance(view, Figure):
        return compose_figure(view, options)
    if not isinstance(view, PlotView):
        raise TypeError(
            "compose() takes a PlotView, Schematic, SMatrixView, MaterialView, FarField or GDSView, "
            f"not {type(view).__name__}"
        )
    return _compose_view(view, options)


def _cmap(kind: str, options: PlotOptions) -> str:
    """The colormap a raster of `kind` is drawn with: the option's name, else the family's."""
    return colormaps.resolve(kind, options.colormap)


def _color(value: Optional[float], color: Optional[str], options: PlotOptions) -> Optional[str]:
    if color is not None:
        return {"grey": "#808080", "k": "#000000", "w": "#ffffff"}.get(color, color)
    if value is None:
        return None
    kind = colormaps.kind_of(options.colormap or "sequential")
    if kind is not None:
        return colormaps.color_at(kind, value)
    from matplotlib.colors import to_hex

    return to_hex(colormaps.get(options.colormap)(value))


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
                vmin=view.scale.vmin if options.vmin is None else options.vmin,
                vmax=view.scale.vmax if options.vmax is None else options.vmax,
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
        color, width, dash, alpha = line_style(kind)
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

    if view.slice_marker is not None:
        # the z of the x-y cut, on a z panel, whatever the toggles (emode#1625)
        add("slice", view.slice_marker.x, view.slice_marker.y)

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
            vmin=view.scale.vmin if options.vmin is None else options.vmin,
            vmax=view.scale.vmax if options.vmax is None else options.vmax,
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
    if options.window_display and view.axes.xlim_window and view.axes.ylim_window:
        (x0, x1), (y0, y1) = view.axes.xlim_window, view.axes.ylim_window
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
    color = line_style(kind)[0]
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


def _compose_schematic(view: Schematic, options: PlotOptions) -> FigureSpec:
    palette = colormaps.schematic_colors()
    face, edge, text = palette["block_face"], palette["block_edge"], palette["text"]
    launch, monitor = palette["launch"], palette["monitor"]
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
                color=face,
                edgecolor=launch if selected else edge,
                width=2.4 if selected else 1.2,
            )
        )
        texts.append(TextSpec(b.label, b.x, b.y, "data", "center", "center", text, 6.5, "label"))
    for s in view.stubs:
        lines.append(LineSpec(np.array([s.x0, s.x1]), np.array([s.y0, s.y1]), "stub", line_style("stub")[0], 1.0))
        if s.label:
            out = 1.0 if s.side == "right" else -1.0
            launched = view.launch is not None and view.launch.label == s.label
            if s.external:
                gap = 0.5 if launched else 0.12
                texts.append(
                    TextSpec(
                        s.label, s.x1 + out * gap, s.y1, "data",
                        "left" if s.side == "right" else "right", "center", text, 6.0, "port",
                        "bold" if launched else "normal",
                    )
                )
            else:
                texts.append(
                    TextSpec(
                        s.label, s.x1 - out * 0.08, s.y1 + 0.12, "data",
                        "right" if s.side == "right" else "left", "bottom", text, 5.5, "port",
                    )
                )
    for c in view.connections:
        color, width, dash, alpha = line_style(c.kind)
        lines.append(LineSpec(np.asarray(c.x), np.asarray(c.y), c.kind, color, width, dash, alpha))
    if view.launch is not None:
        marker = "triangle-right" if view.launch.side == "left" else "triangle-left"
        lines.append(
            LineSpec(np.array([view.launch.x1]), np.array([view.launch.y1]), "launch", launch, 0.0, None, 1.0, marker, 7.0)
        )
    for m in view.monitors:
        texts.append(TextSpec(m.text, m.x, m.y, "data", m.ha, m.va, monitor, 6.0, "monitor"))
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


# ------------------------------------------------------------------ the S-matrix

#: The S-matrix plot's components, as its menu offers them.
SMATRIX_COMPONENTS = ("abs^2", "angle", "real", "imag")

#: Which colormap of the family each S-matrix component is drawn with.
SMATRIX_KINDS = {"abs^2": "sequential", "angle": "cyclic", "real": "sequential", "imag": "sequential"}


def smatrix_values(view: SMatrixView) -> Tuple[np.ndarray, str]:
    """The selected block of `view` as the component the plot shows, rows the output modes."""
    selected = view.menu.selected
    labels = list(view.port_labels)
    i_in = labels.index(selected.get("input", labels[0]))
    i_out = labels.index(selected.get("output", labels[-1]))
    component = selected.get("component", "abs^2")
    if component not in SMATRIX_COMPONENTS:
        raise ArgumentError(
            f"component must be one of {SMATRIX_COMPONENTS}, got {component!r}", "plot_S_matrix", "plot_function"
        )
    block = view.smatrix.block(to_ports=i_out, from_ports=i_in)
    values = {
        "real": np.real,
        "imag": np.imag,
        "abs^2": lambda b: np.abs(b) ** 2,
        "angle": np.angle,
    }[component](block)
    return np.asarray(values, float), component


def smatrix_scale(values: np.ndarray, component: str, options: PlotOptions) -> Tuple[float, float, str]:
    """The color scale of the S-matrix plot and its colorbar's tick format.

    Automatic from the block's extremes, ``abs^2`` snapped to 0 to 1 when its
    range is over a half, and replaced by the user's `vmin` and `vmax`.
    """
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if component == "abs^2" and abs(vmax - vmin) > 0.5:
        vmin, vmax = 0.0, 1.0
    if options.vmin is not None:
        vmin = float(options.vmin)
    if options.vmax is not None:
        vmax = float(options.vmax)
    if max(abs(vmin), abs(vmax)) < 0.05:
        fmt = "%.1e"
    elif abs(vmax - vmin) < 0.05:
        fmt = "%5.4f"
    else:
        fmt = "%3.2f"
    return vmin, vmax, fmt


def _compose_smatrix(view: SMatrixView, options: PlotOptions) -> FigureSpec:
    values, component = smatrix_values(view)
    vmin, vmax, fmt = smatrix_scale(values, component, options)
    n_out, n_in = values.shape
    cmap = _cmap(SMATRIX_KINDS[component], options)
    raster = RasterSpec(
        x=np.arange(n_in, dtype=float),
        y=np.arange(n_out, dtype=float),
        z=values,
        colormap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    width, height = options.figure_size or DEFAULT_SIZE_MM
    axes = AxesSpec(
        xlabel=options.xlabel or "Input mode",
        ylabel=options.ylabel or "Output mode",
        xrange=(-0.5, n_in - 0.5),
        yrange=(-0.5, n_out - 0.5),
        equal_aspect=True,
        integer_ticks=True,
    )
    return FigureSpec(
        axes=axes,
        rasters=(raster,),
        colorbar=ColorbarSpec(cmap, vmin, vmax, label=options.colorbar_label, tick_format=fmt),
        title=options.title,
        width_mm=float(width),
        height_mm=float(height),
        font_size=options.font_size,
        dpi=options.dpi,
    )


# ------------------------------------------------------------------ the material explorer


def material_line() -> str:
    """The material explorer's line color, the line palette's first."""
    return colormaps.line_palette()[0]


def material_component(view: MaterialView) -> int:
    """The row of ``view.n`` the selected component is."""
    components = list(view.menu.components) or ["n_x"]
    selected = view.menu.selected.get("component", components[0])
    return components.index(selected) if selected in components else 0


def material_extrapolated(view: MaterialView) -> bool:
    """Whether the requested range leaves the model's, so the extrapolated curve is drawn."""
    (low, high), (v0, v1) = view.requested, view.valid_range
    return low < v0 or high > v1


def material_ylim(view: MaterialView) -> Optional[Tuple[float, float]]:
    """The index range shown: both curves' extremes within the requested range, 5% margins.

    The extrapolated curve counts whether or not it is drawn, as the explorer
    window had it. None when nothing of the model's range is inside the
    requested one, where the axis is scaled to the curves drawn.
    """
    row = material_component(view)
    n = np.real(np.asarray(view.n)[row])
    low, high = view.requested
    w = np.asarray(view.wavelengths)
    inside = (w >= low) & (w <= high)
    parts = [n[inside]]
    if view.extended_wavelengths is not None and view.n_extended is not None:
        we = np.asarray(view.extended_wavelengths)
        ne = np.real(np.asarray(view.n_extended)[row])
        parts.append(ne[(we >= low) & (we <= high)])
    if parts[0].size == 0:
        return None
    ymin = min(float(parts[0].min()), *(float(np.nanmin(p)) for p in parts[1:] if p.size))
    ymax = max(float(parts[0].max()), *(float(np.nanmax(p)) for p in parts[1:] if p.size))
    # a constant index (a metal's tabulated value) gets a margin of its own size
    margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.05 * (abs(ymax) or 1.0)
    return ymin - margin, ymax + margin


def _autoscale(curves: List[np.ndarray]) -> Tuple[float, float]:
    values = np.concatenate([np.real(np.asarray(c, complex)).ravel() for c in curves])
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = 0.05 * (hi - lo) if hi > lo else 0.5
    return lo - pad, hi + pad


def _compose_material(view: MaterialView, options: PlotOptions) -> FigureSpec:
    row = material_component(view)
    lines: List[LineSpec] = []
    drawn: List[np.ndarray] = []
    if material_extrapolated(view) and view.extended_wavelengths is not None and view.n_extended is not None:
        ne = np.real(np.asarray(view.n_extended)[row])
        lines.append(LineSpec(np.asarray(view.extended_wavelengths, float), ne, "extrapolated", material_line(), LW, "dash"))
        drawn.append(ne)
    n = np.real(np.asarray(view.n)[row])
    lines.append(LineSpec(np.asarray(view.wavelengths, float), n, "index", material_line(), LW))
    drawn.append(n)
    ylim = material_ylim(view) or _autoscale(drawn)
    width, height = options.figure_size or DEFAULT_SIZE_MM
    texts = []
    if view.reference and options.text_display:
        texts.append(
            TextSpec(
                "\n".join(textwrap.wrap(view.reference, 110)), 0.0, -0.22, "axes", "left", "top",
                "#000000", options.font_size - 1, "footer",
            )
        )
    axes = AxesSpec(
        xlabel=options.xlabel or "Wavelength (nm)",
        ylabel=options.ylabel or "Refractive index (-)",
        xrange=(float(view.requested[0]), float(view.requested[1])),
        yrange=(float(ylim[0]), float(ylim[1])),
        grid=True,
    )
    return FigureSpec(
        axes=axes,
        lines=tuple(lines),
        texts=tuple(texts),
        title=options.title,
        width_mm=float(width),
        height_mm=float(height),
        font_size=options.font_size,
        dpi=options.dpi,
    )


# ------------------------------------------------------------------ the far field


@dataclass(frozen=True)
class FarFieldPanel:
    """A ``FarField`` and which of its panels is shown: ``'intensity'``, or ``'screen'`` when it has one."""

    farfield: Any
    panel: str = "intensity"

    @property
    def panels(self) -> List[str]:
        return ["intensity", "screen"] if self.farfield.screen is not None else ["intensity"]


def _compose_farfield(view: Any, options: PlotOptions, panel: str = "intensity") -> FigureSpec:
    width, height = options.figure_size or DEFAULT_SIZE_MM
    if panel == "screen" and view.screen is not None:
        screen = view.screen
        z = np.asarray(screen.Sn, float).T
        x = np.asarray(screen.s, float) / 1000
        y = np.asarray(screen.t, float) / 1000
        labels = ("u (μm)", "v (μm)")
        texts = (
            TextSpec(
                f"screen at {screen.screen.distance / 1000:g} μm, {screen.power_through_screen:.3g} W",
                0.02, 0.98, "axes", "left", "top", "#ffffff", options.font_size, "annotation",
            ),
        )
        bar_label = "flux (W/m²)"
    else:
        z = np.asarray(view.intensity, float).T
        x, y = np.asarray(view.ux, float), np.asarray(view.uy, float)
        labels = ("u_x = sin θx", "u_y = sin θy")
        texts = (
            TextSpec(
                f"divergence {view.divergence[0]:.2f}°, {view.divergence[1]:.2f}°\n"
                f"M² {view.m_squared[0]:.3f}, {view.m_squared[1]:.3f}",
                0.02, 0.98, "axes", "left", "top", "#ffffff", options.font_size, "annotation",
            ),
        )
        bar_label = "intensity (W/sr)"
    vmax = float(np.nanmax(z)) if z.size else 1.0
    vmin = 0.0 if options.vmin is None else float(options.vmin)
    vmax = vmax if options.vmax is None else float(options.vmax)
    cmap = _cmap("sequential", options)
    raster = RasterSpec(x=x, y=y, z=z, colormap=cmap, vmin=vmin, vmax=vmax)
    axes = AxesSpec(
        xlabel=options.xlabel or labels[0],
        ylabel=options.ylabel or labels[1],
        xrange=(float(x.min()), float(x.max())),
        yrange=(float(y.min()), float(y.max())),
        equal_aspect=True,
    )
    return FigureSpec(
        axes=axes,
        rasters=(raster,),
        texts=texts if options.text_display else (),
        colorbar=ColorbarSpec(cmap, vmin, vmax, label=options.colorbar_label or bar_label, tick_format="%.3g"),
        title=options.title or f"far field, {view.wavelength:g} nm",
        width_mm=float(width),
        height_mm=float(height),
        font_size=options.font_size,
        dpi=options.dpi,
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
