"""What ``plot()`` gets back from EMode: the data of a panel, not a picture.

From EMode 1.0.9 the solver no longer draws. ``EM_plot`` answers with a
``PlotView`` (a field, index, shape or outline panel on the x-y, z-x or z-y
plane) or a ``Schematic`` (the circuit as blocks), and this client turns it
into a figure (``emode.ui.figure.compose``) and draws it, in the browser tab
or as a static image.

Everything here is geometry and values, never colors or fonts:

- Every coordinate is in nm, on the panel's own axes. ``AxesInfo`` says which
  unit each axis is shown in by default (``'nm'`` or ``'um'``).
- Every array is float32, the display precision, at half the bytes.
- A raster's ``z`` is the number the cursor readout shows, after the component
  (real, imag, abs, abs^2, log in dB) and the normalization. Its ``kind`` says
  which colormap of the family it wants (sequential, diverging, cyclic), never
  a name.
- The overlays are in the order they are drawn, so a static image and the tab
  agree.

``MenuState`` is what the plot's controls offer, resolved by EMode, so the tab
shows the same choices for every session.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import ConfigDict, field_validator

from .types import TaggedModel, register_type

RasterKind = Literal["sequential", "diverging", "cyclic"]


def _float32(value: object) -> object:
    """An array as float32, anything else unchanged (pydantic validates the rest)."""
    if isinstance(value, np.ndarray) and value.dtype != np.float32 and value.dtype.kind in "fiu":
        return value.astype(np.float32)
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, (int, float)) for v in value):
        return np.asarray(value, dtype=np.float32)
    return value


class _Arrays(TaggedModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


@register_type
class Raster(_Arrays):
    """One image of a panel: ``z`` on the cell centers ``x`` (columns) and ``y`` (rows), in nm.

    ``x`` and ``y`` are monotonic, uniform or not. ``mask`` is True where there is
    nothing to show (outside the chip in the layout view). ``contour_levels`` are
    the levels the contours are drawn at, when the panel has contours.
    ``interpolation`` is how the image is smoothed between the centers.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    mask: Optional[np.ndarray] = None
    kind: RasterKind = "sequential"
    contour_levels: Optional[np.ndarray] = None
    interpolation: Literal["nearest", "bilinear"] = "nearest"

    _as_float32 = field_validator("x", "y", "z", "contour_levels", mode="before")(
        lambda cls, v: _float32(v)
    )


@register_type
class Polyline(_Arrays):
    """A line of the panel in nm: an index outline, a port, a facet, a monitor plane.

    ``kind`` is what it is (``'outline'``, ``'port'``, ``'facet'``, ``'monitor'``,
    ``'connection'``, ``'loop'``), and the figure styles it by kind. ``label`` is
    written at ``label_at`` (nm), aligned by ``align``.
    """

    kind: str
    x: np.ndarray
    y: np.ndarray
    label: Optional[str] = None
    label_at: Optional[Tuple[float, float]] = None
    align: Literal["left", "right", "center"] = "left"

    _as_float32 = field_validator("x", "y", mode="before")(lambda cls, v: _float32(v))


@register_type
class Region(_Arrays):
    """A filled region of the panel in nm.

    ``kind`` is ``'fill'`` (a shape filled with its index or its shape color),
    ``'background'`` (the panel behind the fills), ``'window'`` (the
    simulation window's frame), ``'pml'`` or ``'symmetry'`` (hatched bands).
    ``value`` is the fill's place on the colormap, 0 to 1, and ``color`` an
    explicit color that overrides it (``'grey'`` for a perfect conductor).
    ``holes`` are the interiors, each an ``(n, 2)`` array.
    """

    kind: str
    x: np.ndarray
    y: np.ndarray
    holes: List[np.ndarray] = []
    value: Optional[float] = None
    color: Optional[str] = None
    name: Optional[str] = None
    hatch: Optional[str] = None

    _as_float32 = field_validator("x", "y", mode="before")(lambda cls, v: _float32(v))


@register_type
class Segments(_Arrays):
    """Many straight segments, ``xy`` of shape ``(n, 2, 2)`` in nm: the mesh, the section boundaries."""

    kind: str
    xy: np.ndarray
    alpha: float = 0.25

    _as_float32 = field_validator("xy", mode="before")(lambda cls, v: _float32(v))


@register_type
class Label(TaggedModel):
    """A text of the panel, at ``(x, y)`` in axes fractions (``frame='axes'``) or nm (``'data'``)."""

    model_config = ConfigDict(frozen=True)

    text: str
    x: float
    y: float
    frame: Literal["axes", "data"] = "axes"
    kind: str = "title"
    ha: Literal["left", "right", "center"] = "left"
    va: Literal["top", "bottom", "center", "baseline"] = "bottom"
    color: Optional[str] = None


@register_type
class LegendEntry(TaggedModel):
    """A shape in the legend, with its place on the colormap when it is filled."""

    model_config = ConfigDict(frozen=True)

    label: str
    value: Optional[float] = None


@register_type
class AxesInfo(TaggedModel):
    """The panel's axes: default names and units, limits in nm, orientation and aspect.

    ``xlabel`` and ``ylabel`` are the axis names (``'z'``, ``'x'``, ``'y'``),
    ``x_unit`` and ``y_unit`` the units they are shown in by default.
    ``xlim`` and ``ylim`` are the panel's limits in nm, low to high, and
    ``invert_y`` puts the low end at the top. ``equal_aspect`` keeps one nm the
    same length on both axes, and ``box_aspect`` is the plot box's height over
    its width when the aspect is free.
    """

    model_config = ConfigDict(frozen=True)

    xlabel: str
    ylabel: str
    x_unit: Literal["nm", "um"] = "nm"
    y_unit: Literal["nm", "um"] = "nm"
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    invert_y: bool = False
    equal_aspect: bool = False
    box_aspect: Optional[float] = None


@register_type
class ScaleInfo(TaggedModel):
    """How the panel's values map onto the colorbar.

    ``kind`` is ``'positive'`` (0 to 1), ``'signed'`` (-1 to 1), ``'db'`` (in dB,
    ``vmin`` to 0), ``'index'`` (the fills' indices as ticks) or ``'none'`` (no
    colorbar). ``reference`` is the absolute value a normalized 1 stands for,
    which the cursor readout multiplies by. ``ticks`` and ``tick_labels`` replace
    the colorbar's own ticks when set (the index panel's indices).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["positive", "signed", "db", "index", "none"]
    vmin: float = 0.0
    vmax: float = 1.0
    reference: Optional[float] = None
    tick_format: str = "%3.1f"
    ticks: Optional[List[float]] = None
    tick_labels: Optional[List[str]] = None


@register_type
class Slider(TaggedModel):
    """The third-dimension control: its axis, range in the shown unit, value and preset entries."""

    model_config = ConfigDict(frozen=True)

    label: str
    min: float
    max: float
    unit: str
    value: str
    options: List[str] = []


@register_type
class MenuState(TaggedModel):
    """What the plot's controls offer, and what is selected in each.

    ``selected`` holds the resolved selection by control name (``'data'``,
    ``'component'``, ``'normalization'``, ``'mode'``, ``'plane'``, ``'port'``,
    ``'section'``, ``'profile'``, ``'z_resolution'``), so a label the client
    sent is echoed back as EMode resolved it. ``settings`` holds the eight
    settings as resolved, and ``active_settings`` those the panel lets the user
    change.
    """

    model_config = ConfigDict(frozen=True)

    data: List[str] = []
    components: List[str] = []
    normalizations: List[str] = []
    modes: List[str] = []
    planes: List[str] = []
    ports: List[str] = []
    sections: List[str] = []
    profiles: Optional[List[str]] = None
    wavelengths: List[float] = []
    z_resolutions: List[str] = []
    slider: Optional[Slider] = None
    settings: Dict[str, bool] = {}
    active_settings: List[str] = []
    selected: Dict[str, str] = {}


@register_type
class Arrow(TaggedModel):
    """An arrow at the panel's edge to a neighbor of the selected section, in axes fractions."""

    model_config = ConfigDict(frozen=True)

    direction: Literal["left", "right", "up", "down"]
    target: str
    x: float
    y: float


@register_type
class PlotView(_Arrays):
    """A field, index, shape or outline panel, as data.

    - ``plane``: ``'x-y'`` (a cross-section), ``'z-x'`` (top view) or ``'z-y'``
      (side view). ``view``: ``'layout'`` (on the chip) or ``'simulation'``
      (unrolled along the chain). ``state``: ``'solved'``, ``'preview'``
      (declared, not solved) or ``'profile'`` (a mode solve's profile).
    - ``rasters``: the panel's images, one per slice on a z panel.
    - ``background``: the colormap place of the panel behind the rasters and
      fills, when it is colored.
    - ``fills``, ``outlines``, ``ports``, ``facets``, ``monitors``, ``window``,
      ``mesh``: the overlays, every one returned whatever the settings, so a
      toggle redraws without asking EMode.
    - ``texts``, ``legend``, ``axes``, ``scale``, ``menu``: the annotations, the
      shapes' names, the axes, the colorbar's scale and the controls.
    - ``sections``: the circuit's section names, ``selected_section`` the one
      the panel is zoomed to, ``arrows`` the way to its neighbors.
    - ``wavelength`` in nm, ``z_resolution_nm`` the z sampling used.
    """

    plane: Literal["x-y", "z-x", "z-y"]
    view: Literal["layout", "simulation"] = "layout"
    state: Literal["solved", "preview", "profile"] = "solved"
    rasters: List[Raster] = []
    background: Optional[float] = None
    fills: List[Region] = []
    outlines: List[Polyline] = []
    ports: List[Polyline] = []
    facets: List[Polyline] = []
    monitors: List[Polyline] = []
    window: List[Region] = []
    mesh: List[Segments] = []
    texts: List[Label] = []
    legend: List[LegendEntry] = []
    axes: AxesInfo
    scale: ScaleInfo
    menu: MenuState = MenuState()
    sections: List[str] = []
    selected_section: Optional[str] = None
    arrows: List[Arrow] = []
    wavelength: Optional[float] = None
    z_resolution_nm: Optional[float] = None


@register_type
class Block(TaggedModel):
    """A section of the schematic: its name, its label lines and its box, in layout units."""

    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    x: float
    y: float
    width: float
    height: float


@register_type
class Stub(TaggedModel):
    """A port's short line on a block's side, from ``(x0, y0)`` at the block to ``(x1, y1)``.

    ``label`` is the port's name where one is shown, ``external`` whether it is a
    port of the circuit (labeled outside, bold when launched).
    """

    model_config = ConfigDict(frozen=True)

    section: str
    side: Literal["left", "right"]
    x0: float
    y0: float
    x1: float
    y1: float
    label: Optional[str] = None
    external: bool = False


@register_type
class Schematic(_Arrays):
    """The circuit as a block diagram, in layout units (not nm).

    Blocks run left to right along the light path. ``connections`` are the lines
    between stubs (``kind`` ``'connection'``, or ``'loop'`` for one that closes
    a loop around the top). ``launch`` is the launched port's stub. ``monitors``
    label the sections a monitor sits on. ``limits`` is ``(xmin, xmax, ymin, ymax)``
    of the whole diagram, ``frame`` the zoom to the selected section and its
    neighbors. ``neighbors`` maps each section to the section each arrow key leads to.
    """

    blocks: List[Block] = []
    stubs: List[Stub] = []
    connections: List[Polyline] = []
    launch: Optional[Stub] = None
    monitors: List[Label] = []
    limits: Tuple[float, float, float, float]
    frame: Optional[Tuple[float, float, float, float]] = None
    selected_section: Optional[str] = None
    neighbors: Dict[str, Dict[str, str]] = {}
    arrows: List[Arrow] = []
    menu: MenuState = MenuState()
