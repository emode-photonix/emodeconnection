"""EMode's colormap family, which every plot of this client and of EMode draws with.

The family has three colormaps and a line palette:

- ``sequential``, the brand's "emode" map, for magnitudes, intensities and the index,
- ``diverging``, the brand's "emode_div" map, for real and imaginary parts (signed
  values about zero, drawn with a symmetric range so zero sits at the center, #1F2D50),
- ``cyclic``, for phase, whose two ends meet (derived, since no cyclic map was delivered),
- the line palette (``SERIES``), for curves, the schematic and legend entries.

A panel never names a colormap. Its rasters say which kind they want, and
this module is the one place a kind becomes colors: stop lists for the browser
(``plotly_colorscale``) and colormaps registered with matplotlib under
``NAMES`` for static images (``get``), built from the same stops so the two
agree. ``plot(colormap=...)`` picks another map by name (``check_colormap``).
"emode" and "emode_div" are registered too, as aliases of the first two, so a
script's own matplotlib figure can name them as the brand's files do.

**The colors.** ``BRAND``, ``SERIES``, ``SERIES_DARK``, ``GUI`` and ``PLOT`` are the
brand palette (``emode_palette.json``) and the two maps are the brand's colormap files,
loaded verbatim into ``emode._brand`` by ``scripts/sync_brand_colors.py``. Nothing in
EMode invents or mixes a color: every color a plot or the
browser interface draws is one of these. ``tests/test_colormaps.py`` compares them with the
brand files when those are present. ``delivered()`` is the one function that gathers
the delivery, and ``family()`` fills in what is not delivered.

**Series.** Curves take the line colors in order, slot 1, 2, 3 and so on, and never
cycle (``series_color``). At most 8 are drawn in a color: every further series is folded
into one "Other" series drawn in Mist, with a single legend entry (``series_label``). A
series keeps its color when others are filtered out, as its slot is its own index. A
scatter plot, a map or a small multiple uses only the first 3 (``SCATTER_SERIES``).
Status colors (success, error) are never series colors, and the GUI is light mode only.
"""
from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from typing import Any

# The brand's colors, generated from its color files by scripts/sync_brand_colors.py.
from ._brand import BRAND, GUI, PLOT, RULES, SERIES, SERIES_DARK, SERIES_NAMES  # noqa: F401 (re-exported)
from ._brand import EMODE as _EMODE
from ._brand import EMODE_DIV as _EMODE_DIV
from .types import ArgumentError, EModeError

KINDS = ("sequential", "diverging", "cyclic")

#: The matplotlib name each kind is registered under.
MPL_NAMES = {kind: f"emode-{kind}" for kind in KINDS}

#: Every registered name, in the order of ``KINDS``.
NAMES = tuple(MPL_NAMES.values())

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

# ------------------------------------------------------------------ the brand's colors

#: The kind each brand colormap name is an alias of.
ALIASES = {"emode": "sequential", "emode_div": "diverging"}

#: The color of every series past the 8th, folded into one "Other" series (Mist).
OTHER = BRAND["Mist"]

#: How many series a line plot colors before the rest fold into "Other".
MAX_SERIES = 8

#: How many series a scatter plot, a map or a small multiple colors, as any two colors can touch.
SCATTER_SERIES = 3


@dataclass(frozen=True)
class Family:
    """The family's colors as ``'#rrggbb'`` stops, evenly spaced from 0 to 1.

    ``diverging`` and ``cyclic`` may be None, and are then derived from
    ``sequential``: the diverging map runs it backward to its dark end at
    zero and forward again, so a value's magnitude has the same color on
    either side of zero, and the cyclic map runs it forward and back so its
    ends meet. The brand delivered a diverging map but no cyclic one.
    """

    sequential: tuple[str, ...]
    diverging: tuple[str, ...] | None = None
    cyclic: tuple[str, ...] | None = None
    lines: tuple[str, ...] = tuple(SERIES)


def delivered() -> Family:
    """The brand's colors: the "emode" and "emode_div" stops and the 8 series colors.

    The cyclic map is not delivered, so it derives from the sequential one
    (``family()``). ``tests/test_colormaps.py`` holds what is delivered to the
    brand files and to what a plot needs: the sequential map monotonic in
    lightness, the diverging map symmetric about its center, and the line colors
    distinguishable as the brand palette states it.
    """
    return Family(sequential=_EMODE, diverging=_EMODE_DIV, lines=tuple(SERIES))


# ------------------------------------------------------------------ the family


def family() -> Family:
    """The delivered colors with every kind filled in, checked."""
    given = delivered()
    sequential = _checked(given.sequential, "sequential")
    diverging = given.diverging or sequential[::-1] + sequential[1:]
    cyclic = given.cyclic or sequential + sequential[-2::-1]
    return Family(
        sequential=sequential,
        diverging=_checked(diverging, "diverging"),
        cyclic=_checked(cyclic, "cyclic"),
        lines=_checked(given.lines, "lines", minimum=1),
    )


def _checked(colors: tuple[str, ...], what: str, minimum: int = 2) -> tuple[str, ...]:
    colors = tuple(colors)
    if len(colors) < minimum:
        raise ValueError(f"the {what} colors need at least {minimum}, got {len(colors)}")
    for color in colors:
        if not isinstance(color, str) or not _HEX.match(color):
            raise ValueError(f"the {what} colors are '#rrggbb' strings, got {color!r}")
    return colors


def kind_of(name: str) -> str | None:
    """The kind `name` is (a kind or a registered name), or None for any other map."""
    if name in KINDS:
        return name
    if name in ALIASES:
        return ALIASES[name]
    for kind, registered in MPL_NAMES.items():
        if name == registered:
            return kind
    return None


def _even(colors: tuple[str, ...]) -> list[tuple[float, str]]:
    n = len(colors) - 1
    return [(i / n, c) for i, c in enumerate(colors)]


def stops(kind: str) -> list[tuple[float, str]]:
    """The ``(position, '#rrggbb')`` stops of `kind` (or its registered name), positions 0 to 1."""
    resolved = kind_of(kind)
    if resolved is None:
        raise ValueError(f"unknown colormap kind {kind!r}, expected one of {KINDS}")
    return _even(getattr(family(), resolved))


def line_palette() -> tuple[str, ...]:
    """The line colors, in the order curves take them."""
    return family().lines


def series_color(index: int, limit: int | None = None) -> str:
    """The color of the series at `index` (0 for the first): its slot's, never cycling.

    A series at or past `limit` (``MAX_SERIES`` unless a scatter plot, a map or small
    multiples ask for ``SCATTER_SERIES``) is part of the one "Other" series, in Mist.
    """
    lines = line_palette()
    cap = min(len(lines), MAX_SERIES if limit is None else limit)
    return lines[index] if 0 <= index < cap else OTHER


def series_label(index: int, label: str | None, limit: int | None = None) -> str | None:
    """`label` for a series that has its own color, "Other" for the first folded one, else None.

    The folded series share one legend entry, and a series without a label has none.
    """
    cap = min(len(line_palette()), MAX_SERIES if limit is None else limit)
    if index < cap:
        return label
    return "Other" if index == cap and label is not None else None


def figure_style(active: bool = True) -> Any:
    """A context in which a new matplotlib figure is light (``plot_style``), or one that changes nothing.

    The rc parameters change only inside the ``with`` block, and are restored when it ends.
    """
    if not active:
        return contextlib.nullcontext()
    return _matplotlib().rc_context(plot_style())


def plot_style(mode: str = "light") -> dict[str, Any]:
    """The matplotlib rc parameters of a brand figure: background, text, axes and grid from ``PLOT``.

    ``mode`` is "light" (Chalk, Night Ink text, Charcoal axes, Haze grid), the default and
    the only mode the interface uses, or "dark" for dark slides and figures. EMode's own
    plots apply it with ``matplotlib.rc_context``, never to the global ``rcParams``, since
    this client runs in the user's own Python.
    """
    if mode not in PLOT:
        raise ArgumentError(f"mode is 'light' or 'dark', got {mode!r}", "plot_style", "mode")
    colors = PLOT[mode]
    return {
        "figure.facecolor": colors["background"],
        "axes.facecolor": colors["background"],
        "savefig.facecolor": colors["background"],
        "text.color": colors["text"],
        "axes.labelcolor": colors["text"],
        "axes.titlecolor": colors["text"],
        "xtick.color": colors["axes"],
        "ytick.color": colors["axes"],
        "xtick.labelcolor": colors["text"],
        "ytick.labelcolor": colors["text"],
        "axes.edgecolor": colors["axes"],
        "grid.color": colors["grid"],
        "legend.frameon": False,
    }


def use_style(mode: str = "light") -> None:
    """Style the user's own matplotlib figures with the brand colors, as the brand's ``use_style`` does.

    Registers "emode" and "emode_div" (and the ``NAMES``), makes "emode" the default image
    colormap, and sets the background, text, axes, grid and series colors of ``mode``
    ("light" or "dark") in the global ``rcParams``. Call it once in a script; EMode's own
    plots never call it. Matplotlib's color cycle repeats after the 8th line, so a plot of
    more than 8 series should color them with ``series_color``, which folds the rest into
    "Other".
    """
    style = plot_style(mode)
    matplotlib = _matplotlib()
    from cycler import cycler

    register()
    matplotlib.rcParams.update({
        **style,
        "grid.linewidth": 0.8,
        "axes.prop_cycle": cycler(color=PLOT[mode]["series"]),
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "image.cmap": "emode",
    })


def plotly_colorscale(kind: str) -> list[list[Any]]:
    """`kind` (or its registered name) as a Plotly colorscale, ``[[position, color], ...]``."""
    return [[p, c] for p, c in stops(kind)]


def color_at(kind: str, fraction: float) -> str:
    """The color `fraction` (0 to 1) of the way along `kind`, linearly between its stops."""
    points = stops(kind)
    f = min(1.0, max(0.0, float(fraction)))
    for (p0, c0), (p1, c1) in zip(points, points[1:]):
        if f <= p1:
            t = 0.0 if p1 == p0 else (f - p0) / (p1 - p0)
            a, b = _rgb(c0), _rgb(c1)
            return "#" + "".join(f"{round(x + t * (y - x)):02x}" for x, y in zip(a, b))
    return points[-1][1]


def _rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def schematic_colors() -> dict[str, str]:
    """The schematic's colors, all palette colors: blocks, launch, monitors, connections."""
    return {
        "text": GUI["text"],
        "block_face": BRAND["Haze"],
        "block_edge": SERIES[0],
        "launch": SERIES[1],
        "monitor": SERIES[2],
        "connection": BRAND["Charcoal"],
    }


def overlay_colors() -> dict[str, str]:
    """The colors of the lines drawn over a field panel and between schematic blocks.

    The overlays sit on the field maps, whose background is the dark end, so they take the
    series colors stepped for dark backgrounds: the port is slot 3 (teal), the facet slot 1
    (blue), the monitor slot 4 (coral) and the slice slot 2 (orange). The waveguide outline is
    white, as in the brand's reference mode plot, and the connections are Haze.
    """
    return {
        "outline": "#ffffff",
        "port": SERIES_DARK[2],
        "facet": SERIES_DARK[0],
        "monitor": SERIES_DARK[3],
        "slice": SERIES_DARK[1],
        "connection": BRAND["Haze"],
        "loop": BRAND["Haze"],
        "stub": BRAND["Haze"],
    }


def all_stops() -> dict[str, Any]:
    """Every kind's stops, the registered names and the line palette, as the browser tab asks for them."""
    return {
        "kinds": {k: stops(k) for k in KINDS},
        "names": dict(MPL_NAMES),
        "lines": list(line_palette()),
    }


# ------------------------------------------------------------------ matplotlib


def _matplotlib() -> Any:
    try:
        import matplotlib
    except ImportError as error:
        raise EModeError(
            "Drawing a figure here needs matplotlib. Install it with `pip install matplotlib`."
        ) from error
    return matplotlib


def register() -> tuple[str, ...]:
    """Register the family with matplotlib under ``NAMES`` (once), and return the names."""
    matplotlib = _matplotlib()
    from matplotlib.colors import LinearSegmentedColormap

    for kind, name in MPL_NAMES.items():
        if name not in matplotlib.colormaps:
            points = stops(kind)
            cmap = LinearSegmentedColormap.from_list(name, points, N=max(256, len(points)))
            matplotlib.colormaps.register(cmap, name=name)
    for alias, kind in ALIASES.items():
        # the brand's own files register these names too: the first to arrive wins
        if alias not in matplotlib.colormaps:
            matplotlib.colormaps.register(matplotlib.colormaps[MPL_NAMES[kind]], name=alias)
    return NAMES


def get(name: str = "sequential") -> Any:
    """The matplotlib colormap of a kind, a registered name, or any colormap matplotlib knows.

    For a figure of your own in EMode's colors, pass
    ``cmap=emode.colormaps.get('sequential')``.
    """
    matplotlib = _matplotlib()
    kind = kind_of(name)
    if kind is not None:
        register()
        return matplotlib.colormaps[name if name in ALIASES else MPL_NAMES[kind]]
    return matplotlib.colormaps[name]


def mpl_colormap(kind: str, name: str | None = None) -> Any:
    """The matplotlib colormap of `kind`, or of `name` when one is given."""
    return get(kind if name is None else name)


def resolve(kind: str, colormap: str | None = None) -> str:
    """The name a raster of `kind` is drawn with: `colormap` when given, else the family's."""
    return colormap if colormap is not None else MPL_NAMES[kind]


def check_colormap(name: Any, function: str = "plot") -> str:
    """`name` if a plot can draw with it, else an ``ArgumentError`` naming the choices.

    The family's names and kinds need nothing else. Any other name must be a
    colormap matplotlib has registered.
    """
    choices = (
        f"one of EMode's colormaps ({', '.join(NAMES + tuple(ALIASES))}), a kind ({', '.join(KINDS)}), "
        "or the name of a colormap matplotlib has registered"
    )
    if isinstance(name, str) and kind_of(name) is not None:
        return name
    if not isinstance(name, str) or name not in _matplotlib().colormaps:
        raise ArgumentError(f"colormap must be {choices}, got {name!r}", function, "colormap")
    return name
