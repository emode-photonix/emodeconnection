"""EMode's colormap family, which every plot of this client and of EMode draws with.

The family has three colormaps and a line palette:

- ``sequential``, for magnitudes, intensities and the index,
- ``diverging``, for real and imaginary parts (signed values about zero),
- ``cyclic``, for phase, whose two ends meet,
- the line palette, for curves, the schematic and legend entries.

A panel never names a colormap. Its rasters say which kind they want, and
this module is the one place a kind becomes colors: stop lists for the browser
(``plotly_colorscale``) and colormaps registered with matplotlib under
``NAMES`` for static images (``get``), built from the same stops so the two
agree. ``plot(colormap=...)`` picks another map by name (``check_colormap``).

**The drop-in.** The colors are ``delivered()``, and nothing else in EMode
holds one. EMode's own family replaces the body of that function: the
sequential stops at least, and the diverging and cyclic stops and the line
colors where they are delivered. A kind that is not delivered is derived from
the sequential map (``family()``). Until then the stops are placeholders,
matplotlib's plasma, which is what EMode's plot window drew.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .types import ArgumentError, EModeError

KINDS = ("sequential", "diverging", "cyclic")

#: The matplotlib name each kind is registered under.
MPL_NAMES = {kind: f"emode-{kind}" for kind in KINDS}

#: Every registered name, in the order of ``KINDS``.
NAMES = tuple(MPL_NAMES.values())

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

# ------------------------------------------------------------------ the placeholders

#: PLACEHOLDER: matplotlib's plasma at 33 evenly spaced stops, dark to light.
_PLACEHOLDER_SEQUENTIAL = (
    "#0d0887", "#220690", "#310597", "#3f049c", "#4c02a1", "#5901a5", "#6600a7",
    "#7201a8", "#7e03a8", "#8a09a5", "#9511a1", "#a01a9c", "#aa2395", "#b32c8e",
    "#bc3587", "#c43e7f", "#cc4778", "#d35171", "#da5a6a", "#e06363", "#e66c5c",
    "#eb7655", "#f0804e", "#f58b47", "#f89540", "#fba139", "#fdac33", "#feb82c",
    "#fdc527", "#fcd225", "#f8df25", "#f4ed27", "#f0f921",
)

#: PLACEHOLDER: the line palette. EMode's blue first (the material explorer's
#: curve), then the schematic's launch red and monitor magenta, a teal, an
#: amber and a neutral gray, each at least 20 apart in CIEDE2000.
_PLACEHOLDER_LINES = ("#3634A3", "#d1495b", "#b5179e", "#2a9d8f", "#e9a23b", "#4a4a4a")


@dataclass(frozen=True)
class Family:
    """The family's colors as ``'#rrggbb'`` stops, evenly spaced from 0 to 1.

    ``diverging`` and ``cyclic`` may be None, and are then derived from
    ``sequential``: the diverging map runs it backward to its dark end at
    zero and forward again, so a value's magnitude has the same color on
    either side of zero (EMode's plot window colored a signed field this way),
    and the cyclic map runs it forward and back so its ends meet.
    """

    sequential: tuple[str, ...]
    diverging: tuple[str, ...] | None = None
    cyclic: tuple[str, ...] | None = None
    lines: tuple[str, ...] = _PLACEHOLDER_LINES


def delivered() -> Family:
    """The family's colors. THE DROP-IN POINT: EMode's colormap replaces this body.

    Return a ``Family`` with the delivered stops (``'#rrggbb'``, evenly spaced,
    the sequential map from its dark end to its light end), leaving out what
    is not delivered. Then regenerate the plot snapshots and the docs' figures
    once. ``tests/test_colormaps.py`` checks what is delivered: the sequential
    map monotonic in lightness, the diverging map symmetric in lightness, the
    cyclic map's ends meeting and the line colors distinguishable.
    """
    return Family(sequential=_PLACEHOLDER_SEQUENTIAL, lines=_PLACEHOLDER_LINES)


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


def _tint(color: str, white: float) -> str:
    """`color` mixed with white, `white` the white's share (0 to 1)."""
    return "#" + "".join(f"{round(c + white * (255 - c)):02x}" for c in _rgb(color))


def _line(i: int) -> str:
    lines = line_palette()
    return lines[i % len(lines)]


def schematic_colors() -> dict[str, str]:
    """The schematic's colors, from the line palette: blocks, launch, monitors, connections."""
    return {
        "block_edge": _line(0),
        "block_face": _tint(_line(0), 0.88),
        "text": "#1f2933",
        "launch": _line(1),
        "monitor": _line(2),
        "connection": _line(5),
    }


def overlay_colors() -> dict[str, str]:
    """The colors of the lines drawn over a field panel and between schematic blocks.

    The overlays are bright and saturated so they read on any colormap, and are
    not taken from the line palette.
    """
    return {
        "outline": "#ffffff",
        "port": "#00ff00",
        "facet": "#00bfbf",
        "monitor": "#bf00bf",
        "connection": _line(5),
        "loop": _line(5),
        "stub": _line(5),
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
            cmap = LinearSegmentedColormap.from_list(name, stops(kind), N=256)
            matplotlib.colormaps.register(cmap, name=name)
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
        return matplotlib.colormaps[MPL_NAMES[kind]]
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
        f"one of EMode's colormaps ({', '.join(NAMES)}), a kind ({', '.join(KINDS)}), "
        "or the name of a colormap matplotlib has registered"
    )
    if isinstance(name, str) and kind_of(name) is not None:
        return name
    if not isinstance(name, str) or name not in _matplotlib().colormaps:
        raise ArgumentError(f"colormap must be {choices}, got {name!r}", function, "colormap")
    return name
