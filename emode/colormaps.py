"""The colormap family every plot of this client draws with.

A panel never names a colormap. Its rasters say which kind they want
(``'sequential'`` for magnitudes and the index, ``'diverging'`` for real and
imaginary parts, ``'cyclic'`` for phase), and this module is the one place a
kind becomes colors: stop lists for the browser (``plotly_colorscale``) and
registered matplotlib colormaps for static images (``mpl_colormap``), built
from the same stops so the two agree.

Until EMode's own family arrives (emodeconnection#120) the stops are
matplotlib's plasma. The diverging map is plasma mirrored about zero, which is
how EMode's plot window has always colored a signed field: the color shows the
magnitude and the colorbar the sign. The cyclic map runs plasma up and back
down so its two ends meet. Replacing the stops below changes every plot of
this client and nothing else.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: plasma, 17 stops from 0 to 1.
_PLASMA = [
    "#0d0887", "#310597", "#4c02a1", "#6600a7", "#7e03a8", "#9511a1",
    "#aa2395", "#bc3587", "#cc4778", "#da5a6a", "#e66c5c", "#f0804e",
    "#f89540", "#fdac33", "#fdc527", "#f8df25", "#f0f921",
]

#: The line palette, for shapes and legend entries drawn as lines.
LINE_PALETTE = ["#0d0887", "#9511a1", "#e66c5c", "#fdac33", "#2b5c9e", "#4a4a4a"]

KINDS = ("sequential", "diverging", "cyclic")

#: The matplotlib name each kind is registered under.
MPL_NAMES = {kind: f"emode-{kind}" for kind in KINDS}


def _even(colors: List[str]) -> List[Tuple[float, str]]:
    n = len(colors) - 1
    return [(i / n, c) for i, c in enumerate(colors)]


def stops(kind: str) -> List[Tuple[float, str]]:
    """The ``(position, '#rrggbb')`` stops of `kind`, positions from 0 to 1."""
    if kind == "sequential":
        return _even(_PLASMA)
    if kind == "diverging":
        return _even(_PLASMA[::-1] + _PLASMA[1:])
    if kind == "cyclic":
        return _even(_PLASMA + _PLASMA[-2::-1])
    raise ValueError(f"unknown colormap kind {kind!r}, expected one of {KINDS}")


def plotly_colorscale(kind: str) -> List[List[Any]]:
    """`kind` as a Plotly colorscale, ``[[position, color], ...]``."""
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


def _rgb(color: str) -> Tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def all_stops() -> Dict[str, Any]:
    """Every kind's stops and the line palette, as the browser tab asks for them."""
    return {"kinds": {k: stops(k) for k in KINDS}, "lines": LINE_PALETTE}


def mpl_colormap(kind: str, name: Optional[str] = None) -> Any:
    """The matplotlib colormap of `kind`, or the colormap registered as `name`.

    The family is registered with matplotlib under ``MPL_NAMES`` the first time
    it is asked for. The sequential kind is matplotlib's own plasma until the
    family arrives, so a static image matches the window it replaces.
    """
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    if name is not None:
        return matplotlib.colormaps[name]
    registered = MPL_NAMES[kind]
    if registered not in matplotlib.colormaps:
        if kind == "sequential":
            cmap = matplotlib.colormaps["plasma"].copy()
            cmap.name = registered
        elif kind == "diverging":
            # mirrored as the window built it: 128 samples down, 128 up
            import numpy as np

            plasma = matplotlib.colormaps["plasma"].resampled(128)
            colors = plasma(np.linspace(0, 1, 128))
            cmap = ListedColormap(np.concatenate((np.flip(colors, axis=0), colors)), name=registered)
        else:
            cmap = LinearSegmentedColormap.from_list(registered, stops(kind))
        matplotlib.colormaps.register(cmap, name=registered)
    return matplotlib.colormaps[registered]
