"""An ``emode.Figure`` as a ``FigureSpec`` of panels.

Each panel starts from its result's own plot (``compose_sweep``) when it has
one, then takes the script's curves, reference lines and bands, then its
labels and ranges. The figure's spec holds the panels in row-major order, and
both renderers lay them out on one page.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Optional, Tuple

import numpy as np

from .. import colormaps
from .figure import DEFAULT_SIZE_MM, LW, AxesSpec, FigureSpec, FillSpec, LineSpec, PlotOptions

#: A style string's markers, as the spec names them.
MARKERS = {"o": "circle", "s": "square", "^": "triangle-up", "D": "diamond", "d": "diamond",
           ">": "triangle-right", "<": "triangle-left"}

#: A style string's dashes.
DASHES = (("--", "dash"), (":", "dot"), ("-", None))


def parse_style(style: str) -> Tuple[Optional[str], Optional[str], bool]:
    """``(marker, dash, has_line)`` of a matplotlib-like style string (``'o-'``, ``'--'``, ``'s'``)."""
    marker = next((MARKERS[c] for c in style if c in MARKERS), None)
    rest = "".join(c for c in style if c not in MARKERS)
    for token, dash in DASHES:
        if token in rest:
            return marker, dash, True
    return marker, None, marker is None


def _color(color: Any, i: int) -> str:
    palette = colormaps.line_palette()
    if color is None:
        return palette[i % len(palette)]
    if isinstance(color, int):
        return palette[color % len(palette)]
    return str(color)


def _tint(color: str, white: float = 0.75) -> str:
    """`color` mixed with `white` of white, for a band behind the curves."""
    try:
        from matplotlib.colors import to_hex, to_rgb

        r, g, b = to_rgb(color)
    except (ImportError, ValueError):
        return color
    return to_hex(tuple(c + (1 - c) * white for c in (r, g, b)))


def _span(values: List[np.ndarray], pad: float = 0.05, log: bool = False) -> Optional[Tuple[float, float]]:
    """The padded range of `values`. On a log axis only the positive values count and the pad is in decades."""
    finite = [v[np.isfinite(v) & (v > 0 if log else True)] for v in values if v.size]
    finite = [v for v in finite if v.size]
    if not finite:
        return None
    lo = min(float(v.min()) for v in finite)
    hi = max(float(v.max()) for v in finite)
    if log:
        lo, hi = float(np.log10(lo)), float(np.log10(hi))
        width = hi - lo if hi > lo else 1.0
        return 10 ** (lo - pad * width), 10 ** (hi + pad * width)
    width = hi - lo if hi > lo else (abs(hi) or 1.0)
    return lo - pad * width, hi + pad * width


def compose_panel(panel: Any, options: PlotOptions, size: Tuple[float, float]) -> FigureSpec:
    """One panel of a ``Figure`` as a ``FigureSpec``."""
    from .sweep_figure import compose_sweep, sweep_panel

    base: Optional[FigureSpec] = None
    if panel.source is not None:
        base = compose_sweep(sweep_panel(panel.source, **panel.selection), replace(options, title=None))
    lines: List[LineSpec] = list(base.lines) if base else []
    fills: List[FillSpec] = list(base.fills) if base else []
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    count = sum(1 for line in lines if line.kind == "trace")
    count = _curves(panel.items, count, "left", lines, xs, ys)
    twin = panel.right
    y2s: List[np.ndarray] = []
    right_lines: List[LineSpec] = []
    if twin is not None:
        # the right curves continue the palette, so no two curves share a color by default
        _curves(twin.items, count, "right", right_lines, xs, y2s)

    settings = panel.settings
    xscale, yscale = settings.get("xscale", "linear"), settings.get("yscale", "linear")
    y2scale = twin.settings.get("yscale", "linear") if twin is not None else "linear"
    xrange = settings.get("xlim") or _merge(base.axes.xrange if base else None, _span(xs, 0.0, xscale == "log"))
    yrange = settings.get("ylim") or _merge(base.axes.yrange if base else None, _span(ys, 0.05, yscale == "log"))
    xrange = tuple(float(v) for v in (xrange or (0.0, 1.0)))
    yrange = tuple(float(v) for v in (yrange or (0.0, 1.0)))
    y2range: Optional[Tuple[float, float]] = None
    if twin is not None:
        y2range = tuple(float(v) for v in (twin.settings.get("ylim") or _span(y2s, 0.05, y2scale == "log") or (0.0, 1.0)))
        for kind, item in twin.items:
            if kind == "hline":
                _marker, dash, _ = parse_style(item["style"])
                right_lines.append(
                    LineSpec(np.array(xrange), np.array([item["at"], item["at"]], float), "hline",
                             _color(item["color"], 5), LW, dash, label=item["label"], axis="right")
                )
    for kind, item in panel.items:
        if kind in ("vline", "hline"):
            _marker, dash, _ = parse_style(item["style"])
            color = _color(item["color"], 5)
            if kind == "vline":
                x, y = np.array([item["at"], item["at"]], float), np.array(yrange)
            else:
                x, y = np.array(xrange), np.array([item["at"], item["at"]], float)
            lines.append(LineSpec(x, y, kind, color, LW, dash, label=item["label"]))
        elif kind == "vspan":
            x0, x1 = float(item["x0"]), float(item["x1"])
            color = _tint(_color(item["color"], 5))
            fills.append(
                FillSpec(np.array([x0, x1, x1, x0]), np.array([yrange[0], yrange[0], yrange[1], yrange[1]]), "vspan", color)
            )
            if item["label"]:
                # the band's legend entry, a wide line of its color
                lines.append(LineSpec(np.array([np.nan]), np.array([np.nan]), "vspan", color, 6.0, label=item["label"]))
    lines.extend(right_lines)
    x2label, x2ticks = None, ()
    if panel.secondary is not None:
        x2label = panel.secondary["label"] or ""
        x2ticks = _secondary_ticks(panel.secondary, xrange, xscale == "log")
    axes = AxesSpec(
        xscale=xscale,
        yscale=yscale,
        y2scale=y2scale,
        x2label=x2label,
        x2ticks=x2ticks,
        xlabel=settings.get("xlabel") or (base.axes.xlabel if base else ""),
        ylabel=settings.get("ylabel") or (base.axes.ylabel if base else ""),
        xrange=xrange,
        yrange=yrange,
        grid=not (base and base.rasters),
        y2label=(twin.settings.get("ylabel") or "") if twin is not None else None,
        y2range=y2range,
        ylabel_color=_one_color(lines, "left") if twin is not None else None,
        y2label_color=_one_color(lines, "right") if twin is not None else None,
    )
    return FigureSpec(
        axes=axes,
        rasters=base.rasters if base else (),
        fills=tuple(fills),
        lines=tuple(lines),
        texts=base.texts if base else (),
        colorbar=base.colorbar if base else None,
        title=settings.get("title"),
        width_mm=size[0],
        height_mm=size[1],
        font_size=options.font_size,
        dpi=options.dpi,
    )


def _nice_ticks(lo: float, hi: float, count: int = 5) -> List[float]:
    """Round tick values (1, 2 or 5 times a power of ten) across [lo, hi], about `count` of them."""
    if hi < lo:
        lo, hi = hi, lo
    raw = (hi - lo) / max(count, 1)
    if raw <= 0 or not np.isfinite(raw):
        return [lo]
    magnitude = 10.0 ** np.floor(np.log10(raw))
    step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw)
    first = np.ceil(lo / step - 1e-9) * step
    return [float(v) for v in np.arange(first, hi + step * 1e-9, step)]


def _secondary_ticks(
    secondary: dict, xrange: Tuple[float, float], log: bool
) -> Tuple[Tuple[float, str], ...]:
    """The top axis's ticks as ``(position on x, text)``, round values of ``function(x)``."""
    function, inverse = secondary["function"], secondary["inverse"]
    x = np.geomspace(*xrange, 512) if log and min(xrange) > 0 else np.linspace(*xrange, 512)
    s = np.asarray(function(x), float)
    values = secondary["ticks"]
    if values is None:
        values = _nice_ticks(float(np.nanmin(s)), float(np.nanmax(s)))
    values = np.asarray(values, float)
    if inverse is not None:
        positions = np.asarray(inverse(values), float)
    else:
        order = np.argsort(s)
        positions = np.interp(values, s[order], x[order], left=np.nan, right=np.nan)
    lo, hi = min(xrange), max(xrange)
    keep = np.isfinite(positions) & (positions >= lo * (1 - 1e-9)) & (positions <= hi * (1 + 1e-9))
    return tuple((float(p), f"{v:g}") for p, v in zip(positions[keep], values[keep]))


def _curves(
    items: List[Tuple[str, dict]],
    count: int,
    axis: str,
    lines: List[LineSpec],
    xs: List[np.ndarray],
    ys: List[np.ndarray],
) -> int:
    """Append the curves among `items` to `lines` on `axis`, their data to `xs` and `ys`.

    Returns the palette count after them.
    """
    for kind, item in items:
        if kind != "line":
            continue
        x = np.asarray(item["x"], float).ravel()
        y = np.real(np.asarray(item["y"])).astype(float).ravel()
        xs.append(x)
        ys.append(y)
        marker, dash, has_line = parse_style(item["style"])
        color = _color(item["color"], count)
        count += 1
        width = LW * 1.5 if item["width"] is None else float(item["width"])
        if has_line:
            lines.append(LineSpec(x, y, "user", color, width, dash, label=item["label"], axis=axis))
        if marker is not None:
            label = None if has_line else item["label"]
            lines.append(LineSpec(x, y, "user", color, 0.0, None, 1.0, marker, 4.0, label=label, axis=axis))
    return count


def _one_color(lines: List[LineSpec], axis: str) -> Optional[str]:
    """The color of every curve on `axis` when they share one, so its label can say which axis is whose."""
    colors = {line.color for line in lines if line.axis == axis and line.kind in ("user", "trace")}
    return colors.pop() if len(colors) == 1 else None


def _merge(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if a is None or b is None:
        return a or b
    return min(a[0], b[0]), max(a[1], b[1])


def compose_figure(figure: Any, options: PlotOptions) -> FigureSpec:
    """`figure` (an ``emode.Figure``) as a ``FigureSpec`` of its panels, row-major."""
    width, height = options.figure_size or figure.figure_size or (
        DEFAULT_SIZE_MM[0] * figure.cols,
        DEFAULT_SIZE_MM[1] * figure.rows,
    )
    cell = (float(width) / figure.cols, float(height) / figure.rows)
    panels = tuple(compose_panel(p, options, cell) for p in figure.panels)
    return FigureSpec(
        axes=panels[0].axes,
        panels=panels,
        panel_grid=(figure.rows, figure.cols),
        title=options.title or figure.title,
        width_mm=float(width),
        height_mm=float(height),
        font_size=options.font_size,
        dpi=options.dpi,
    )
